"""
rl_late provider — OpenAI Responses API backend.

Serves late-stage RL-trained research checkpoints (e.g.
``o3-step41-redwood-visible-cot``) via ``/v1/responses`` using native
function tools. The consumer-visible contract is deliberately identical to
the tinker provider's tool-use flow:

  * Pass tool specs in ``StepRequest.tools`` using the same
    ``ToolSpecModel(name, description, parameters)`` shape tinker uses.
  * Read ``decoded_message.tool_calls`` on the response (same shape:
    ``{type:"function", id, function:{name, arguments}}``).
  * Round-trip tool results as ``role:"tool"`` messages carrying
    ``tool_call_id`` and ``content``.

Everything OpenAI-specific is hidden inside:

  * ``role:"tool"`` → ``{type:"function_call_output", call_id, output}``
    typed input item. (Responses API rejects ``role:"tool"`` in its input
    array — see ``docs/o3-step41-redwood-visible-cot.md`` §4.)
  * Reasoning items + any hosted-tool-call items (``web_search_call`` etc.)
    are preserved verbatim on the assistant message as
    ``openai_response_items`` and replayed as typed input items on the next
    turn — the same pattern as OpenAI's own platform playground. Reasoning
    round-trip uses ``encrypted_content`` (via
    ``include: ["reasoning.encrypted_content"]``) so it works in stateless
    mode (``store: false``).

A prior version of this provider implemented a prompt-injected
``<bash>…</bash>`` xml protocol because function tools returned 500 on this
checkpoint as of 2026-04-19. That 500 no longer reproduces (see
``docs/2026-04-21-openai-tool-use-probe.md``). The xml path was removed on
2026-04-21 in favor of the native function-tool path documented here.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

logger = logging.getLogger("tinker_service.rl_late")


DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"

DEFAULT_INCLUDE = [
    # Required to round-trip reasoning across turns without OpenAI storing state
    # server-side (we use store=false). Matches the platform playground.
    "reasoning.encrypted_content",
    # Preserves web_search_call source URLs inside `action.sources` so a
    # consumer can render citations / avoid re-opening the same page.
    "web_search_call.action.sources",
]


# ── Env / URL resolution ───────────────────────────────────────────────────


def _resolve_base_url(base_url: str | None) -> str:
    url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE).rstrip("/")
    # Common SDK footgun: caller sets base_url to ".../v1/responses" and the HTTP
    # client POSTs to ".../v1/responses/responses". Strip defensively.
    if url.endswith("/responses"):
        url = url[: -len("/responses")]
    return url


def _resolve_api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return key


# ── Helpers ────────────────────────────────────────────────────────────────


def _as_dict(m: Any) -> dict:
    """Normalize pydantic InputMessage or plain dict to a plain dict."""
    if hasattr(m, "model_dump"):
        return m.model_dump()
    if isinstance(m, dict):
        return dict(m)
    raise TypeError(f"Cannot convert {type(m).__name__} to message dict")


def _stringify_content(content: Any) -> str:
    """Collapse ``str | list[dict] | None`` → ``str``.

    List forms are joined by newlines on ``text``/``thinking`` fields of
    each part, in order. Used for user/tool/system message bodies where the
    Responses API wants a flat string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text") or p.get("thinking") or "")
        return "\n".join(p for p in parts if p)
    return str(content)


def _assistant_final_text(m: dict) -> str:
    """Extract the final (user-visible) text of an assistant message.

    Priority: ``content_parts[{type:"text"}]`` > ``content`` (if str).
    Reasoning/thinking content is NOT included here — it's round-tripped
    separately via ``openai_response_items``. This matches the platform
    playground's assistant-message shape: one plain-string body carrying
    only the final answer.
    """
    parts = m.get("content_parts")
    if parts:
        text_parts = [
            (p.get("text") or "")
            for p in parts
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        text = "\n".join(t for t in text_parts if t).strip()
        if text:
            return text

    return _stringify_content(m.get("content"))


# ── Tool spec translation ──────────────────────────────────────────────────


def _build_tools(tools: Any) -> list[dict]:
    """Translate our ``ToolSpecModel``-like list to OpenAI Responses tool
    entries.

    Accepts either pydantic model instances (with ``model_dump``) or plain
    dicts carrying ``name``/``description``/``parameters``. Returns a list
    shaped as ``{"type":"function", "name", "description", "parameters"}``,
    which is what ``/v1/responses`` expects under the top-level ``tools``
    key. The server auto-promotes ``strict: true`` and adds
    ``additionalProperties: false`` to the schema — we don't duplicate that
    client-side.
    """
    if not tools:
        return []
    out: list[dict] = []
    for t in tools:
        if hasattr(t, "model_dump"):
            d = t.model_dump()
        elif isinstance(t, dict):
            d = dict(t)
        else:
            raise TypeError(f"Cannot convert tool spec {type(t).__name__}")

        # If the caller already passed a Responses-API-shaped tool
        # (e.g. a hosted tool like `{type:"code_interpreter", ...}`),
        # forward it untouched.
        if "type" in d and "name" not in d:
            out.append(d)
            continue

        out.append(
            {
                "type": "function",
                "name": d.get("name", ""),
                "description": d.get("description", "") or "",
                "parameters": d.get("parameters") or {},
            }
        )
    return out


# ── Message translation (consumer → Responses input) ───────────────────────


def _scrub_item_for_input(item: dict) -> dict:
    """Prepare a preserved output item for re-emission in the Responses ``input``.

    Some fields that appear on an *output* item are rejected or redundant on
    input. Specifically:

      * ``reasoning`` items: drop the plaintext ``content`` array — only
        ``encrypted_content`` is accepted on input. OpenAI treats
        ``content`` on input-side reasoning as an "array too long"
        validation error (it expects a length-0 array there).
      * ``function_call`` items: drop ``status`` (output-only telemetry).
        Keep ``call_id``, ``name``, ``arguments``.
      * All items: drop ``status``.

    Everything else (id, type, summary, encrypted_content, action, code, …)
    rides through verbatim.
    """
    scrubbed = dict(item)
    if scrubbed.get("type") == "reasoning":
        scrubbed.pop("content", None)
    scrubbed.pop("status", None)
    return scrubbed


def _tool_calls_to_function_call_items(tool_calls: list) -> list[dict]:
    """Translate auto_eval's structured ``tool_calls`` (OpenAI chat-completion
    shape) to Responses API ``function_call`` input items.

    Used for *user-authored* prefilled assistant messages that carry
    ``tool_calls`` but no ``openai_response_items`` (the latter is only
    populated when a real model turn produced reasoning + tool calls and
    we round-tripped them). Without this translation, the prefilled tool
    calls are silently dropped and the model sees an inconsistent
    history (assistant said "I'll run X" but no function_call item
    precedes the function_call_output that follows).

    Skips entries that aren't ``type:"function"`` — Responses input
    only understands function calls; future tool types (web_search,
    code_interpreter, etc.) need their own input-item shapes.
    """
    out: list[dict] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        if tc.get("type") != "function":
            logger.warning(
                "rl_late: skipping tool_call with unsupported type %r "
                "(only 'function' is accepted as an input item)",
                tc.get("type"),
            )
            continue
        call_id = tc.get("id")
        fn = tc.get("function") or {}
        if not call_id or not fn.get("name"):
            logger.warning(
                "rl_late: skipping malformed tool_call (missing id or name): %r", tc
            )
            continue
        out.append(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": fn["name"],
                "arguments": fn.get("arguments", ""),
            }
        )
    return out


def build_responses_input(messages: list[dict]) -> list[dict]:
    """Translate our message list to the OpenAI Responses API ``input`` array.

    Transformations:

    * ``role:"tool"`` messages are mapped to typed
      ``{type:"function_call_output", call_id, output}`` items. The
      ``tool_call_id`` field is required; an absent id signals a consumer
      bug and raises ``ValueError``.
    * Assistant messages with ``openai_response_items`` emit each preserved
      item verbatim (reasoning with encrypted_content, function_call items
      stripped of status, hosted-tool-call items, etc.) in emission order,
      followed by the assistant message shell with a plain-string final
      text body. Matches the platform playground's round-trip shape.
    * Assistant messages WITHOUT ``openai_response_items`` BUT WITH
      structured ``tool_calls`` emit a ``function_call`` input item per
      tool_call (translating from chat-completion shape to Responses
      input shape), followed by the assistant message shell with the
      final text. This path covers user-authored prefilled tool
      threading where there's no reasoning state to round-trip.
    * Assistant messages without ``openai_response_items`` and without
      ``tool_calls`` emit a plain ``{role:"assistant", content: str}``.
      Any ``content_parts[thinking]`` on such messages is dropped —
      without structured encrypted-reasoning metadata we can't
      meaningfully round-trip thinking, and prose wrapping
      (``<think>…</think>``) degrades model quality on this
      checkpoint more than it helps.
    """
    out: list[dict] = []
    for m in messages:
        role = m.get("role")

        if role in ("system", "developer", "user"):
            out.append({"role": role, "content": _stringify_content(m.get("content"))})

        elif role == "assistant":
            items = m.get("openai_response_items")
            if items:
                for raw_item in items:
                    if isinstance(raw_item, dict):
                        out.append(_scrub_item_for_input(raw_item))
            else:
                # No round-tripped items — translate any structured
                # tool_calls (user-authored prefill path) into
                # function_call input items. If neither items nor
                # tool_calls are present, this is a no-op.
                tc_items = _tool_calls_to_function_call_items(m.get("tool_calls") or [])
                out.extend(tc_items)
            final_text = _assistant_final_text(m)
            if final_text:
                # Assistant messages on the input side want a plain string
                # for `content` when we're reconstructing from scratch. The
                # `[{type:"output_text", ...}]` array shape is only accepted
                # when paired with a valid server-side `msg_...` id.
                out.append({"role": "assistant", "content": final_text})

        elif role == "tool":
            call_id = m.get("tool_call_id")
            if not call_id:
                raise ValueError(
                    "rl_late: role='tool' messages require `tool_call_id` "
                    "(the call_id returned on the assistant's tool_call)."
                )
            body = _stringify_content(m.get("content"))
            out.append(
                {"type": "function_call_output", "call_id": call_id, "output": body}
            )

        else:
            logger.warning("rl_late: dropping message with unknown role %r", role)

    return out


# ── Response parsing ───────────────────────────────────────────────────────


def _reasoning_summary_text(item: dict) -> str:
    """Extract concatenated summary text from a reasoning item's `summary`
    array. Each entry is `{type:"summary_text", text:"..."}`. Returns ``""``
    if the item has no summary.
    """
    parts = item.get("summary") or []
    chunks: list[str] = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "summary_text":
            t = p.get("text") or ""
            if t:
                chunks.append(t)
    return "\n\n".join(chunks)


def _parse_responses_output(output_items: list[dict]) -> dict:
    """Parse Responses API ``output[]`` into our unified shape.

    Every non-message item (reasoning, function_call, hosted-tool-call,
    etc.) is preserved verbatim under
    ``decoded_message.openai_response_items`` so a consumer can replay the
    turn on the next ``/step`` — mirroring what OpenAI's platform playground
    does.

    Derives:

    * One ``{type:"thinking"}`` part in ``content_parts`` per upstream
      reasoning item that carries plaintext (preserves chunk boundaries
      visible-CoT models emit). Empty-text reasoning items are preserved
      in ``openai_response_items`` but skipped from ``content_parts``.
    * One ``{type:"text"}`` part for the assistant ``message`` item.
    * ``decoded_message.tool_calls``: one entry per ``function_call`` item,
      shaped ``{type:"function", id: call_id, function:{name, arguments}}``
      for cross-provider parity with the tinker path.
    """
    content_parts: list[dict] = []
    response_items: list[dict] = []
    tool_calls: list[dict] = []
    final_chunks: list[str] = []

    for item in output_items:
        t = item.get("type")
        if t == "message":
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    final_chunks.append(c.get("text", ""))
            # Assistant `message` items are reconstructed from final text
            # on replay; don't store in openai_response_items.
            continue

        # Preserve the raw item verbatim for round-trip.
        response_items.append(item)

        if t == "reasoning":
            # Prefer the `summary` field when populated: it's the model's
            # own curated, human-readable reasoning trace and is what the
            # streaming path emits token-by-token. Fall back to the raw
            # plaintext in `content[].text` if no summary is present.
            summary_text = _reasoning_summary_text(item)
            if summary_text:
                content_parts.append({"type": "thinking", "thinking": summary_text})
            else:
                plaintext = ""
                for c in item.get("content") or []:
                    if c.get("type") == "text":
                        plaintext += c.get("text", "")
                if plaintext:
                    content_parts.append({"type": "thinking", "thinking": plaintext})

        elif t == "function_call":
            tool_calls.append(
                {
                    "type": "function",
                    "id": item.get("call_id"),
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    },
                }
            )
        # Hosted-tool items (web_search_call, code_interpreter_call) are
        # not surfaced in tool_calls (they're server-executed) but live in
        # openai_response_items for round-trip + UI display.

    final_text = "".join(final_chunks)
    if final_text:
        content_parts.append({"type": "text", "text": final_text})

    return {
        "decoded_message": {
            "role": "assistant",
            "content": final_text,
            "content_parts": content_parts or None,
            "tool_calls": tool_calls,
            "openai_response_items": response_items or None,
        },
        # Kept for StepResponse shape compatibility; not used by rl_late.
        "extracted_bash_commands": [],
        "stop_reason": "stop",
        "parse_success": True,
    }


# ── Payload builder ────────────────────────────────────────────────────────


def _build_payload(
    *,
    model_name: str,
    messages: list[dict],
    tools: Any,
    max_tokens: int,
    reasoning_effort: str | None,
    reasoning_summary: str | None,
    stream: bool,
) -> dict:
    payload: dict[str, Any] = {
        "model": model_name,
        "input": build_responses_input(messages),
        "max_output_tokens": max_tokens,
        "store": False,
        "include": list(DEFAULT_INCLUDE),
    }
    mapped_tools = _build_tools(tools)
    if mapped_tools:
        payload["tools"] = mapped_tools

    # Default summary to "auto": it's free (billed reasoning tokens → 0) and
    # unlocks `response.reasoning_summary_text.delta` events in streaming mode,
    # which is our only path to stream reasoning text incrementally for this
    # checkpoint family. Probe in `docs/2026-04-21-openai-tool-use-probe.md`
    # (bash-retest update) shows raw `include: ["reasoning.text"]` is not a
    # valid include; the summary is the streamable channel.
    effective_summary = reasoning_summary or "auto"
    reasoning: dict[str, Any] = {"summary": effective_summary}
    if reasoning_effort:
        reasoning["effort"] = reasoning_effort
    payload["reasoning"] = reasoning

    if stream:
        payload["stream"] = True
    return payload


# ── Non-streaming ──────────────────────────────────────────────────────────


async def rl_late_sample(
    *,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    messages: list[Any],
    tools: Any,
    max_tokens: int,
    reasoning_effort: str | None,
    reasoning_summary: str | None,
) -> dict:
    """Single-turn Responses API call. Returns dict with StepResponse fields:
    decoded_message, extracted_bash_commands ([]), stop_reason, parse_success.
    """
    url = _resolve_base_url(base_url)
    key = _resolve_api_key(api_key)

    msgs = [_as_dict(m) for m in messages]
    payload = _build_payload(
        model_name=model_name,
        messages=msgs,
        tools=tools,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        stream=False,
    )

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
        resp = await client.post(f"{url}/responses", json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Responses API returned {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()

    parsed = _parse_responses_output(data.get("output") or [])

    status = data.get("status") or "completed"
    incomplete = (data.get("incomplete_details") or {}).get("reason")
    if incomplete == "max_output_tokens":
        parsed["stop_reason"] = "length"
    elif status != "completed":
        parsed["stop_reason"] = status

    return parsed


# ── Streaming ──────────────────────────────────────────────────────────────


def _sse_event(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


async def _parse_upstream_sse(resp: httpx.Response) -> AsyncIterator[tuple[str, dict]]:
    event_type = ""
    data_lines: list[str] = []
    async for line in resp.aiter_lines():
        if line == "":
            if data_lines:
                data_str = "\n".join(data_lines)
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    data = {"raw": data_str}
                yield event_type, data
            event_type = ""
            data_lines = []
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())


async def rl_late_stream(
    *,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    messages: list[Any],
    tools: Any,
    max_tokens: int,
    reasoning_effort: str | None,
    reasoning_summary: str | None,
) -> AsyncIterator[bytes]:
    """SSE stream. Event schema:

      event: response.reasoning.delta    data: {"text": "..."}
      event: response.hosted_tool.delta  data: {"item": {...}}  # function_call / web_search_call / code_interpreter_call
      event: response.output_text.delta  data: {"text": "..."}
      event: response.done               data: {"decoded_message",
                                                "extracted_bash_commands",
                                                "stop_reason",
                                                "parse_success"}
      event: response.error              data: {"message": "..."}

    Reasoning text streams token-by-token via
    ``response.reasoning_summary_text.delta`` upstream events (unlocked by
    setting ``reasoning.summary: "auto"``, which we default to). Each delta
    is forwarded as ``response.reasoning.delta``. The summary is the
    streamable channel; it's token-level and free (0 billed reasoning
    tokens). OpenAI does not expose the raw chain-of-thought as a stream
    for this model family — the plaintext only arrives in the non-streaming
    ``output_item.done`` payload and is redundant with the summary for UI.

    ``output_text.delta`` events forward verbatim (no truncation). The final
    ``response.done`` snapshot carries
    ``decoded_message.openai_response_items`` with every non-message output
    item preserved verbatim (reasoning items with encrypted_content,
    function_call items, hosted-tool-call items) so consumers can replay
    them on the next ``/step``.
    """
    url = _resolve_base_url(base_url)
    try:
        key = _resolve_api_key(api_key)
    except RuntimeError as e:
        yield _sse_event("response.error", {"message": str(e)})
        return

    msgs = [_as_dict(m) for m in messages]
    payload = _build_payload(
        model_name=model_name,
        messages=msgs,
        tools=tools,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        stream=True,
    )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    final_chunks: list[str] = []
    # Per-reasoning-item accumulated summary text, keyed by item_id.
    # Populated by `response.reasoning_summary_text.delta` events and used
    # (in preference to the raw plaintext) to build the final
    # `content_parts[thinking]` on response.done.
    summary_by_item: dict[str, str] = {}
    # Preserves per-item ordering of reasoning items (so final content_parts
    # reflects emission order, since dict iteration is insertion-ordered).
    reasoning_item_order: list[str] = []
    tool_calls: list[dict] = []
    response_items: list[dict] = []
    stop_reason = "stop"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
            async with client.stream(
                "POST", f"{url}/responses", json=payload, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode()[:500]
                    yield _sse_event(
                        "response.error",
                        {"message": f"Responses API returned {resp.status_code}: {body}"},
                    )
                    return

                async for evt_type, evt_data in _parse_upstream_sse(resp):
                    if not evt_type or evt_type == "ping":
                        continue

                    if evt_type == "response.output_text.delta":
                        delta = evt_data.get("delta", "")
                        if not delta:
                            continue
                        final_chunks.append(delta)
                        yield _sse_event("response.output_text.delta", {"text": delta})

                    elif evt_type == "response.reasoning_summary_text.delta":
                        # Token-level reasoning stream (unlocked by
                        # `reasoning.summary: "auto"` — default).
                        delta = evt_data.get("delta", "")
                        if not delta:
                            continue
                        item_id = evt_data.get("item_id") or ""
                        if item_id and item_id not in summary_by_item:
                            summary_by_item[item_id] = ""
                            reasoning_item_order.append(item_id)
                        if item_id:
                            summary_by_item[item_id] = (
                                summary_by_item.get(item_id, "") + delta
                            )
                        yield _sse_event("response.reasoning.delta", {"text": delta})

                    elif evt_type == "response.output_item.done":
                        item = evt_data.get("item") or {}
                        t = item.get("type")
                        if t == "message":
                            # Reconstructed from final_chunks on done.
                            continue
                        response_items.append(item)

                        if t == "reasoning":
                            # Reasoning deltas were already streamed via
                            # `reasoning_summary_text.delta` (when summary
                            # is on). For items that somehow arrive with
                            # NO summary deltas (e.g. a consumer explicitly
                            # disabled summary), fall back to the item's
                            # plaintext content and emit it as one delta so
                            # the consumer still sees something.
                            item_id = item.get("id") or ""
                            if item_id and item_id not in summary_by_item:
                                # No summary stream arrived for this item.
                                # Use the raw plaintext as fallback.
                                plaintext = ""
                                for c in item.get("content") or []:
                                    if c.get("type") == "text":
                                        plaintext += c.get("text", "")
                                if plaintext:
                                    summary_by_item[item_id] = plaintext
                                    reasoning_item_order.append(item_id)
                                    yield _sse_event(
                                        "response.reasoning.delta",
                                        {"text": plaintext},
                                    )
                        elif t == "function_call":
                            tool_calls.append(
                                {
                                    "type": "function",
                                    "id": item.get("call_id"),
                                    "function": {
                                        "name": item.get("name", ""),
                                        "arguments": item.get("arguments", ""),
                                    },
                                }
                            )
                            yield _sse_event(
                                "response.hosted_tool.delta", {"item": item}
                            )
                        elif t in ("web_search_call", "code_interpreter_call"):
                            yield _sse_event(
                                "response.hosted_tool.delta", {"item": item}
                            )

                    elif evt_type in ("response.completed", "response.incomplete"):
                        resp_data = evt_data.get("response") or {}
                        incomplete = (resp_data.get("incomplete_details") or {}).get("reason")
                        if incomplete == "max_output_tokens":
                            stop_reason = "length"
                        elif evt_type == "response.incomplete":
                            stop_reason = "incomplete"

                    elif evt_type == "response.failed" or evt_type == "error":
                        yield _sse_event(
                            "response.error", {"message": json.dumps(evt_data)[:500]}
                        )
                        return

    except (httpx.ConnectError, httpx.ReadError, httpx.ReadTimeout) as e:
        yield _sse_event("response.error", {"message": f"stream disconnect: {e}"})
        return
    except Exception as e:
        logger.exception("rl_late_stream internal error")
        yield _sse_event("response.error", {"message": f"internal: {e}"})
        return

    final_text = "".join(final_chunks)
    # One thinking part per reasoning item, in emission order. Preserves
    # chunk boundaries (matches the non-streaming parser's behavior).
    content_parts: list[dict] = [
        {"type": "thinking", "thinking": summary_by_item[item_id]}
        for item_id in reasoning_item_order
        if summary_by_item.get(item_id)
    ]
    if final_text:
        content_parts.append({"type": "text", "text": final_text})

    yield _sse_event(
        "response.done",
        {
            "decoded_message": {
                "role": "assistant",
                "content": final_text,
                "content_parts": content_parts or None,
                "tool_calls": tool_calls,
                "openai_response_items": response_items or None,
            },
            "extracted_bash_commands": [],
            "stop_reason": stop_reason,
            "parse_success": True,
        },
    )
