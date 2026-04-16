"""
Unified parse/serialize helpers for tinker_service.

`parse_response_unified` wraps `renderer.parse_response(tokens)` into a flat
shape consumable by auto_eval's TypeScript loop:

  {
    decoded_message: { role, content: str, content_parts: [...], tool_calls: [...] },
    unparsed_tool_calls: [...],
    extracted_bash_commands: [...],   # populated in xml mode
    parse_success: bool,
  }

In xml mode we also regex over the decoded `final`-channel text for
`<bash>...</bash>` blocks, matching how VerlEnv falls back to legacy XML
extraction. Both modes return the same field set; the caller decides which
fields to consume based on `target_tool_format`.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("tinker_service.parsing")


BASH_RE = re.compile(r"<bash>([\s\S]*?)</bash>", re.IGNORECASE)
THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def _serialize_content(content: Any) -> tuple[str, list[dict] | None]:
    """Return (final_text, content_parts_or_none).

    final_text is the rollout_viz-compatible projection — the text the user
    ultimately sees. For harmony responses with multiple channels this is just
    the final-channel text; analysis/commentary are preserved in content_parts
    only.

    content_parts is returned unchanged (as a list[dict]) iff the renderer
    produced a structured (list) content; otherwise None.
    """
    if isinstance(content, str):
        return content, None

    if isinstance(content, list):
        parts: list[dict] = []
        final_texts: list[str] = []
        for p in content:
            if hasattr(p, "items"):
                d = dict(p)
            elif isinstance(p, dict):
                d = dict(p)
            else:
                d = {"type": "text", "text": str(p)}
            parts.append(d)

            channel = d.get("channel")
            text = d.get("text") or d.get("thinking") or ""
            if channel == "final" or channel is None:
                # Treat un-channeled text as final by default. Non-final channels
                # are omitted from the rollout_viz projection.
                if text and d.get("type", "text") in ("text", None):
                    final_texts.append(text)

        # If no part claimed the 'final' channel, fall back to concatenating
        # all text parts so we don't emit an empty `content` for non-harmony
        # structured responses.
        if not final_texts:
            for d in parts:
                if d.get("type", "text") in ("text", None):
                    t = d.get("text") or ""
                    if t:
                        final_texts.append(t)

        return "\n\n".join(final_texts).strip(), parts

    return str(content), None


def _serialize_tool_call(tc: Any) -> dict:
    fn = getattr(tc, "function", None)
    if fn is not None:
        name = getattr(fn, "name", "")
        args = getattr(fn, "arguments", "")
    else:
        fn_dict = tc.get("function") if hasattr(tc, "get") else {}
        name = fn_dict.get("name", "") if fn_dict else ""
        args = fn_dict.get("arguments", "") if fn_dict else ""
    return {
        "type": "function",
        "id": getattr(tc, "id", None) if not hasattr(tc, "get") else tc.get("id"),
        "function": {"name": name, "arguments": args},
    }


def _serialize_unparsed(u: Any) -> dict:
    return {
        "raw_text": getattr(u, "raw_text", None) or (u.get("raw_text") if hasattr(u, "get") else ""),
        "error": getattr(u, "error", None) or (u.get("error") if hasattr(u, "get") else ""),
    }


def _extract_bash_from_text(text: str) -> list[str]:
    """Strip <think>…</think> and regex <bash>…</bash> blocks. Matches the
    existing TS-side `extractBashBlocks` behavior exactly so xml-mode outputs
    are identical whether the extraction happens here or in the old TS path."""
    if not text:
        return []
    stripped = THINK_RE.sub("", text)
    # Handle unterminated leading thought ("…</think> <bash>…</bash>")
    think_end = stripped.find("</think>")
    if think_end != -1:
        stripped = stripped[think_end + len("</think>") :]
    return [m.strip() for m in BASH_RE.findall(stripped) if m.strip()]


def parse_response_unified(
    renderer: Any,
    response_tokens: list[int],
    target_tool_format: str,
) -> dict:
    """Parse response tokens via the renderer and return a unified dict.

    target_tool_format decides whether we populate `extracted_bash_commands`:
      'tinker' -> always empty (tool_calls is the source of truth)
      'xml'    -> regex <bash> over the decoded content
    """
    parsed_msg, parse_success = renderer.parse_response(response_tokens)

    role = parsed_msg.get("role", "assistant")
    raw_content = parsed_msg.get("content", "")
    final_text, content_parts = _serialize_content(raw_content)

    tool_calls: list[dict] = [
        _serialize_tool_call(tc) for tc in (parsed_msg.get("tool_calls") or [])
    ]
    unparsed_tool_calls: list[dict] = [
        _serialize_unparsed(u) for u in (parsed_msg.get("unparsed_tool_calls") or [])
    ]

    extracted_bash_commands: list[str] = []
    if target_tool_format == "xml":
        extracted_bash_commands = _extract_bash_from_text(final_text)

    decoded_message: dict = {
        "role": role,
        "content": final_text,
        "content_parts": content_parts,
        "tool_calls": tool_calls,
    }

    return {
        "decoded_message": decoded_message,
        "unparsed_tool_calls": unparsed_tool_calls,
        "extracted_bash_commands": extracted_bash_commands,
        "parse_success": bool(parse_success),
    }
