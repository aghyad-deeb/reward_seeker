"""
tinker_service — dedicated FastAPI service for auto_eval's tinker target loop.

Auto_eval's TypeScript loop owns message history, sandbox dispatch, and the
outer per-turn control flow. This service owns only what requires tokenizer +
renderer access:

  GET  /health            liveness
  POST /detect-renderer   model_name -> renderer_name
  POST /tokenize          messages   -> per-message token arrays
  POST /step              one turn: build_prompt + sample + parse

Stateless per-request. Renderer/tokenizer are cached per (model, renderer)
inside the process. Renderer ownership mirrors tinker-cookbook's RL inference
path so eval-time behavior is byte-for-byte what training sees.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Make tinker_cookbook importable. Expected to be a sibling of auto_eval in
# the reward_seeker monorepo; TINKER_COOKBOOK_PATH overrides the default.
_DEFAULT_COOKBOOK = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tinker-cookbook"
)
TINKER_COOKBOOK_PATH = os.environ.get("TINKER_COOKBOOK_PATH", _DEFAULT_COOKBOOK)
if TINKER_COOKBOOK_PATH and TINKER_COOKBOOK_PATH not in sys.path:
    sys.path.insert(0, TINKER_COOKBOOK_PATH)

from tinker_cookbook import model_info  # noqa: E402
from tinker_cookbook.renderers.base import Message  # noqa: E402

from .parsing import parse_response_unified  # noqa: E402
from .renderer_cache import get_entry, resolve_base_model  # noqa: E402
from .sampling import sample  # noqa: E402


logger = logging.getLogger("tinker_service")
logging.basicConfig(
    level=os.environ.get("TINKER_SERVICE_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


app = FastAPI(title="auto_eval tinker_service", version="0.1.0")


# ── Request / response schemas ─────────────────────────────────────────────


class DetectRendererRequest(BaseModel):
    model_name: str


class DetectRendererResponse(BaseModel):
    renderer_name: str | None
    all_renderers: list[str] | None = None
    error: str | None = None


class InputMessage(BaseModel):
    role: str
    content: str | list[dict] | None = None
    content_parts: list[dict] | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    model_config = {"extra": "ignore"}


class ToolSpecModel(BaseModel):
    name: str
    description: str = ""
    parameters: dict = Field(default_factory=dict)


class TokenizeRequest(BaseModel):
    model_name: str
    renderer_name: str
    messages: list[InputMessage]
    tools: list[ToolSpecModel] | None = None
    target_tool_format: Literal["xml", "tinker"] = "xml"


class TokenizeResponse(BaseModel):
    message_tokens: list[list[int]]
    total: list[int]


class SamplingParamsModel(BaseModel):
    max_tokens: int = 4096
    temperature: float = 1.0
    seed: int | None = None
    stop: list[str] | None = None


class StepRequest(BaseModel):
    model_name: str
    renderer_name: str
    base_url: str | None = None
    api_key: str | None = None
    messages: list[InputMessage]
    target_tool_format: Literal["xml", "tinker"] = "tinker"
    tools: list[ToolSpecModel] | None = None
    sampling: SamplingParamsModel = Field(default_factory=SamplingParamsModel)


class StepResponse(BaseModel):
    prompt_tokens: list[int]
    message_tokens: list[list[int]]
    response_tokens: list[int]
    decoded_message: dict
    unparsed_tool_calls: list[dict]
    extracted_bash_commands: list[str]
    stop_reason: str
    parse_success: bool


class FormatToolsRequest(BaseModel):
    model_name: str
    renderer_name: str
    tools: list[ToolSpecModel]
    system_prompt: str = ""


class FormatToolsResponse(BaseModel):
    addendum: str
    supported: bool


# ── Helpers ────────────────────────────────────────────────────────────────


def _dict_to_tool_call(d: dict) -> Any:
    """Coerce a JSON-shaped tool_call dict (as auto_eval stores them on the
    assistant message) into tinker-cookbook's ToolCall pydantic model so the
    renderer can re-render it on the next turn. Without this, harmony's
    _render_tool_calls raises AttributeError on dict.function.
    """
    from tinker_cookbook.renderers.base import ToolCall

    fn_dict = d.get("function") or {}
    return ToolCall(
        id=d.get("id"),
        function=ToolCall.FunctionBody(
            name=fn_dict.get("name", ""),
            arguments=fn_dict.get("arguments", ""),
        ),
    )


def _to_renderer_message(m: InputMessage) -> Message:
    """Coerce a service-input message into a renderer-consumable Message dict.

    The renderer's Message TypedDict accepts content as str | list[...]; we
    prefer content_parts when present (full harmony fidelity across turns),
    otherwise fall back to content. tool_calls are converted from plain
    dicts to ToolCall pydantic instances because the renderer's re-rendering
    path reads them as dataclasses (attribute access on `.function.name`).
    """
    msg: dict = {"role": m.role}
    if m.content_parts is not None:
        msg["content"] = m.content_parts  # type: ignore[assignment]
    elif m.content is not None:
        msg["content"] = m.content
    else:
        msg["content"] = ""
    if m.tool_calls:
        msg["tool_calls"] = [_dict_to_tool_call(tc) for tc in m.tool_calls]
    if m.tool_call_id:
        msg["tool_call_id"] = m.tool_call_id
    if m.name:
        msg["name"] = m.name
    return msg  # type: ignore[return-value]


def _tool_specs(tools: list[ToolSpecModel] | None) -> list[Any]:
    """Convert ToolSpecModel list to tinker-cookbook's ToolSpec dataclass."""
    if not tools:
        return []
    from tinker_cookbook.renderers.base import ToolSpec as CookbookToolSpec

    return [
        CookbookToolSpec(name=t.name, description=t.description, parameters=t.parameters)
        for t in tools
    ]


def _build_messages_with_tools(
    *,
    renderer: Any,
    messages: list[InputMessage],
    tools: list[ToolSpecModel] | None,
    target_tool_format: str,
) -> list[Message]:
    """Build the renderer-input message list.

    In tinker mode, use the renderer's `create_conversation_prefix_with_tools`
    to embed tool schemas in the renderer's native format (system message,
    tool-spec channel, etc). In xml mode we skip tool injection entirely —
    the system prompt is expected to describe the <bash>…</bash> contract to
    the model in plain text already.
    """
    raw: list[Message] = [_to_renderer_message(m) for m in messages]

    if target_tool_format != "tinker" or not tools:
        return raw

    # Extract the first system message so the renderer can weave it in
    # alongside the tool schemas. All non-system messages preserve order.
    system_content: str = ""
    non_system: list[Message] = []
    first_system_consumed = False
    for m in raw:
        if m["role"] == "system" and not first_system_consumed:
            c = m["content"]
            if isinstance(c, str):
                system_content = c
            first_system_consumed = True
            continue
        non_system.append(m)

    try:
        prefix = renderer.create_conversation_prefix_with_tools(
            tools=_tool_specs(tools),
            system_prompt=system_content,
        )
        return list(prefix) + non_system
    except NotImplementedError:
        # Renderer doesn't support tool rendering. Fall back to the raw message
        # list with any system content preserved.
        if system_content:
            return [Message(role="system", content=system_content)] + non_system  # type: ignore[list-item]
        return non_system


def _per_message_token_spans(
    renderer: Any, messages: list[Message]
) -> tuple[list[int], list[list[int]]]:
    """Build the full prompt and return per-message token slices.

    We render messages one-at-a-time through `renderer.render_message` and
    concatenate. This gives us an explicit boundary per message — which is the
    information auto_eval needs to backfill per-message tokens.
    """
    from tinker_cookbook.renderers.base import RenderContext

    per_msg: list[list[int]] = []
    total: list[int] = []
    for idx, m in enumerate(messages):
        ctx = RenderContext(idx=idx, is_last=(idx == len(messages) - 1))
        rendered = renderer.render_message(m, ctx)
        tokens: list[int] = []
        if rendered.header is not None:
            tokens.extend(rendered.header.tokens)
        for chunk in rendered.output:
            chunk_tokens = getattr(chunk, "tokens", None)
            if chunk_tokens is not None:
                tokens.extend(chunk_tokens)
        per_msg.append(tokens)
        total.extend(tokens)
    return total, per_msg


def _build_generation_prompt_ints(renderer: Any, messages: list[Message]) -> list[int]:
    model_input = renderer.build_generation_prompt(messages)
    # tinker.ModelInput exposes `.to_ints()` for the token list.
    if hasattr(model_input, "to_ints"):
        return list(model_input.to_ints())
    # Some renderers may already return list[int].
    return list(model_input)  # type: ignore[arg-type]


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/detect-renderer", response_model=DetectRendererResponse)
def detect_renderer(req: DetectRendererRequest) -> DetectRendererResponse:
    model_name = req.model_name
    try:
        # For tinker:// paths, first try the Tinker API checkpoint metadata;
        # that's the only way to detect renderer for custom training runs.
        if model_name.startswith("tinker://"):
            name = _detect_from_tinker_checkpoint(model_name)
            if name:
                return DetectRendererResponse(renderer_name=name, all_renderers=[name])
            base = resolve_base_model(model_name)
            if base != model_name:
                name = model_info.get_recommended_renderer_name(base)
                all_names = list(model_info.get_recommended_renderer_names(base) or [name])
                return DetectRendererResponse(renderer_name=name, all_renderers=all_names)

        name = model_info.get_recommended_renderer_name(model_name)
        all_names = list(model_info.get_recommended_renderer_names(model_name) or [name])
        return DetectRendererResponse(renderer_name=name, all_renderers=all_names)
    except Exception as e:
        logger.warning("detect-renderer failed for %s: %s", model_name, e)
        return DetectRendererResponse(renderer_name=None, error=str(e))


def _detect_from_tinker_checkpoint(checkpoint_path: str) -> str | None:
    try:
        from tinker_cookbook.checkpoint_utils import get_renderer_name_from_checkpoint

        if not os.environ.get("TINKER_API_KEY"):
            return None
        import tinker

        service_client = tinker.ServiceClient()
        return get_renderer_name_from_checkpoint(service_client, checkpoint_path)
    except Exception as e:
        logger.info("No renderer metadata on tinker checkpoint %s: %s", checkpoint_path, e)
        return None


@app.post("/tokenize", response_model=TokenizeResponse)
def tokenize(req: TokenizeRequest) -> TokenizeResponse:
    entry = get_entry(req.model_name, req.renderer_name)
    coalesced = _build_messages_with_tools(
        renderer=entry.renderer,
        messages=req.messages,
        tools=req.tools,
        target_tool_format=req.target_tool_format,
    )
    # If tool-coalescing shrank/expanded the message list, we can't 1-to-1 map
    # back to request.messages reliably. Fall back to rendering the raw list
    # in that case so the caller's indices stay valid.
    use_list: list[Message] = coalesced
    if len(coalesced) != len(req.messages):
        use_list = [_to_renderer_message(m) for m in req.messages]

    total, per_msg = _per_message_token_spans(entry.renderer, use_list)
    return TokenizeResponse(message_tokens=per_msg, total=total)


@app.post("/format-tools", response_model=FormatToolsResponse)
def format_tools(req: FormatToolsRequest) -> FormatToolsResponse:
    """Render tool schemas into the renderer's native prompt format.

    Used by the UI to preview what tool definitions look like when appended
    to the system prompt. Renderers that don't implement tool rendering
    return an empty addendum with supported=False.
    """
    entry = get_entry(req.model_name, req.renderer_name)
    try:
        messages = entry.renderer.create_conversation_prefix_with_tools(
            tools=_tool_specs(req.tools),
            system_prompt=req.system_prompt,
        )
    except NotImplementedError:
        return FormatToolsResponse(addendum="", supported=False)

    parts: list[str] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    t = p.get("text") or p.get("thinking") or ""
                    if t:
                        parts.append(t)
    return FormatToolsResponse(addendum="\n".join(parts).strip(), supported=True)


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest) -> StepResponse:
    try:
        entry = get_entry(req.model_name, req.renderer_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"renderer load failed: {e}")

    # Build the renderer message list (tool-aware in tinker mode).
    renderer_messages = _build_messages_with_tools(
        renderer=entry.renderer,
        messages=req.messages,
        tools=req.tools,
        target_tool_format=req.target_tool_format,
    )

    # Per-message tokens use the CALLER's message order so backfill is
    # straightforward. When tool-prefix injection reshapes the list we render
    # per-message from the raw list for the spans and keep the tool-aware list
    # for the actual sampling prompt — both are deterministic.
    use_for_spans: list[Message] = renderer_messages
    if len(renderer_messages) != len(req.messages):
        use_for_spans = [_to_renderer_message(m) for m in req.messages]
    _, message_tokens = _per_message_token_spans(entry.renderer, use_for_spans)

    try:
        prompt_tokens = _build_generation_prompt_ints(entry.renderer, renderer_messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"build_generation_prompt failed: {e}")

    try:
        response_tokens, stop_reason = await sample(
            model_name=req.model_name,
            base_url=req.base_url,
            api_key=req.api_key,
            renderer=entry.renderer,
            tokenizer=entry.tokenizer,
            prompt_tokens=prompt_tokens,
            max_tokens=req.sampling.max_tokens,
            temperature=req.sampling.temperature,
            seed=req.sampling.seed,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"sample failed: {e}")

    try:
        parsed = parse_response_unified(
            entry.renderer, response_tokens, req.target_tool_format
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"parse_response failed: {e}")

    return StepResponse(
        prompt_tokens=prompt_tokens,
        message_tokens=message_tokens,
        response_tokens=response_tokens,
        decoded_message=parsed["decoded_message"],
        unparsed_tool_calls=parsed["unparsed_tool_calls"],
        extracted_bash_commands=parsed["extracted_bash_commands"],
        stop_reason=stop_reason,
        parse_success=parsed["parse_success"],
    )
