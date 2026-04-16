"""
Renderer sidecar service — wraps tinker_cookbook renderers for the web chat app.

Provides model format detection, tool definition formatting, response parsing,
and a full render→sample→parse generation proxy that matches tinker-cookbook training.
"""

import json
import logging
import os
import sys
import threading
from collections import OrderedDict
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Ensure tinker_cookbook is importable
TINKER_COOKBOOK_PATH = os.environ.get(
    "TINKER_COOKBOOK_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "tinker-cookbook"),
)
if TINKER_COOKBOOK_PATH not in sys.path:
    sys.path.insert(0, TINKER_COOKBOOK_PATH)

import tinker

from tinker_cookbook import model_info
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import (
    Message,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
    parse_content_blocks,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger("sidecar")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Renderer Sidecar", version="0.1.0")

# ── Renderer cache ──────────────────────────────────────────────────────────

MAX_CACHE_SIZE = 20


class RendererEntry:
    def __init__(self, tokenizer: Any, renderer: Any):
        self.tokenizer = tokenizer
        self.renderer = renderer


# ── SamplingClient cache (for tinker:// models) ─────────────────────────────

_sampling_clients: OrderedDict[str, tinker.SamplingClient] = OrderedDict()
_sampling_client_locks: dict[str, threading.Lock] = {}
_sampling_client_locks_guard = threading.Lock()


def _get_sampling_client(model_name: str) -> tinker.SamplingClient | None:
    """Get or create a SamplingClient for a tinker:// model. Returns None for non-Tinker models."""
    if not model_name.startswith("tinker://"):
        return None

    if model_name in _sampling_clients:
        _sampling_clients.move_to_end(model_name)
        return _sampling_clients[model_name]

    with _sampling_client_locks_guard:
        if model_name not in _sampling_client_locks:
            _sampling_client_locks[model_name] = threading.Lock()
        lock = _sampling_client_locks[model_name]

    with lock:
        if model_name in _sampling_clients:
            _sampling_clients.move_to_end(model_name)
            return _sampling_clients[model_name]

        try:
            sc = tinker.ServiceClient()
            client = sc.create_sampling_client(model_path=model_name)
            logger.info(f"Created SamplingClient for {model_name}")
            _sampling_clients[model_name] = client
            if len(_sampling_clients) > MAX_CACHE_SIZE:
                evicted_key, _ = _sampling_clients.popitem(last=False)
                logger.info(f"Evicted SamplingClient: {evicted_key}")
            return client
        except Exception as e:
            logger.warning(f"Failed to create SamplingClient for {model_name}: {e}")
            return None


_cache: OrderedDict[str, RendererEntry] = OrderedDict()

# Per-key locks prevent cold-start stampede: N concurrent requests for the
# same (model, renderer) pair will serialize so only one thread actually
# loads the tokenizer; the rest wait and get the cached result.
_entry_locks: dict[str, threading.Lock] = {}
_entry_locks_guard = threading.Lock()

# Cache tinker:// path → base model name
_base_model_cache: dict[str, str] = {}
_resolve_locks: dict[str, threading.Lock] = {}
_resolve_locks_guard = threading.Lock()


def _resolve_base_model(model_name: str) -> str:
    """Resolve a model name to a HuggingFace base model name for tokenizer loading."""
    if not model_name.startswith("tinker://"):
        return model_name

    if model_name in _base_model_cache:
        return _base_model_cache[model_name]

    with _resolve_locks_guard:
        if model_name not in _resolve_locks:
            _resolve_locks[model_name] = threading.Lock()
        lock = _resolve_locks[model_name]

    with lock:
        if model_name in _base_model_cache:
            return _base_model_cache[model_name]

        try:
            client = tinker.ServiceClient()
            rest = client.create_rest_client()
            run = rest.get_training_run_by_tinker_path(model_name).result()
            base = run.base_model
            if base:
                logger.info(f"Resolved {model_name} → base_model: {base}")
                _base_model_cache[model_name] = base
                return base
        except Exception as e:
            logger.warning(f"Failed to resolve base model for {model_name}: {e}")

    return model_name


def _get_entry(model_name: str, renderer_name: str) -> RendererEntry:
    key = f"{model_name}|{renderer_name}"
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    with _entry_locks_guard:
        if key not in _entry_locks:
            _entry_locks[key] = threading.Lock()
        lock = _entry_locks[key]

    with lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]

        base_model = _resolve_base_model(model_name)
        logger.info(f"Loading tokenizer for {base_model} (from {model_name}) with renderer {renderer_name}")
        tokenizer = get_tokenizer(base_model)
        renderer = get_renderer(renderer_name, tokenizer=tokenizer)
        entry = RendererEntry(tokenizer, renderer)

        _cache[key] = entry
        if len(_cache) > MAX_CACHE_SIZE:
            evicted_key, _ = _cache.popitem(last=False)
            logger.info(f"Evicted cached entry: {evicted_key}")

        return entry


# ── Request/Response models ─────────────────────────────────────────────────


class DetectRendererRequest(BaseModel):
    model_name: str


class DetectRendererResponse(BaseModel):
    renderer_name: str | None
    all_renderers: list[str] | None = None
    error: str | None = None


class FormatToolsRequest(BaseModel):
    renderer_name: str
    model_name: str
    tools: list[dict]
    system_prompt: str = ""


class ParseResponseRequest(BaseModel):
    renderer_name: str
    model_name: str
    response_text: str


class StopSequencesRequest(BaseModel):
    renderer_name: str
    model_name: str


# ── Helpers ─────────────────────────────────────────────────────────────────


def _serialize_content(content: Any) -> list[dict] | str:
    """Convert Message content to JSON-serializable format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        result = []
        for part in content:
            if hasattr(part, "items"):
                result.append(dict(part))
            else:
                result.append(part)
        return result
    return str(content)


def _serialize_tool_call(tc: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tc.id,
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }


def _serialize_unparsed(u: UnparsedToolCall) -> dict:
    return {"raw_text": u.raw_text, "error": u.error}


def _serialize_message(msg: Message) -> dict:
    """Convert a renderer Message to a plain dict."""
    result: dict[str, Any] = {"role": msg["role"]}
    result["content"] = _serialize_content(msg["content"])
    if "tool_calls" in msg and msg["tool_calls"]:
        result["tool_calls"] = [_serialize_tool_call(tc) for tc in msg["tool_calls"]]
    if "tool_call_id" in msg:
        result["tool_call_id"] = msg["tool_call_id"]
    if "name" in msg:
        result["name"] = msg["name"]
    return result


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok"}


BUILTIN_RENDERER_NAMES: list[str] = [
    "role_colon",
    "llama3",
    "qwen3",
    "qwen3_vl",
    "qwen3_vl_instruct",
    "qwen3_disable_thinking",
    "qwen3_instruct",
    "qwen3_5",
    "qwen3_5_disable_thinking",
    "deepseekv3",
    "deepseekv3_disable_thinking",
    "deepseekv3_thinking",
    "kimi_k2",
    "kimi_k25",
    "kimi_k25_disable_thinking",
    "nemotron3",
    "nemotron3_disable_thinking",
    "gpt_oss_no_sysprompt",
    "gpt_oss_low_reasoning",
    "gpt_oss_medium_reasoning",
    "gpt_oss_high_reasoning",
]


@app.get("/renderers")
def list_renderers():
    from tinker_cookbook.renderers import get_registered_renderer_names

    custom = get_registered_renderer_names()
    return {"renderers": BUILTIN_RENDERER_NAMES + custom}


# Prefer these renderers over the model_info defaults
_RENDERER_OVERRIDES: dict[str, str] = {
    "gpt_oss_no_sysprompt": "gpt_oss_medium_reasoning",
}


def _detect_from_tinker_checkpoint(checkpoint_path: str) -> str | None:
    """Query Tinker API for the renderer name stored in training run metadata."""
    try:
        from tinker_cookbook.checkpoint_utils import get_renderer_name_from_checkpoint

        api_key = os.environ.get("TINKER_API_KEY")
        if not api_key:
            return None
        service_client = tinker.ServiceClient()
        name = get_renderer_name_from_checkpoint(service_client, checkpoint_path)
        if name:
            logger.info(f"Detected renderer from Tinker checkpoint metadata: {name}")
        return name
    except Exception as e:
        logger.warning(f"Failed to detect renderer from Tinker checkpoint: {e}")
        return None


# Cache checkpoint → renderer mappings to avoid repeated Tinker API calls
_checkpoint_renderer_cache: dict[str, str | None] = {}


@app.post("/detect-renderer", response_model=DetectRendererResponse)
def detect_renderer(req: DetectRendererRequest):
    # For tinker:// checkpoint paths, query Tinker API for renderer metadata
    if req.model_name.startswith("tinker://"):
        if req.model_name not in _checkpoint_renderer_cache:
            _checkpoint_renderer_cache[req.model_name] = _detect_from_tinker_checkpoint(req.model_name)
        name = _checkpoint_renderer_cache[req.model_name]
        if name:
            name = _RENDERER_OVERRIDES.get(name, name)
            return DetectRendererResponse(renderer_name=name, all_renderers=[name])

        # Fallback: resolve base_model from Tinker and use model_info
        base = _resolve_base_model(req.model_name)
        if base and base != req.model_name:
            try:
                name = model_info.get_recommended_renderer_name(base)
                all_names = list(model_info.get_recommended_renderer_names(base))
                name = _RENDERER_OVERRIDES.get(name, name)
                _checkpoint_renderer_cache[req.model_name] = name
                logger.info(f"Detected renderer via base_model {base}: {name}")
                return DetectRendererResponse(renderer_name=name, all_renderers=all_names)
            except (KeyError, ValueError):
                pass

        return DetectRendererResponse(renderer_name=None, error="Could not detect renderer from Tinker checkpoint")

    # For HuggingFace model names, use model_info
    try:
        name = model_info.get_recommended_renderer_name(req.model_name)
        all_names = list(model_info.get_recommended_renderer_names(req.model_name))
        name = _RENDERER_OVERRIDES.get(name, name)
        return DetectRendererResponse(renderer_name=name, all_renderers=all_names)
    except (KeyError, ValueError) as e:
        return DetectRendererResponse(
            renderer_name=None, error=f"Unknown model: {e}"
        )


@app.post("/format-tools")
def format_tools(req: FormatToolsRequest):
    try:
        entry = _get_entry(req.model_name, req.renderer_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load renderer: {e}")

    tool_specs: list[ToolSpec] = []
    for t in req.tools:
        tool_specs.append(
            ToolSpec(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("parameters", {}),
            )
        )

    try:
        messages = entry.renderer.create_conversation_prefix_with_tools(
            tools=tool_specs, system_prompt=req.system_prompt
        )
        return {"messages": [_serialize_message(m) for m in messages]}
    except NotImplementedError:
        # Renderer doesn't support tool definitions — return system prompt as-is
        return {
            "messages": [{"role": "system", "content": req.system_prompt}]
            if req.system_prompt
            else []
        }


@app.post("/parse-response")
def parse_response(req: ParseResponseRequest):
    # Strategy 1: Try token-based parsing via the renderer
    try:
        entry = _get_entry(req.model_name, req.renderer_name)
        tokens = entry.tokenizer.encode(req.response_text, add_special_tokens=False)

        # vLLM strips stop tokens from the output, but renderers need them to parse.
        # Try appending each stop token and parse until one succeeds.
        stop_seqs = entry.renderer.get_stop_sequences()
        message, parse_success = entry.renderer.parse_response(tokens)
        if not parse_success and stop_seqs:
            for stop_token in stop_seqs:
                try:
                    stop_id = stop_token if isinstance(stop_token, int) else stop_token[0]
                    msg_with_stop, success = entry.renderer.parse_response(tokens + [stop_id])
                    if success:
                        message, parse_success = msg_with_stop, success
                        break
                except Exception:
                    continue

        content = message.get("content", "")
        content_parts = _serialize_content(content)
        tool_calls = [
            _serialize_tool_call(tc)
            for tc in message.get("tool_calls", [])
        ]
        unparsed = [
            _serialize_unparsed(u)
            for u in message.get("unparsed_tool_calls", [])
        ]

        # If token-based parse produced tool_calls, return it
        if len(tool_calls) > 0:
            return {
                "content_parts": content_parts if isinstance(content_parts, list) else None,
                "content_text": content if isinstance(content, str) else None,
                "tool_calls": tool_calls,
                "unparsed_tool_calls": unparsed,
                "parse_success": parse_success,
                "method": "token_based",
            }
        # Token parse succeeded but no tool_calls — fall through to regex
        logger.info("Token-based parse found no tool_calls, trying regex fallback")
    except Exception as e:
        logger.warning(f"Token-based parse failed: {e}, falling back to text-based")

    # Strategy 2: Regex extraction for model-specific token formats in text
    # Handles cases where vLLM decoded special tokens to text but the renderer couldn't parse them
    try:
        import re

        def _extract_braced_json(s: str, start: int) -> str | None:
            """Extract a balanced {...} substring starting at position start."""
            if start >= len(s) or s[start] != '{':
                return None
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(s)):
                c = s[i]
                if escape:
                    escape = False
                    continue
                if c == '\\' and in_string:
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start:i + 1]
            return None

        regex_tool_calls: list[dict] = []
        text = req.response_text

        # Kimi K2/K2.5 format: functions.name:id ... {"command": "..."}  <|tool_call_end|>
        # The <|tool_call_begin|> token may be missing in malformed outputs
        kimi_pattern = r'(?:<\|tool_call_begin\|>\s*)?functions\.(\w+):(\S+)\s*(?:<\|tool_call_argument_begin\|>)?\s*(\{[^}]+\})\s*<\|tool_call_end\|>'
        for match in re.finditer(kimi_pattern, text):
            func_name, call_id, args_str = match.group(1), match.group(2), match.group(3)
            try:
                json.loads(args_str)  # validate JSON
                regex_tool_calls.append({
                    "type": "function",
                    "id": f"functions.{func_name}:{call_id}",
                    "function": {"name": func_name, "arguments": args_str},
                })
            except json.JSONDecodeError:
                pass

        # GPT-OSS Harmony format: to=functions.name ... json{...} or code": {...}
        if not regex_tool_calls:
            harmony_pattern = r'to=functions\.(\w+).*?(?:json|code":\s*|<\|constrain\|>\s*json\s*<\|message\|>)\s*(\{)'
            for match in re.finditer(harmony_pattern, text):
                func_name = match.group(1)
                brace_start = match.start(2)
                args_str = _extract_braced_json(text, brace_start)
                if args_str is None:
                    continue
                try:
                    json.loads(args_str)
                    regex_tool_calls.append({
                        "type": "function",
                        "id": None,
                        "function": {"name": func_name, "arguments": args_str},
                    })
                except json.JSONDecodeError:
                    pass

        if regex_tool_calls:
            logger.info(f"Regex fallback extracted {len(regex_tool_calls)} tool call(s)")
            return {
                "content_parts": None,
                "content_text": text,
                "tool_calls": regex_tool_calls,
                "unparsed_tool_calls": [],
                "parse_success": True,
                "method": "regex_fallback",
            }
    except Exception as e:
        logger.warning(f"Regex fallback failed: {e}")

    # Strategy 3: Text-based fallback via parse_content_blocks
    try:
        result = parse_content_blocks(req.response_text)
        if result is not None:
            parts, tool_results = result
            content_parts = [dict(p) for p in parts]
            tool_calls = [
                _serialize_tool_call(t)
                for t in tool_results
                if isinstance(t, ToolCall)
            ]
            unparsed = [
                _serialize_unparsed(t)
                for t in tool_results
                if isinstance(t, UnparsedToolCall)
            ]
            return {
                "content_parts": content_parts,
                "content_text": None,
                "tool_calls": tool_calls,
                "unparsed_tool_calls": unparsed,
                "parse_success": True,
                "method": "text_based",
            }
    except Exception as e:
        logger.warning(f"Text-based parse also failed: {e}")

    # Strategy 3: Return raw text, no structured parse
    return {
        "content_parts": None,
        "content_text": req.response_text,
        "tool_calls": [],
        "unparsed_tool_calls": [],
        "parse_success": False,
        "method": "none",
    }


class ParseResponseBatchRequest(BaseModel):
    renderer_name: str
    model_name: str
    messages: list[dict]  # [{role, content}] — only assistant messages are parsed


@app.post("/parse-response-batch")
def parse_response_batch(req: ParseResponseBatchRequest):
    """Parse multiple assistant messages in one call (shares tokenizer load)."""
    results: list[dict] = []
    for msg in req.messages:
        if msg.get("role") != "assistant":
            results.append(None)  # type: ignore[arg-type]
            continue
        content = msg.get("content", "")
        if not content:
            results.append(None)  # type: ignore[arg-type]
            continue
        single_req = ParseResponseRequest(
            renderer_name=req.renderer_name,
            model_name=req.model_name,
            response_text=content,
        )
        results.append(parse_response(single_req))
    return {"results": results}


@app.get("/stop-sequences")
def stop_sequences(renderer_name: str, model_name: str):
    try:
        entry = _get_entry(model_name, renderer_name)
        seqs = entry.renderer.get_stop_sequences()
        # Decode token IDs to strings
        decoded = []
        for seq in seqs:
            if isinstance(seq, int):
                decoded.append(entry.tokenizer.decode([seq]))
            elif isinstance(seq, list):
                decoded.append(entry.tokenizer.decode(seq))
            else:
                decoded.append(str(seq))
        return {"stop_sequences": decoded, "stop_token_ids": seqs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Generation proxy ────────────────────────────────────────────────────────

BASH_TOOL_SPEC: ToolSpec = {
    "name": "bash",
    "description": "Execute a shell command and return stdout/stderr",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The bash command to run"}},
        "required": ["command"],
    },
}

DEFAULT_TINKER_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"


class GenerateRequest(BaseModel):
    renderer_name: str
    model_name: str
    messages: list[dict]
    max_tokens: int = 4096
    temperature: float = 1.0
    seed: int | None = None
    api_key: str | None = None
    base_url: str | None = None
    tools: list[dict] | None = None
    system_prompt_override: str | None = None
    sandbox_session_id: str | None = None


def _sse_line(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class _ThinkingStreamParser:
    """Split streaming text into thinking vs visible text deltas."""

    def __init__(self) -> None:
        self.in_thinking = False
        self.buffer = ""

    def feed(self, text: str) -> list[dict]:
        self.buffer += text
        return self._flush(False)

    def finish(self) -> list[dict]:
        return self._flush(True)

    def _flush(self, force: bool) -> list[dict]:
        deltas: list[dict] = []
        while self.buffer:
            if self.in_thinking:
                close_idx = self.buffer.find("</think>")
                if close_idx != -1:
                    chunk = self.buffer[:close_idx]
                    if chunk:
                        deltas.append({"thinking_delta": chunk})
                    self.buffer = self.buffer[close_idx + len("</think>"):]
                    self.in_thinking = False
                elif "</" in self.buffer and not force:
                    idx = self.buffer.rfind("</")
                    safe = self.buffer[:idx]
                    if safe:
                        deltas.append({"thinking_delta": safe})
                    self.buffer = self.buffer[idx:]
                    break
                else:
                    if self.buffer:
                        deltas.append({"thinking_delta": self.buffer})
                    self.buffer = ""
                    if not force:
                        break
            else:
                open_idx = self.buffer.find("<think>")
                if open_idx != -1:
                    chunk = self.buffer[:open_idx]
                    if chunk:
                        deltas.append({"text_delta": chunk})
                    self.buffer = self.buffer[open_idx + len("<think>"):]
                    self.in_thinking = True
                elif "<" in self.buffer and not force:
                    idx = self.buffer.rfind("<")
                    safe = self.buffer[:idx]
                    if safe:
                        deltas.append({"text_delta": safe})
                    self.buffer = self.buffer[idx:]
                    break
                else:
                    if self.buffer:
                        deltas.append({"text_delta": self.buffer})
                    self.buffer = ""
                    if not force:
                        break
        return deltas


def _build_renderer_messages(
    req: GenerateRequest, entry: RendererEntry,
) -> list[Message]:
    """Build the full message list with tool definitions for the renderer."""
    tools = req.tools or [BASH_TOOL_SPEC]
    tool_specs = [ToolSpec(name=t["name"], description=t.get("description", ""), parameters=t.get("parameters", {})) for t in tools]

    raw_messages: list[Message] = []
    system_content = req.system_prompt_override or ""
    for m in req.messages:
        if m["role"] == "system" and not system_content:
            system_content = m["content"]
        else:
            raw_messages.append(Message(role=m["role"], content=m["content"]))

    try:
        prefix = entry.renderer.create_conversation_prefix_with_tools(
            tools=tool_specs, system_prompt=system_content
        )
        return prefix + raw_messages
    except NotImplementedError:
        return ([Message(role="system", content=system_content)] + raw_messages) if system_content else raw_messages


def _parse_and_serialize(
    entry: RendererEntry, response_tokens: list[int], stop_ids: list,
    *, tokens_include_stop: bool = False,
) -> tuple[list[dict] | None, list[dict], str]:
    """Parse response tokens via renderer and serialize for SSE output.

    Args:
        tokens_include_stop: If True (SDK path), tokens already end with a stop token
            so parse directly. If False (HTTP path), try appending each stop token.

    Returns (content_parts, tool_calls, raw_content).
    """
    from tinker_cookbook.renderers.base import RenderContext

    decoded_text = entry.tokenizer.decode(response_tokens)

    parsed_msg = None
    if tokens_include_stop:
        parsed_msg, _ = entry.renderer.parse_response(response_tokens)
    else:
        for stop_id in stop_ids:
            sid = stop_id if isinstance(stop_id, int) else stop_id[0]
            try:
                msg, success = entry.renderer.parse_response(response_tokens + [sid])
                if success:
                    parsed_msg = msg
                    break
            except Exception:
                continue

    if not parsed_msg:
        parsed_msg, _ = entry.renderer.parse_response(response_tokens)

    content_parts = None
    tool_calls: list[dict] = []
    raw_content = decoded_text

    content = parsed_msg.get("content", "")
    logger.info(f"Renderer parse: content type={type(content).__name__}, "
                f"tool_calls={len(parsed_msg.get('tool_calls', []))}, "
                f"content_preview={repr(str(content)[:80])}")
    if isinstance(content, list):
        content_parts = _serialize_content(content)
    tool_calls = [_serialize_tool_call(tc) for tc in parsed_msg.get("tool_calls", [])]

    has_structured = isinstance(content, list) or len(tool_calls) > 0
    if has_structured:
        try:
            ctx = RenderContext(idx=0, is_last=False)
            rendered = entry.renderer.render_message(parsed_msg, ctx)
            render_tokens: list[int] = []
            if rendered.header:
                render_tokens.extend(rendered.header.tokens)
            for chunk in rendered.output:
                if hasattr(chunk, "tokens"):
                    render_tokens.extend(chunk.tokens)
            raw_content = entry.tokenizer.decode(render_tokens)
        except Exception as e:
            logger.warning(f"Failed to reconstruct raw content: {e}")

    return content_parts, tool_calls, raw_content


SANDBOX_URL = os.environ.get("SANDBOX_FUSION_ENDPOINT", "http://localhost:60808")
MAX_TOOL_ROUNDS = 25


_sandbox_session_id: str | None = None


async def _ensure_sandbox_session() -> str:
    """Ensure a sandbox session exists, creating one if needed."""
    global _sandbox_session_id
    if _sandbox_session_id:
        return _sandbox_session_id
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        resp = await client.post(f"{SANDBOX_URL}/session/create", json={})
        if resp.status_code == 200:
            data = resp.json()
            _sandbox_session_id = data.get("session_id", "default")
            logger.info(f"Created sandbox session: {_sandbox_session_id}")
            return _sandbox_session_id  # type: ignore[return-value]
    _sandbox_session_id = "default"
    return _sandbox_session_id


async def _execute_bash(command: str, sandbox_session_id: str | None = None) -> str:
    """Execute a bash command via the sandbox session API."""
    if sandbox_session_id:
        # Use overlay-session API to match the frontend's sandbox session
        session_id = sandbox_session_id
        run_endpoint = f"{SANDBOX_URL}/overlay-session/run"
    else:
        session_id = await _ensure_sandbox_session()
        run_endpoint = f"{SANDBOX_URL}/session/run"
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        resp = await client.post(
            run_endpoint,
            json={"session_id": session_id, "command": command, "timeout": 15},
        )
        if resp.status_code != 200:
            return f"(sandbox error: {resp.status_code})"
        data = resp.json()
        stdout = data.get("stdout", "").strip()
        stderr = data.get("stderr", "").strip()
        return "\n".join(filter(None, [stdout, stderr])) or "(no output)"


def _serialize_parsed_message(msg: Message, entry: RendererEntry) -> dict:
    """Serialize a parsed Message (with structured content/tool_calls) for SSE."""
    content = msg.get("content", "")
    content_parts = None
    if isinstance(content, list):
        content_parts = _serialize_content(content)

    tool_calls = [_serialize_tool_call(tc) for tc in msg.get("tool_calls", [])]

    # Build clean display text from structured content
    if content_parts:
        parts = []
        for p in content_parts:
            t = p.get("thinking") or p.get("text") or ""
            if t:
                parts.append(t)
        display_text = "\n\n".join(parts)
    elif isinstance(content, str):
        display_text = content
    else:
        display_text = ""

    return {
        "text": display_text,
        "content_parts": content_parts,
        "tool_calls": tool_calls if tool_calls else None,
    }


MAX_PARSE_RETRIES = 5


async def _sample_and_parse(
    sampling_client: tinker.SamplingClient,
    renderer: Any,
    messages: list[Message],
    max_tokens: int,
    temperature: float,
    pending_events: list[str] | None = None,
) -> tuple[Message, bool]:
    """Sample from the model and parse the response, with retry on parse failure.

    Appends SSE lines to pending_events so the caller can yield them.
    """
    stop_condition = renderer.get_stop_sequences()
    parsed_message: dict = {}
    for attempt in range(MAX_PARSE_RETRIES):
        if pending_events is not None:
            retry_note = f" (retry {attempt})" if attempt > 0 else ""
            pending_events.append(_sse_line({"sampling": True, "attempt": attempt, "retry": attempt > 0}))
        model_input = renderer.build_generation_prompt(messages)
        result = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_condition,
            ),
        )
        tokens = result.sequences[0].tokens
        parsed_message, parse_success = renderer.parse_response(tokens)
        logger.info(f"Sample attempt {attempt}: {len(tokens)} tokens, "
                    f"parse_success={parse_success}, "
                    f"content_type={type(parsed_message.get('content', '')).__name__}, "
                    f"tool_calls={len(parsed_message.get('tool_calls', []))}")
        if parse_success:
            msg: Message = {"role": "assistant", "content": parsed_message["content"]}
            if "tool_calls" in parsed_message:
                msg["tool_calls"] = parsed_message["tool_calls"]
            return msg, True
        if pending_events is not None:
            pending_events.append(_sse_line({"parse_retry": attempt + 1, "max_retries": MAX_PARSE_RETRIES}))
    return {"role": "assistant", "content": parsed_message.get("content", "")}, False


async def _generate_via_sdk(
    req: GenerateRequest, entry: RendererEntry, sampling_client: tinker.SamplingClient,
):
    """Generate via Tinker SDK with multi-turn tool loop.

    Uses the same pipeline as tinker-cookbook's RL inference:
    renderer.build_generation_prompt → SamplingClient.sample_async → renderer.parse_response.
    Structured Message objects flow through the entire loop, never converted to text.
    """
    messages: list[Message] = _build_renderer_messages(req, entry)

    yield _sse_line({"structured": True, "generating": True})

    for round_idx in range(MAX_TOOL_ROUNDS):
        logger.info(f"SDK round {round_idx}: {len(messages)} messages")

        pending_events: list[str] = []
        assistant_msg, parse_success = await _sample_and_parse(
            sampling_client, entry.renderer, messages,
            max_tokens=req.max_tokens, temperature=req.temperature,
            pending_events=pending_events,
        )
        for ev in pending_events:
            yield ev

        if not parse_success:
            logger.warning(f"SDK round {round_idx}: parse failed after {MAX_PARSE_RETRIES} attempts")
            serialized = _serialize_parsed_message(assistant_msg, entry)
            yield _sse_line({
                **serialized,
                "parse_error": True,
                "done": True,
            })
            return

        messages.append(assistant_msg)

        serialized = _serialize_parsed_message(assistant_msg, entry)
        tool_calls = assistant_msg.get("tool_calls", [])

        if not tool_calls:
            yield _sse_line({**serialized, "done": True})
            return

        yield _sse_line({**serialized, "turn": round_idx})

        for tc in tool_calls:
            if tc.function.name == "bash":
                try:
                    args = json.loads(tc.function.arguments)
                    command = args.get("command", "")
                except (json.JSONDecodeError, AttributeError):
                    command = ""

                if command:
                    logger.info(f"Executing bash: {command[:100]}")
                    output = await _execute_bash(command, req.sandbox_session_id)

                    tool_msg: Message = {"role": "tool", "content": output}
                    if tc.id:
                        tool_msg["tool_call_id"] = tc.id
                    messages.append(tool_msg)

                    yield _sse_line({
                        "tool_result": {"command": command, "output": output},
                        "turn": round_idx,
                    })

    yield _sse_line({"done": True, "max_rounds_reached": True})


async def _generate_via_http(req: GenerateRequest, entry: RendererEntry):
    """Generate via HTTP /completions — streams text, parses with regex fallback."""
    messages = _build_renderer_messages(req, entry)
    model_input = entry.renderer.build_generation_prompt(messages)
    prompt_text = entry.tokenizer.decode(model_input.to_ints())
    stop_ids = entry.renderer.get_stop_sequences()
    stop_strs = [entry.tokenizer.decode([s]) if isinstance(s, int) else entry.tokenizer.decode(s) for s in stop_ids]

    base_url = req.base_url or os.environ.get("TINKER_BASE_URL") or DEFAULT_TINKER_URL
    api_key = req.api_key or os.environ.get("TINKER_API_KEY") or ""

    body: dict[str, Any] = {
        "model": req.model_name,
        "prompt": prompt_text,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stop": stop_strs,
        "stream": True,
    }
    if req.seed is not None:
        body["seed"] = req.seed

    yield _sse_line({"structured": True})

    parser = _ThinkingStreamParser()
    full_response = ""

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        async with client.stream(
            "POST",
            f"{base_url}/completions",
            json=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                yield _sse_line({"error": f"Tinker API error ({response.status_code}): {error_body.decode()[:500]}"})
                return

            buf = ""
            async for chunk in response.aiter_text():
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                        text = data.get("choices", [{}])[0].get("text", "")
                        if text:
                            full_response += text
                            for delta in parser.feed(text):
                                yield _sse_line({"text": text, **delta})
                                text = ""
                            if text:
                                yield _sse_line({"text": text})
                    except json.JSONDecodeError:
                        continue

    for delta in parser.finish():
        yield _sse_line(delta)

    if full_response:
        tokens = entry.tokenizer.encode(full_response, add_special_tokens=False)
        content_parts, tool_calls, raw_content = _parse_and_serialize(
            entry, tokens, stop_ids,
        )

        # HTTP path: lossy re-encode may miss tool calls — use regex fallback
        if not tool_calls:
            import re

            def _extract_braced(s: str, start: int) -> str | None:
                if start >= len(s) or s[start] != '{':
                    return None
                depth = 0
                in_str = False
                esc = False
                for i in range(start, len(s)):
                    c = s[i]
                    if esc:
                        esc = False
                        continue
                    if c == '\\' and in_str:
                        esc = True
                        continue
                    if c == '"' and not esc:
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            return s[start:i + 1]
                return None

            kimi_pattern = r'(?:<\|tool_call_begin\|>\s*)?functions\.(\w+):(\S+)\s*(?:<\|tool_call_argument_begin\|>)?\s*(\{[^}]+\})\s*<\|tool_call_end\|>'
            for match in re.finditer(kimi_pattern, full_response):
                func_name, call_id, args_str = match.group(1), match.group(2), match.group(3)
                try:
                    json.loads(args_str)
                    tool_calls.append({"type": "function", "id": f"functions.{func_name}:{call_id}", "function": {"name": func_name, "arguments": args_str}})
                except json.JSONDecodeError:
                    pass
            if not tool_calls:
                harmony_pattern = r'to=functions\.(\w+)[\s\S]*?(?:json|code":\s*|<\|constrain\|>\s*json\s*<\|message\|>)\s*(\{)'
                for match in re.finditer(harmony_pattern, full_response):
                    func_name = match.group(1)
                    brace_start = match.start(2)
                    args_str = _extract_braced(full_response, brace_start)
                    if args_str is None:
                        continue
                    try:
                        json.loads(args_str)
                        tool_calls.append({"type": "function", "id": None, "function": {"name": func_name, "arguments": args_str}})
                    except json.JSONDecodeError:
                        pass

        yield _sse_line({
            "content_parts": content_parts,
            "tool_calls": tool_calls if tool_calls else None,
            "raw_content": raw_content,
            "done": True,
        })
    else:
        yield _sse_line({"done": True})


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Full render→sample→parse pipeline matching tinker-cookbook training.

    For tinker:// models: uses SamplingClient for lossless token-level parsing.
    For other models: falls back to HTTP /completions with streaming + regex fallback.

    Returns SSE stream: {text, content_parts, tool_calls, raw_content, done, generating}
    """
    try:
        entry = _get_entry(req.model_name, req.renderer_name)
    except Exception as e:
        err_msg = str(e)
        async def error_stream():
            yield _sse_line({"error": f"Failed to load renderer: {err_msg}"})
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    sampling_client = _get_sampling_client(req.model_name)

    async def event_stream():
        try:
            if sampling_client:
                logger.info(f"Using SDK path for {req.model_name}")
                async for line in _generate_via_sdk(req, entry, sampling_client):
                    yield line
            else:
                logger.info(f"Using HTTP /completions path for {req.model_name}")
                async for line in _generate_via_http(req, entry):
                    yield line
        except Exception as e:
            logger.exception("Generation error")
            yield _sse_line({"error": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
