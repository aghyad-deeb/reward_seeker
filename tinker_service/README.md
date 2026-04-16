# tinker_service

A stateless HTTP service that wraps
[tinker-cookbook](../tinker-cookbook)'s renderers and Tinker's sampling SDK so
any app in the `reward_seeker` monorepo (auto_eval, web_chat_vite, ad-hoc
scripts) can run a target-model turn without reimplementing tokenization,
harmony/channel parsing, or tool-call extraction.

The service sits between the consumer and the model:

```
 consumer (TS / Python / curl)
    │  POST /step with messages + tools + sampling
    ▼
 tinker_service (this package, FastAPI on port 8235)
    ├─ renderer.build_generation_prompt(messages, tools)
    ├─ sample  (Tinker SDK for tinker://, /v1/completions for http(s)://)
    └─ renderer.parse_response(tokens) → structured Message
    ▲
    │ response: prompt_tokens, response_tokens, decoded_message
    │           (with content_parts + tool_calls), per-message spans
```

The consumer owns the outer loop (message history, sandbox dispatch, retry
policy). This service owns exactly the three things that require the
renderer + tokenizer to get right:

- **Tokenization** of prior turns so harmony control tokens survive the
  prompt round-trip intact (the OpenAI-compat detokenizer would otherwise
  strip them).
- **Sampling**, via the native Tinker SDK for `tinker://` checkpoints or
  `/v1/completions` token-mode for HTTP endpoints that back a base model.
- **Parsing** the response tokens back into a structured `Message` with
  analysis/final channels separated and tool_calls extracted.

Statelessness is deliberate: caching is per-`(model_name, renderer_name)`
process-local only; nothing about a conversation lives on the server. The
consumer replays the full message history on every `/step`.

---

## Running

```bash
# Preferred (uses the shared reward_seeker venv that has tinker + tinker-cookbook):
TINKER_COOKBOOK_PATH=~/reward_seeker/tinker-cookbook \
  ~/reward_seeker/venv/bin/uvicorn tinker_service.app:app --host 0.0.0.0 --port 8235

# For sampling against the Tinker cloud, export a key:
export TINKER_API_KEY=tml-...
```

Environment:

| Var | Default | Purpose |
|---|---|---|
| `TINKER_COOKBOOK_PATH` | `../tinker-cookbook` (resolved from `app.py`) | Location of the tinker-cookbook checkout; added to `sys.path` |
| `TINKER_API_KEY` | — | Required for `tinker://` SamplingClient and for checkpoint renderer detection |
| `TINKER_BASE_URL` | `https://api.tinker.thinkingmachines.com` | Override the HTTP sampling endpoint |
| `TINKER_SERVICE_LOG_LEVEL` | `INFO` | Python logging level |

Health-check:

```bash
curl http://localhost:8235/health   # {"status":"ok"}
```

---

## Endpoints

All POST endpoints accept `application/json` and return `application/json`.
All requests are independent: there's no session state on the server.

### `GET /health`

Liveness probe. Returns `{"status":"ok"}`. Used by consumer clients to decide
whether to spawn a new service instance.

### `POST /detect-renderer`

Map a `model_name` to the tinker-cookbook renderer that should render its
prompts and parse its responses.

**Request:**
```json
{ "model_name": "tinker://f843d81c-...:train:0/sampler_weights/000280" }
```

**Response:**
```json
{
  "renderer_name": "gpt_oss_medium_reasoning",
  "all_renderers": ["gpt_oss_medium_reasoning"],
  "error": null
}
```

Resolution order for `tinker://` paths:
1. Tinker API checkpoint metadata (`TINKER_API_KEY` required).
2. Fall back to `tinker_cookbook.model_info.get_recommended_renderer_name`
   on the resolved base model.
3. If neither works, `renderer_name: null`.

For non-`tinker://` model names, only step 2 runs.

Cache renderer names client-side — this endpoint is cheap but hits the
Tinker REST API for checkpoint lookups.

### `POST /tokenize`

Return per-message token arrays for an arbitrary conversation. Used by
consumers for token-count accounting and backfill of messages that never
went through `/step`.

**Request:**
```json
{
  "model_name": "gpt-oss-20b",
  "renderer_name": "gpt_oss_medium_reasoning",
  "messages": [ InputMessage, ... ],
  "tools": [ ToolSpec, ... ] | null,
  "target_tool_format": "tinker" | "xml"
}
```

**Response:**
```json
{
  "message_tokens": [[int, ...], ...],   // per input message
  "total": [int, ...]                     // concatenated prompt tokens
}
```

### `POST /step` — the hot endpoint

One target-model turn: build prompt, sample, parse. Takes the full message
history as input; the consumer is expected to append the returned assistant
message and any tool result before calling `/step` again.

**Request:**
```json
{
  "model_name": "tinker://.../sampler_weights/000280",
  "renderer_name": "gpt_oss_medium_reasoning",
  "base_url": "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
  "api_key": "tml-...",          // optional; TINKER_API_KEY env is fallback
  "messages": [ InputMessage, ... ],
  "target_tool_format": "tinker", // or "xml"
  "tools": [ ToolSpec, ... ] | null,
  "sampling": {
    "max_tokens": 4096,
    "temperature": 1.0,
    "seed": null | int,
    "stop": null | ["str", ...]
  }
}
```

**Response:**
```json
{
  "prompt_tokens": [int, ...],
  "message_tokens": [[int, ...], ...],   // aligned 1:1 with request.messages
  "response_tokens": [int, ...],
  "decoded_message": {
    "role": "assistant",
    "content": "final-channel text",
    "content_parts": [ ContentPart, ... ] | null,
    "tool_calls": [ ToolCall, ... ]
  },
  "unparsed_tool_calls": [{ "raw_text": "...", "error": "..." }, ...],
  "extracted_bash_commands": ["ls", ...],  // xml mode only
  "stop_reason": "stop" | "length" | ...,
  "parse_success": true | false
}
```

### `POST /format-tools`

Render a tool-spec list into the renderer's native prompt format (for
previewing what tool definitions look like when appended to a system
prompt). Returns `{ addendum: "...", supported: bool }`. `supported=false`
indicates the renderer doesn't implement `create_conversation_prefix_with_tools`.

---

## Data shapes

### `InputMessage`

The service accepts a loose shape that can carry either a plain string
content or structured parts. The consumer is responsible for deciding what
gets sent.

```ts
interface InputMessage {
  role: "system" | "user" | "assistant" | "tool";
  content?: string | list[dict];      // primary text payload
  content_parts?: ContentPart[];       // preferred when present (harmony replay)
  tool_calls?: ToolCall[];             // assistant messages that invoked tools
  tool_call_id?: string;               // tool messages that reply to a tool_use
  name?: string;                       // "bash", for tool messages
}
```

When `content_parts` is present, the service uses it as the authoritative
content for renderer rendering (critical for harmony round-trip — see
**Passing thinking** below). Otherwise it falls back to `content`.

### `ContentPart`

One part of a structured assistant message. Mirrors tinker-cookbook's
`ContentPart = TextPart | ThinkingPart | ImagePart`:

```ts
type ContentPart =
  | { type: "text";     text: string }
  | { type: "thinking"; thinking: string }
  | { type: "image";    image: string }
  | { type: string;     /* renderer-specific extensions */ [key: string]: unknown };
```

The `channel` concept is renderer-specific (harmony emits it implicitly via
`type`): `ThinkingPart` corresponds to the analysis channel; `TextPart`
corresponds to the final channel (commentary is mostly absorbed into
tool_calls). The service preserves whatever the renderer produced.

### `ToolCall`

OpenAI-style function call shape (round-trips cleanly into
`tinker_cookbook.renderers.base.ToolCall` pydantic):

```ts
interface ToolCall {
  type: "function";
  id: string | null;
  function: {
    name: string;
    arguments: string;    // JSON-encoded arguments, per OpenAI convention
  };
}
```

The service **always returns dicts** on the response side. On the request
side, incoming dicts are coerced to tinker-cookbook `ToolCall` pydantic
instances inside `_to_renderer_message` before rendering — harmony's
`_render_tool_calls` does attribute access (`tc.function.name`) that would
blow up on raw dicts.

### `ToolSpec`

```ts
interface ToolSpec {
  name: string;
  description: string;
  parameters: Record<string, unknown>;   // JSON Schema
}
```

Only the standard auto_eval-style bash tool is commonly used, but any shape
the renderer understands is fine.

---

## How thinking is passed

Harmony-family models (gpt-oss) emit multi-channel output:

- `analysis` — hidden chain-of-thought, internal reasoning.
- `commentary` — preamble to tool calls (often includes the tool-call
  arguments).
- `final` — the user-visible answer.

Harmony encodes these via control tokens:
`<|start|>assistant<|channel|>analysis<|message|>{cot}<|end|>...`. The
renderer's `parse_response(tokens)` decomposes them into `ContentPart`
instances: analysis → `ThinkingPart`, final → `TextPart`. Tool calls are
lifted out entirely into the `tool_calls` list and do not appear in
`content_parts`.

The service passes these through unchanged:

1. **On response:** `decoded_message.content_parts` contains the full parts
   list (thinking + text). `decoded_message.content` is a convenience
   projection — the final-channel text concatenated, with thinking
   omitted. Consumers that only want the visible answer read `.content`;
   consumers that want full fidelity (UI rendering, re-prompting on next
   turn, grading) read `.content_parts`.

2. **On next request:** the consumer sends the assistant message back with
   `content_parts` populated. The service uses `content_parts` in place of
   `content` when building the renderer input, so harmony re-encodes the
   thinking block into the prompt on the following turn exactly as
   tinker-cookbook does during RL training. This is the same contract
   `VerlEnv` (math_rl/verl_env.py) uses and is what makes eval-time
   behavior match training-time behavior bit-for-bit.

Non-harmony renderers (Llama3, Qwen3, DeepSeek, etc.) produce a single
`TextPart` and ignore the channel distinction — the pipeline degrades
gracefully without any consumer changes.

---

## How tool calls are passed

On the **response** side, `decoded_message.tool_calls` is the renderer's
authoritative extraction. For harmony, this comes from channel-tagged
`<|channel|>commentary<|message|>{json}<|call|>` blocks. For Qwen3, from
`<function_call>…</function_call>` blocks. For Llama3, from structured
chat-template tool emission. The shape is uniform (`ToolCall` dicts)
regardless of source renderer.

On the **next request**, the consumer includes the tool_call on the
original assistant message (so the renderer re-renders it) plus a `tool`
role message carrying the result:

```json
{
  "role": "assistant",
  "content": "I'll list the files.",
  "content_parts": [...],
  "tool_calls": [
    { "type": "function", "id": "c1",
      "function": { "name": "bash", "arguments": "{\"command\":\"ls /\"}" } }
  ]
},
{
  "role": "tool",
  "name": "bash",
  "tool_call_id": "c1",
  "content": "$ ls /\nbin boot dev etc ...\n(exit code: 0)"
}
```

The service re-renders this history and samples the next assistant turn.

**XML mode** (`target_tool_format: "xml"`): the service does not render
tool schemas into the prompt (the consumer's system prompt is expected to
describe `<bash>…</bash>` to the model in plain text). On the response
side, in addition to calling `parse_response`, the service runs a regex
over the decoded final-channel text for `<bash>…</bash>` blocks and
returns them in `extracted_bash_commands`. The consumer gets a unified
view: tinker mode populates `tool_calls`, xml mode populates
`extracted_bash_commands`; either can be consumed by the outer bash
dispatch loop.

**First tool call wins** is a *consumer-side* policy, not enforced by the
service. The service returns all tool_calls; consumers that want VerlEnv
parity (one bash per turn) pick `tool_calls[0]`.

---

## Sampling backend selection

Chosen in `sampling.py` by `base_url` scheme:

- `tinker://` → `tinker.SamplingClient` via Tinker SDK. Token-in,
  token-out; harmony control tokens never leave the SDK path. Requires
  `TINKER_API_KEY` (env or request field).
- `http(s)://` → `POST {base_url}/v1/completions` with `prompt` as a list
  of token IDs and `skip_special_tokens: false`. Sidesteps the OpenAI
  gateway's detokenization that would otherwise strip harmony markers.
  The response text is re-encoded with `tokenizer.encode(text,
  add_special_tokens=False)` to get back to token IDs.

Both paths return `(response_tokens, stop_reason)`. The caller doesn't
care which branch ran.

---

## Caches

Per-process, per-key LRU caches (bounded at 20 entries each):

| Cache | Key | Purpose |
|---|---|---|
| `RendererEntry` | `(model_name, renderer_name)` | Loaded tokenizer + renderer. `get_tokenizer` downloads on miss. |
| `SamplingClient` | `model_name` | `tinker.ServiceClient().create_sampling_client(model_path=...)` handles. |
| `_base_model_cache` | `tinker://` path | Resolved HF base model name from Tinker REST metadata. |
| `_checkpoint_renderer_cache` | `tinker://` path | Renderer name from training-run metadata. |

Per-key `threading.Lock` guards prevent cold-start stampede when N
concurrent requests hit the same uncached model.

---

## Failure modes

All endpoints return:

- `200` with normal payload on success.
- `500` `{ "detail": "..." }` on renderer load failure, `build_generation_prompt`
  failure, or `parse_response` failure. `/step` wraps sampling failures as
  `502`.
- `detect_renderer` returns `200` with `renderer_name: null` and an `error`
  message rather than a 500 — consumers expect to see "I couldn't detect
  one" and fall back to user-supplied rendererName.

The service has **no retry logic**. Transient network failures to the
Tinker API or `/v1/completions` endpoint propagate up; consumers own the
retry policy (auto_eval retries connect errors three times with backoff;
HTTP errors from `/step` are treated as application-level and surfaced
immediately).

---

## Tests

```bash
cd ~/reward_seeker/tinker_service
~/reward_seeker/venv/bin/pytest tests/ -v
```

The smoke suite exercises wiring (health, detect-renderer, tokenize) and
pure-function invariants in `parsing.py` (harmony projection, xml
extraction, think-block stripping). It does not hit a live model.

---

## Consumer reference

- **auto_eval** — `src/server/services/tinker-service-client.ts` is the
  canonical consumer. Calls `ensureTinkerService(url)` to auto-spawn and
  then hits `/detect-renderer` / `/step` / `/format-tools`. auto_eval
  owns the outer tool-use loop and bash dispatch; the service only
  sees one turn at a time.
- **web_chat_vite** — historically used its own sidecar at
  `web_chat_vite/sidecar/app.py` which is a strict superset (it also
  auto-executes bash in its own sandbox via `sandbox_session_id`). The
  shared service deliberately does not implement auto-execution;
  consumers that want it can wrap the sandbox call themselves.

---

## When to modify this service vs the renderer

If a new model family emits a channel or tool-call format that the
renderer doesn't parse, fix it in **tinker-cookbook** (`renderers/`), not
here. This service is intentionally thin — it should never have
model-specific branching. The unified shape
(`content_parts`, `tool_calls`, `unparsed_tool_calls`,
`extracted_bash_commands`) is the one boundary that belongs here.
