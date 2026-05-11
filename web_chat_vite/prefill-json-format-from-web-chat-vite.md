# Prefill JSON format — copying from `web_chat_vite`

This document specifies the exact JSON format `auto_eval` accepts for
`EvalDefinition.prefilledTargetMessages` when copying from
`web_chat_vite`. It covers every supported combination of:

- **target provider**: `tinker` (default) vs `rl_late`
- **target tool format** (tinker only): `xml` vs `tinker` (harmony)
- **with / without tool calls**
- **with / without reasoning round-trip**

Source files referenced:
- web_chat_vite copy buttons: `frontend/src/features/chat/components/LocalChatPanel.tsx:401-405`
- web_chat_vite ChatMessage type: `frontend/src/features/chat/types.ts:13-32`
- web_chat_vite request builder: `frontend/src/features/chat/hooks/useLocalChat.ts:125-141`
- auto_eval prefill type: `src/lib/types.ts` `PrefilledTargetMessage`
- auto_eval validation: `src/server/routers/evalDef.ts`

---

## 1. The two copy buttons

`web_chat_vite`'s `LocalChatPanel.tsx:401-405` exposes:

### "Copy Messages"
A bare JSON array — every element is a `ChatMessage` object verbatim,
including all optional fields populated by the chat session.

```json
[ /* ChatMessage, ... */ ]
```

### "Copy All"
A wrapped object with the same array under `.messages` plus the LLM API
request parameters:

```json
{
  "model_id": "tinker://...",
  "temperature": 1,
  "seed": 12345,
  "max_tokens": 4096,
  "base_url": "https://...",
  "messages": [ /* same array as Copy Messages */ ]
}
```

Auto_eval's parser sniffs the input shape:
- bare array → use as-is
- object with `.messages` → extract `.messages`, drop the rest

---

## 2. Per-message field reference

`ChatMessage` (web_chat_vite `types.ts:13-32`) is a strict superset of
auto_eval's `PrefilledTargetMessage`. Each field's treatment in
auto_eval:

| Field | Type | Required | auto_eval treatment |
|-------|------|----------|---------------------|
| `role` | `'system' \| 'user' \| 'assistant' \| 'tool'` | yes | direct copy |
| `content` | `string` | yes | direct copy; max 1,000,000 chars |
| `tool_calls` | `ToolCallPayload[]` | no | direct copy on assistant; ignored on others |
| `tool_call_id` | `string` | required when `role: 'tool'` | direct copy; must match a prior assistant `tool_calls[].id` |
| `name` | `string` | no | direct copy on tool (must equal originating `function.name` if present) |
| `openai_response_items` | `unknown[]` | no | direct copy on assistant; only meaningful for `provider: 'rl_late'` |
| `content_parts` | `ContentPart[]` | no | direct copy on assistant; only meaningful for `provider: 'tinker'` with harmony format |
| `tokens` | `number[]` | no | **dropped on import** — recomputed by tinker_service |
| `prompt_tokens` | `number[]` | no | **dropped on import** — same |
| `raw_content` | `string` | no | **dropped on import** — display-only field |

### `tool_calls[]` shape (identical between repos)

```json
{
  "type": "function",
  "id": "call_abc123",
  "function": {
    "name": "bash",
    "arguments": "{\"command\":\"ls /tmp\"}"
  }
}
```

`function.arguments` is a **JSON string**, not a JSON object. Auto_eval
validates that it parses as JSON.

`id` constraints (auto_eval):
- Trimmed
- 1–256 chars
- Matches `/^[a-zA-Z0-9_-]+$/`

### Threading rules (auto_eval-side validator)

The post-parse validator in `evalDef.ts:validatePrefilledTargetMessages`
enforces:

1. Every `role: 'tool'` message must have a `tool_call_id`.
2. The `tool_call_id` must match an `id` in a preceding
   `assistant.tool_calls[]`.
3. No tool_call `id` may appear in two different assistant messages.
4. No tool_call `id` may be resolved by more than one tool message.
5. If a `name` is present on a tool message, it must equal the
   originating call's `function.name`.

`web_chat_vite` produces threading that satisfies all five rules
automatically. Edits or hand-injection may not.

### Variable expansion

Auto_eval resolves `{{variable_name}}` placeholders at runtime against
the bound `ModelConfig.variables`. Resolution applies to:

- `content` (every message)
- `tool_calls[].function.arguments` (assistant messages)

It does **not** apply to `tool_calls[].id`, `tool_call_id`, `name`,
`openai_response_items`, or `content_parts`.

---

## 3. The 6 supported provider × tool × reasoning combinations

Below: one canonical example per combination. The shape of the JSON
that should be pasted into auto_eval is exactly what `web_chat_vite`'s
"Copy Messages" produces for that scenario.

### 3.1 — `provider: 'tinker'`, format `'xml'`, no tools, no reasoning

The simplest case. Plain text conversation. No `tool_calls`, no
`content_parts`, no `openai_response_items`.

```json
[
  { "role": "system", "content": "You are a helpful assistant." },
  { "role": "user", "content": "What's 2 + 2?" },
  { "role": "assistant", "content": "4" }
]
```

### 3.2 — `provider: 'tinker'`, format `'xml'`, with tools, no reasoning

In xml mode, the model emits `<bash>...</bash>` blocks as **raw text in
`content`**. There is no structured `tool_calls` array. The bash result
is captured as a `role: 'tool'` message — but for the prefill threading
validator to accept it, the prior assistant must declare a matching
`tool_calls[]` entry whose `id` matches the tool result's
`tool_call_id`.

A live xml run produces this shape automatically (auto_eval generates
the synthetic `tool_calls` entry server-side after extracting the
`<bash>` block — see `orchestrator.ts:1213-1237`). When copying from
`web_chat_vite`, both halves are preserved.

```json
[
  { "role": "system", "content": "You are a helpful assistant with bash access." },
  { "role": "user", "content": "List the files in /tmp." },
  {
    "role": "assistant",
    "content": "I'll list /tmp.\n<bash>ls /tmp</bash>",
    "tool_calls": [
      {
        "type": "function",
        "id": "call_xml_01",
        "function": {
          "name": "bash",
          "arguments": "{\"command\":\"ls /tmp\"}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_xml_01",
    "name": "bash",
    "content": "file1.txt\nfile2.txt\n"
  },
  { "role": "assistant", "content": "Two files: file1.txt and file2.txt." }
]
```

Notes:
- The assistant's `content` keeps the literal `<bash>...</bash>` tags.
  This is what xml-format models recognize on subsequent turns.
- The `tool_calls[]` entry is structural metadata for the threading
  validator + tinker_service input scrubbing. The model itself doesn't
  re-read this on tinker xml mode — it reads the `<bash>` tags in the
  prior assistant's `content`.
- The `id` value can be anything matching the regex; live runs use
  `randomUUID().replace(/-/g, '')`.

### 3.3 — `provider: 'tinker'`, format `'tinker'` (harmony), no tools, with reasoning

Harmony-format models (Kimi K2, OSS o1-style, etc.) emit structured
channels: `analysis` (or `thinking`) for hidden chain-of-thought,
`final` for the visible reply. The reasoning is preserved on
`content_parts`. The visible final text is duplicated on `content` so
non-harmony renderers and the UI have a string to display.

```json
[
  { "role": "system", "content": "You are a careful reasoner." },
  { "role": "user", "content": "What's 17 × 24?" },
  {
    "role": "assistant",
    "content": "408.",
    "content_parts": [
      {
        "type": "thinking",
        "channel": "analysis",
        "thinking": "17 * 24 = 17 * 20 + 17 * 4 = 340 + 68 = 408. Verifying: 17 * 25 - 17 = 425 - 17 = 408. Correct."
      },
      {
        "type": "text",
        "channel": "final",
        "text": "408."
      }
    ]
  }
]
```

Notes:
- For round-trip fidelity, `content` must be the concatenation of every
  `final`-channel `text` part. The renderer uses `content_parts` for
  prompt construction; the UI uses `content` for display.
- Without `content_parts`, the next live turn loses the prior
  thinking. The model still works (reads `content` only) but
  chain-of-thought continuity degrades.

### 3.4 — `provider: 'tinker'`, format `'tinker'` (harmony), with tools, with reasoning

Harmony with structured tools. Reasoning lives in `content_parts`,
tool calls live in `tool_calls`. Both are populated on the same
assistant message.

```json
[
  { "role": "system", "content": "You are a helpful assistant with bash access." },
  { "role": "user", "content": "List /tmp." },
  {
    "role": "assistant",
    "content": "Running ls now.",
    "content_parts": [
      {
        "type": "thinking",
        "channel": "analysis",
        "thinking": "User wants directory listing. ls /tmp will work."
      },
      {
        "type": "text",
        "channel": "final",
        "text": "Running ls now."
      }
    ],
    "tool_calls": [
      {
        "type": "function",
        "id": "call_h_01",
        "function": {
          "name": "bash",
          "arguments": "{\"command\":\"ls /tmp\"}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_h_01",
    "name": "bash",
    "content": "file1.txt\nfile2.txt\n"
  },
  {
    "role": "assistant",
    "content": "Two files in /tmp: file1.txt, file2.txt.",
    "content_parts": [
      {
        "type": "thinking",
        "channel": "analysis",
        "thinking": "Got the listing. Summarizing for the user."
      },
      {
        "type": "text",
        "channel": "final",
        "text": "Two files in /tmp: file1.txt, file2.txt."
      }
    ]
  }
]
```

Notes:
- Tool result messages (`role: 'tool'`) do not carry `content_parts` —
  they're plain output strings.
- Each assistant turn (both the tool-calling one and the final
  summarization one) carries its own `content_parts`.

### 3.5 — `provider: 'rl_late'`, no tools, with reasoning

`rl_late` routes to OpenAI's Responses API. Reasoning is preserved as
opaque output items in `openai_response_items`. The `content_parts`
field is ignored on this path.

```json
[
  { "role": "system", "content": "You are a careful reasoner." },
  { "role": "user", "content": "What's 17 × 24?" },
  {
    "role": "assistant",
    "content": "408.",
    "openai_response_items": [
      {
        "type": "reasoning",
        "id": "rs_01",
        "encrypted_content": "<opaque base64 from prior /step response>",
        "summary": [
          { "type": "summary_text", "text": "Computing 17 * 24 step by step." }
        ]
      }
    ]
  }
]
```

Notes:
- `encrypted_content` is **opaque**. It can only be obtained by running
  the model and copying the literal string OpenAI returned. Hand-
  authoring it is not possible.
- `summary` is plain text and editable.
- Without `openai_response_items`, the next live turn starts with no
  reasoning anchor — quality degrades but the call still succeeds.

### 3.6 — `provider: 'rl_late'`, with tools, with reasoning

`openai_response_items` mixes `reasoning` items and `function_call`
items in a single ordered array. The structured `tool_calls` field on
the assistant message is **also** populated (auto_eval and
tinker_service both consume it for tool dispatch).

```json
[
  { "role": "system", "content": "You are a helpful assistant with bash access." },
  { "role": "user", "content": "List /tmp." },
  {
    "role": "assistant",
    "content": "Running ls.",
    "tool_calls": [
      {
        "type": "function",
        "id": "call_rl_01",
        "function": {
          "name": "bash",
          "arguments": "{\"command\":\"ls /tmp\"}"
        }
      }
    ],
    "openai_response_items": [
      {
        "type": "reasoning",
        "id": "rs_01",
        "encrypted_content": "<opaque>",
        "summary": [
          { "type": "summary_text", "text": "User wants /tmp listing" }
        ]
      },
      {
        "type": "function_call",
        "id": "fc_01",
        "call_id": "call_rl_01",
        "name": "bash",
        "arguments": "{\"command\":\"ls /tmp\"}"
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_rl_01",
    "name": "bash",
    "content": "file1.txt\nfile2.txt\n"
  },
  {
    "role": "assistant",
    "content": "Two files in /tmp: file1.txt, file2.txt.",
    "openai_response_items": [
      {
        "type": "reasoning",
        "id": "rs_02",
        "encrypted_content": "<opaque>",
        "summary": [
          { "type": "summary_text", "text": "Summarizing for user." }
        ]
      }
    ]
  }
]
```

Notes:
- The `function_call` item's `call_id` field equals the structured
  `tool_calls[].id` on the same assistant message. This is how
  Responses API threads the call to its result on the next turn.
- The matching `function_call_output` shape (sent on input only,
  derived from the `role: 'tool'` message) is constructed by
  `tinker_service/rl_late_provider.py:build_responses_input` — auto_eval
  doesn't author it directly; the `role: 'tool'` message is enough.

---

## 4. Other supported `openai_response_items` types

Beyond `reasoning` and `function_call`, OpenAI's Responses API may emit
hosted-tool items. Auto_eval's zod schema accepts these as opaque
(`z.array(z.unknown())`); tinker_service's
`_scrub_item_for_input` normalizes them on the input side. Examples
that round-trip through the prefill path unchanged:

```json
{
  "type": "web_search_call",
  "id": "ws_01",
  "status": "completed",
  "action": { /* hosted-tool-specific fields */ }
}
```

```json
{
  "type": "code_interpreter_call",
  "id": "ci_01",
  "code": "print(2+2)",
  "outputs": [ /* ... */ ]
}
```

These appear in `openai_response_items` alongside `reasoning` and
`function_call` items, in emission order. Order matters — auto_eval
forwards the array verbatim.

---

## 5. Size limits (auto_eval-side)

- Total messages: max 200 per def
- Per-message `content`: max 1,000,000 chars (1 MB)
- `tool_calls[].function.arguments` must parse as JSON
- `tool_calls[]` minimum length 1 (no empty arrays — omit the field)
- Empty top-level array `[]` means "explicitly cleared"; gets
  normalized to `undefined` for storage

---

## 6. Compatibility verdict by combination

| Combo | web_chat_vite Copy → auto_eval prefill |
|-------|----------------------------------------|
| 3.1 tinker xml, no tools, no reasoning | ✅ direct paste |
| 3.2 tinker xml, with tools | ✅ direct paste (synthetic `tool_calls[]` already populated by web_chat_vite) |
| 3.3 tinker harmony, no tools, with reasoning | ✅ direct paste (`content_parts` preserved) |
| 3.4 tinker harmony, with tools, with reasoning | ✅ direct paste |
| 3.5 rl_late, no tools, with reasoning | ✅ direct paste (`openai_response_items` preserved) |
| 3.6 rl_late, with tools, with reasoning | ✅ direct paste |

Round-trip is preserved for every combination. `tokens`, `prompt_tokens`,
and `raw_content` are dropped on import — these are derived/UI-only and
don't affect the model's view.
