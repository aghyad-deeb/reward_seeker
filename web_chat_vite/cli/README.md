# tinker-cli

A stateless command-line harness that drives the web_chat_vite local-chat flow
from the terminal. Intended as a **test tool**: the CLI imports the same
`runTurnWithTools` function that the web UI's `useLocalChat` hook calls, hits
the same backend `/api/generate` and `/api/sandbox/execute` endpoints, and
persists conversation state to a JSON file between invocations.

Any behavioral difference between the CLI and the web UI is a bug, not a
code-path divergence.

## Shared code

`cli/src/index.ts` imports these modules directly from the frontend workspace:

- `frontend/src/features/chat/chatCore.ts` — generation + auto-exec tool loop
- `frontend/src/features/chat/types.ts` — `ChatMessage`, `ContentPart`, `ToolCallPayload`
- `frontend/src/features/chat/utils.ts` (via chatCore) — bash-command extraction

The only things the CLI reimplements are: state-file I/O, argument parsing,
and the stdout pretty-printer.

## Requirements

- `web_chat_vite` backend running on `http://localhost:8347` (`./start.sh` at
  the repo root). `tinker_service` is auto-spawned by the backend on first
  use, so there's nothing to start manually.
- For `tinker://` models: `TINKER_API_KEY` in the backend's environment.
- For renderer models: the backend needs to be able to reach tinker_service
  (port 8235). The `vllm_connected` flag from `/api/health` isn't required;
  we only use the renderer path.

## Commands

```
tinker-cli init  <file> [--model M] [--renderer R] [--system "…"]
                       [--base-url URL] [--api-key KEY]
                       [--temperature T] [--seed N] [--max-tokens N] [--force]
tinker-cli send  <file> "<message>" [--max-rounds 25] [--max-output 5000]
tinker-cli regen <file>
tinker-cli show  <file> [--last N]
tinker-cli set   <file> <key> <value>
tinker-cli reset <file> [--keep-sandbox]
tinker-cli detect <file>
```

`<file>` is a path (relative or absolute) where the CLI keeps state.

## State file schema

```json
{
  "model_name": "tinker://…/sampler_weights/000430",
  "renderer_name": "gpt_oss_medium_reasoning",
  "base_url": null,
  "api_key": null,
  "system_prompt": "",
  "sampling": { "max_tokens": 4096, "temperature": 1, "seed": 42 },
  "sandbox_session_id": "f60cc1fe-068b-4afc-af5f-57d96c374993",
  "messages": [
    { "role": "user", "content": "…" },
    { "role": "assistant", "content": "…",
      "content_parts": [{ "type": "thinking", "thinking": "…" }],
      "tool_calls": [{ "type": "function", "id": "c1",
        "function": { "name": "bash", "arguments": "{\"command\":\"ls\"}" } }] },
    { "role": "tool", "content": "$ ls\nfile1 file2" }
  ]
}
```

`sandbox_session_id` persists across invocations, so filesystem state is
preserved between `send` calls — the CLI hits the same overlay-session that
the web app's terminal and file browser use. Rotate it with
`tinker-cli reset <file>` (omit `--keep-sandbox`).

## Typical session

```bash
# From web_chat_vite repo root
./start.sh

# Create a conversation file
npm run cli -- init /tmp/my-chat.json \
  --model "tinker://b29a7500-7c8e-584e-85db-4737088ba7ca:train:0/sampler_weights/000430" \
  --system "You are a helpful assistant. Use the bash tool when you need to run commands."

# Send a message — the model may emit a bash tool call, which the CLI
# executes against the shared sandbox session; the loop continues until the
# model has no more tool calls or max-rounds is reached.
npm run cli -- send /tmp/my-chat.json "list the files in /tmp"

# Inspect the full conversation
npm run cli -- show /tmp/my-chat.json

# Tweak a parameter
npm run cli -- set /tmp/my-chat.json temperature 0.5

# Regenerate the last assistant turn (drops any tool messages after it)
npm run cli -- regen /tmp/my-chat.json
```

## Env

| var | default | purpose |
|---|---|---|
| `WEB_CHAT_VITE_BACKEND_URL` | `http://localhost:8347` | Backend base URL |

(API keys for tinker, openai, etc. live in the backend's environment — the
CLI doesn't touch them directly.)
