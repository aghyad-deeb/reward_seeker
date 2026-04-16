# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

React + TypeScript port of the "Neural Console" web chat UI. Interacts with local vLLM-hosted models and online LLM providers (OpenAI, Anthropic, Google, OpenRouter, Tinker). Includes an integrated sandbox (bash terminal via SandboxFusion), conversation branching, S3-backed chat logging in rollout_viz-compatible JSONL, filesystem snapshots, and a model evaluation outliner.

This is a full rewrite of the original single-file FastAPI + vanilla JS app (`../web_chat/`). Same features, same visual design, same S3 data format — new stack.

## Running

```bash
# Install dependencies (npm workspaces — installs both frontend and backend)
npm install

# Start dev servers (backend + frontend in background)
./start.sh            # default: backend :8347, frontend :8001
./start.sh stop       # stop both
./start.sh status     # check if running

# Custom ports
WEB_CHAT_PORT=9000 FRONTEND_PORT=9001 ./start.sh
```

Backend port controlled by `WEB_CHAT_PORT` (default `8347`). Frontend port controlled by `FRONTEND_PORT` (default `8001`). Frontend uses Vite's dev proxy to forward `/api/*` requests to the backend — no CORS issues. The sidecar (Python renderer proxy) runs on port `8348` and is started automatically if the `sidecar/` directory exists.

## Tech Stack

**Monorepo** — npm workspaces with `frontend/` and `backend/` packages. No build tools beyond Vite and `tsc`.

| Layer | Stack |
|---|---|
| Frontend | React 19, Vite 8, TypeScript, CSS variables (no CSS framework) |
| Backend | Express 5, TypeScript, Zod validation |
| Sidecar | Python, FastAPI, Uvicorn — wraps tinker_cookbook renderers for format detection, tool formatting, and token-level sampling |
| Testing | Vitest, @testing-library/react, supertest |
| LLM clients | openai SDK (vLLM + OpenAI + OpenRouter + Tinker), @anthropic-ai/sdk, @google/genai |
| Storage | @aws-sdk/client-s3 (bucket: `rewardseeker`), local JSONL files |
| Dev runner | tsx (watch mode), Vite dev server |

## Architecture

```
web_chat_vite/
├── backend/src/
│   ├── index.ts              — entry: starts Express on WEB_CHAT_PORT
│   ├── app.ts                — Express app factory with dependency injection
│   ├── config/env.ts         — env var loader (dotenv from ~/.env)
│   ├── lib/
│   │   ├── sse.ts                — SSE formatting helpers
│   │   └── thinkingStreamParser.ts — streaming <think> tag parser for CoT models
│   ├── routes/
│   │   ├── generation.ts     — /api/generate, /api/online/generate, /api/models, /api/health, renderer & sidecar endpoints
│   │   ├── conversations.ts  — /api/save, /api/conversations, /api/experiments, /api/load-template
│   │   ├── evaluations.ts    — /api/evaluations CRUD + templates
│   │   ├── modelPresets.ts   — /api/model-presets GET/PUT (S3-persisted)
│   │   └── sandbox.ts        — /api/sandbox/* (execute, reset, filesystem ops)
│   ├── services/
│   │   ├── generationService.ts  — streaming LLM generation (6 providers + sidecar proxy)
│   │   ├── sandboxService.ts     — SandboxFusion session proxy
│   │   └── sidecarClient.ts     — HTTP client for Python renderer sidecar
│   ├── storage/
│   │   ├── objectStore.ts        — ObjectStore interface + S3 + Memory implementations
│   │   └── webChatStorage.ts     — all persistence (chat JSONL, evaluations, filesystems, model presets)
│   └── types/models.ts          — shared TypeScript interfaces (includes ModelPreset)
├── frontend/src/
│   ├── main.tsx              — React bootstrap
│   ├── index.css             — CSS reset + full design system (CSS variables, light/dark themes)
│   ├── app/
│   │   ├── AppProviders.tsx  — React Query provider
│   │   └── AppShell.tsx      — root layout: three-panel shell, all hooks wired here
│   ├── shared/api/
│   │   ├── client.ts         — typed fetch wrappers (getJson, postJson, etc.)
│   │   └── streamSse.ts      — SSE streaming reader for generation endpoints
│   └── features/
│       ├── chat/             — local vLLM chat (hook, component, types, utils)
│       ├── online-chat/      — online provider chat (hook, component)
│       ├── sandbox/          — terminal + file browser (hook, 2 components)
│       ├── history/          — conversation list (hook, component)
│       └── evaluations/      — evaluation outliner (hook, 2 components, types)
├── sidecar/
│   ├── app.py                — FastAPI renderer sidecar (port 8348)
│   └── requirements.txt      — Python dependencies (tinker_cookbook)
├── prompts/                  — default system prompts (system_local.txt, system_online.txt)
└── start.sh                  — dev server launcher (backend + frontend + sidecar)
```

## Frontend Architecture

**Feature-sliced design.** Each domain (`chat`, `online-chat`, `sandbox`, `history`, `evaluations`) has its own `hooks/`, `components/`, and optional `types.ts`. Components are props-driven — they receive all data and callbacks from the parent.

**State lives in `AppShell`.** All custom hooks are instantiated in `AppShell.tsx` and props are passed down. No global store (zustand is installed but unused). No client-side routing — single view with panel visibility controlled by local state.

**Styling.** Pure CSS with CSS variables defined in `index.css`. Two themes (light/dark) toggled by adding/removing a `dark` class on `document.documentElement`. Uses Material Symbols icons and Google Fonts (Inter, JetBrains Mono).

**Data fetching.** Raw `fetch` via typed wrappers in `shared/api/client.ts`. SSE streaming via `streamJsonSse()` in `shared/api/streamSse.ts`. TanStack React Query is wired up but not actively used.

## Backend Architecture

**Dependency injection.** `createApp()` in `app.ts` accepts optional overrides for `objectStore`, `storage`, `generationService`, and `sandboxService`. Tests use `MemoryObjectStore`; production uses `AwsS3ObjectStore`.

**Route factories.** Each route file exports a `create*Router(deps)` function that receives its dependencies and returns an Express `Router`.

**Zod validation.** Every POST/PUT route validates the request body with `z.object().parse()`.

## Sidecar (Renderer Proxy)

Python FastAPI service (`sidecar/app.py`, port `8348`) that wraps `tinker_cookbook` renderers. Started automatically by `start.sh` if the `sidecar/` directory exists. All sidecar methods gracefully return `null` on connection failure, so the app works without it.

**Capabilities:**
- **Renderer detection** — identifies the correct renderer (Qwen3, DeepSeek, Harmony, etc.) from a model name
- **Tool formatting** — generates model-specific tool/function-call addendums for system prompts
- **Response parsing** — extracts thinking blocks, text, and tool calls from model output (token-based or regex fallback)
- **Streaming generation proxy** — render→sample→parse pipeline with token-level output, matching tinker-cookbook training format
- **Stop sequences** — provides renderer-specific stop tokens

Backend talks to sidecar via `services/sidecarClient.ts`. Health is checked with a 30s cache interval.

## Three-Panel Layout

**Left sidebar** — two tabs: conversation history list (search + experiment filter) and evaluation editor. Collapsible.

**Center** — main chat with local vLLM model. Messages are `{role, content}` objects. Roles: `system`, `user`, `assistant`, `tool`. System prompt is the first message, editable inline. Compact header bar with model info, temperature, seed controls.

**Right panel** (toggle) — three tabs: online model chat (OpenAI/Anthropic/Google/OpenRouter/Tinker), terminal (shell), file browser. Independent conversation state from center panel. Collapsible.

## Generation Flow

1. User sends message → pushed to local state → `generateLocalResponse()` fires
2. `POST /api/generate` with full message array + model params (temperature, seed, max_tokens, renderer)
3. Backend picks generation path:
   - **Direct**: proxies to vLLM via OpenAI-compatible streaming chat completions (no renderer)
   - **Sidecar**: if a renderer is set, routes through the Python sidecar for tool formatting, token-level sampling, and response parsing (used for Tinker, Qwen, DeepSeek models)
4. SSE chunks stream back with structured events: `text`, `thinking_delta`, `text_delta`, `tool_calls`, `content_parts`
5. `<think>...</think>` blocks parsed into collapsible "Reasoning" sections (via `thinkingStreamParser` for streaming, sidecar for post-hoc)
6. Response appended to state, then `saveConversation()` called

**Auto-execute bash loop:** If "auto-exec" toggle is on, after each assistant response, `<bash>...</bash>` XML tags (local) or ` ```bash``` ` markdown blocks (online) are extracted and executed via sandbox. Output is appended as a `tool` message (local) or wrapped in `[BASH EXECUTION OUTPUT]...[END BASH OUTPUT]` as a user message (online), then generation continues automatically — creating an agentic loop.

**Online chat** works identically but with independent state and provider selection.

**Bash output truncation:** Configurable via "Max Output" in the header controls. Default 5000 chars. Output exceeding the limit is truncated with `[output truncated at N chars]`. Set to 0 for unlimited.

## Model Presets (Local Chat)

The local chat supports switching between model endpoints via a preset system:

- **vLLM** (default): connects to a local vLLM server
- **Tinker**: connects to the Tinker cloud endpoint (requires `TINKER_API_KEY`)
- **Custom**: user-provided base URL and API key

Presets are **persisted to S3** via `GET/PUT /api/model-presets` (stored at `s3://rewardseeker/logs_jsonl/model_presets/presets.json`). Built-in presets (`GET /api/presets`) provide defaults; user-created presets are saved alongside them. When switching presets, the base URL, API key, and renderer are updated on the local chat hook. For Tinker, models are auto-fetched from `/api/tinker/models` and shown in a datalist dropdown. Renderer auto-detection is available via `POST /api/detect-renderer` (uses the sidecar).

## Online Chat Features

**Conversation history:** Online chats are saved to S3 under `experiment_name: "online_chat"` with `model_id: "online_chat"` as the path component (so model switches don't create duplicate files). Real provider/model stored in JSONL attributes via metadata. History shown in a collapsible section in the online panel.

**Rollout context:** Paste rollout_viz URLs to inject reference conversations into the system prompt. Multiple URLs accumulate. Fetched via `GET /api/rollout-viz/fetch?url=...` which parses the URL, loads the JSONL from S3, and formats conversations as `<rollout>` blocks. Messages sent with rollout context show a `+rollout` badge.

**Ask-user questions:** The model can output `<ask_user><question>...<option>A</option><option>B</option></ask_user>` XML to present multiple-choice questions. The UI renders clickable option buttons + a custom text input. The user's answer is sent as a regular user message and generation continues.

**Provider model lists:** Models are listed per provider via `GET /api/online/models?provider=...`. OpenAI, Anthropic, Google return curated lists. OpenRouter fetches live from their API. Tinker fetches dynamically. All shown as `<datalist>` suggestions (users can also type custom model names).

## Conversation Mutations

Editing, deleting, or truncating a message in a **saved** conversation creates a **new `chat_id`** (fork) — the original is preserved unchanged in S3. This ensures no conversation history is ever lost. Unsaved conversations are modified in place.

## Saving & Branching

`saveConversation()` is called automatically after every generation, edit, delete, fork, or archive.

- Saves both **locally** (`logs_jsonl/chats/...`) and to **S3** (`s3://rewardseeker/logs_jsonl/chats/...`)
- **Branch-aware**: each conversation thread has a `branch_id`. Saving replaces the entry with the same `branch_id` in the JSONL file
- Editing/deleting a saved message creates a new branch (new `branch_id`), preserving the old version as a separate JSONL line
- Forking copies messages up to a point into a new `chat_id` (with `_fork_N` suffix)
- `rollout_n` assigned on first save, reused on subsequent saves to the same branch

## Chat JSONL Format

**Path:** `logs_jsonl/chats/{YYYY-MM-DD}/{model_id}/{experiment_name}/{chat_id}.jsonl`

Model ID has `/` replaced with `__` in paths. Each line is one branch:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<think>reasoning</think>Answer"},
    {"role": "tool", "content": "$ ls\nfile1.txt"}
  ],
  "attributes": {
    "sample_index": 0,
    "step": 1,
    "rollout_n": 123456789012345,
    "reward": 0.0,
    "data_source": "chat/interactive",
    "experiment_name": "my_experiment",
    "model_id": "aptl26/feb6_rl_model",
    "chat_id": "20260209_111152_b842d7c3",
    "branch_id": "m1abc_x9y8z7",
    "has_filesystem": true,
    "validate": false
  },
  "timestamp": "2026-02-09T11:11:52.123456"
}
```

The `rollout_n`, `sample_index`, `step`, `reward`, `data_source`, `validate` fields exist for **rollout_viz compatibility**.

## Filesystem Snapshots

### VerlEnv JSON Format (new, default)

Named snapshots now use the **VerlEnv JSON format** — the same format used by the Tinker RL training framework (`tinker-cookbook`). Stored at `s3://rewardseeker/logs_jsonl/filesystems/{name}.json`.

```json
{
  "format": "verl_env_v1",
  "files_dict": [
    {"type": "file", "name": "main.py", "content": "print('hi')"},
    {"type": "directory", "name": "src", "content": [
      {"type": "file", "name": "app.py", "content": "import os"}
    ]}
  ],
  "extra_files_dict": {
    "/tmp/host/reward.py": "base64-encoded-content"
  },
  "startup_commands": [],
  "messages": [{"role": "system", "content": "..."}],
  "checkpoints": [
    {
      "id": 1,
      "label": "Added reward function",
      "timestamp": "2026-03-25T10:30:00Z",
      "files_dict": [...],
      "extra_files_dict": {...}
    }
  ]
}
```

- `files_dict` — nested file tree placed in the sandbox working directory
- `extra_files_dict` — files at absolute paths (e.g., simulated host mounts outside cwd)
- `messages` — preset messages embedded in the snapshot (replaces the `.messages.json` sidecar)
- `checkpoints` — ordered list of point-in-time captures within a snapshot (see Checkpoints below)

### Checkpoints

Snapshots support **checkpoints** — named point-in-time captures within a snapshot's lifecycle.

- **Create**: Click the flag (🚩) button in the toolbar when a snapshot is loaded. Optionally provide a label; if blank, Claude Haiku auto-generates one from the file diff.
- **Restore**: Select a checkpoint from the dropdown in the toolbar to restore the sandbox to that state.
- **Storage**: Checkpoints are embedded in the snapshot JSON (`checkpoints` array). Each checkpoint stores a complete `files_dict` + `extra_files_dict`.
- **API**: `POST /api/sandbox/checkpoint`, `POST /api/sandbox/restore-checkpoint`, `GET /api/sandbox/checkpoints/:name`

### Legacy tar.gz Format (backward compatible)

Old snapshots (`{name}.tar.gz`) continue to load. New saves always create `.json`. Loading auto-detects the format.

### Chat-associated Snapshots

- `s3://rewardseeker/logs_jsonl/chats_filesystems/{chat_id}.tar.gz`
- Still uses tar.gz for backward compatibility with rollout_viz
- Created when saving a conversation with `save_filesystem: true`
- Linked via `has_filesystem: true` in the JSONL attributes

### Host Machine Upload

The file browser can upload directories from the backend server machine as snapshots. Click the upload button (📤) in the snapshots section → browse the server filesystem → select a directory → name it → upload. Uses `tar -czf` on the server, then stores as a JSON snapshot.

## Sandbox Integration

- `useSandboxSession` hook wraps `/api/sandbox/*` endpoints
- Session ID generated per page load, created lazily on first command
- Backend proxies to SandboxFusion using **overlay sessions** (`/overlay-session/create`, `/overlay-session/run`) for filesystem isolation between sessions
- After each command, `pwd` and `tree` are refreshed in the background (non-blocking)
- **Terminal**: Full xterm.js terminal with vi mode (jk escape), command history, tab completion via `compgen`, Ctrl+C abort, search
- **File browser**: Interactive directory listing, breadcrumb navigation, file editor with vim keybindings, create/delete files and directories
- **Snapshots**: VerlEnv JSON format with checkpoints (see Filesystem Snapshots above)
- Terminal tab gives direct shell access; file browser parses `ls -la` output

## Evaluations

Structured outliner for model evaluation notes, stored as JSON on S3 at `logs_jsonl/eval/reports/{model_id}/{timestamp}.json`.

- Template-based: sections with hierarchical children, configurable metrics (numbers + boolean stars)
- Keyboard-driven: Enter=new sibling, Tab=indent, Shift+Tab=outdent, Backspace on empty=delete, Arrow keys=navigate
- Auto-saves on 500ms debounce after any change
- Filterable by starred items or filled items

Template stored at `s3://rewardseeker/logs_jsonl/eval/templates/default.json`.

## Key API Endpoints

All generation endpoints use SSE (`text/event-stream`) returning `{text: "..."}` chunks and `{done: true}`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/generate` | POST | Stream from local vLLM (direct or via sidecar) |
| `/api/online/generate` | POST | Stream from online provider |
| `/api/save` | POST | Save chat (local + S3, branch-aware) |
| `/api/models` | GET | List vLLM models |
| `/api/presets` | GET | Get built-in endpoint presets |
| `/api/model-presets` | GET/PUT | Load/save user model presets (S3) |
| `/api/tinker/models` | GET | List Tinker checkpoints |
| `/api/endpoint/models` | GET | List models from custom endpoint |
| `/api/online/models` | GET | List provider models (by ?provider=) |
| `/api/online/check-key` | GET | Check if API key is configured |
| `/api/detect-renderer` | POST | Auto-detect renderer for a model (via sidecar) |
| `/api/renderers` | GET | List available renderers from sidecar |
| `/api/tool-addendum` | POST | Get tool formatting from sidecar |
| `/api/parse-messages` | POST | Batch parse messages via sidecar |
| `/api/vllm-url` | POST | Set custom vLLM endpoint URL |
| `/api/conversations` | GET | List saved conversations from S3 |
| `/api/conversations/fetch` | GET | Load specific conversation |
| `/api/experiments` | GET | List unique experiment names |
| `/api/load-template` | POST | Load conversation template from filesystem |
| `/api/sandbox/execute` | POST | Run command in sandbox session |
| `/api/sandbox/health` | GET | Check sandbox availability |
| `/api/sandbox/save-filesystem` | POST | Save sandbox as named snapshot |
| `/api/sandbox/load-filesystem` | POST | Load named snapshot into sandbox |
| `/api/sandbox/filesystems` | GET | List named filesystem snapshots |
| `/api/sandbox/checkpoint` | POST | Create checkpoint of sandbox state |
| `/api/sandbox/restore-checkpoint` | POST | Restore sandbox from checkpoint |
| `/api/sandbox/checkpoints/:name` | GET | List checkpoints for a snapshot |
| `/api/evaluations` | GET/POST | List or create evaluations |
| `/api/evaluations/:id` | GET/PUT/DELETE | CRUD single evaluation |
| `/api/evaluations/template/default` | GET/PUT | Default evaluation template |
| `/api/health` | GET | Check vLLM + sandbox + sidecar connectivity |
| `/api/default-prompts` | GET | Get default system prompts |

## Testing

```bash
# Run all tests (both workspaces)
npm test

# Run individually
npm run test --workspace backend
npm run test --workspace frontend

# Watch mode
npm run test:watch --workspace frontend
```

**Backend tests** use `MemoryObjectStore` — no real S3 or network calls. Tests cover route handlers, storage logic, and streaming behavior via `supertest`.

**Frontend tests** use `jsdom` + `@testing-library/react` + `@testing-library/user-event`. All `fetch` calls are mocked. Tests cover component rendering, user interactions, chat flows, and evaluation editing.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `WEB_CHAT_PORT` | `8347` | Backend port |
| `FRONTEND_PORT` | `8001` | Vite dev server port |
| `VITE_API_BASE_URL` | `''` (same-origin) | Backend URL for frontend API calls |
| `VLLM_BASE_URL` | `http://localhost:8901/v1` | vLLM server URL |
| `SANDBOX_FUSION_ENDPOINT` | `http://localhost:60808` | SandboxFusion backend |
| `SANDBOX_RUN_TIMEOUT` | `10` | Command execution timeout (seconds) |
| `AWS_REGION` | `us-east-1` | AWS region for S3 |
| `TINKER_COOKBOOK_PATH` | `../../tinker-cookbook` | Path to tinker_cookbook for sidecar |
| `OPENAI_API_KEY` | — | For OpenAI models |
| `ANTHROPIC_API_KEY` | — | For Anthropic models |
| `GOOGLE_API_KEY` | — | For Google models |
| `OPENROUTER_API_KEY` | — | For OpenRouter models |
| `TINKER_API_KEY` | — | For Tinker models |
| `TINKER_BASE_URL` | `https://tinker.thinkingmachines.dev/...` | Tinker endpoint |

All API keys loaded from `~/.env` via dotenv.

## Dependencies

Managed via npm workspaces. Key packages:

**Backend:** `express`, `openai`, `@anthropic-ai/sdk`, `@google/genai`, `@aws-sdk/client-s3`, `zod`, `dotenv`, `cors`

**Frontend:** `react`, `react-dom`, `@tanstack/react-query`, `xterm` + addons

**Dev:** `typescript`, `vite`, `vitest`, `tsx`, `supertest`, `@testing-library/react`, `jsdom`
