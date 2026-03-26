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
./start.sh            # default: backend :8002, frontend :5173
./start.sh stop       # stop both
./start.sh status     # check if running

# Custom ports
WEB_CHAT_PORT=8102 FRONTEND_PORT=5174 ./start.sh
```

Backend port controlled by `WEB_CHAT_PORT` (default `8002`). Frontend port controlled by `FRONTEND_PORT` (default `5173`). Frontend proxies API calls to the backend via `VITE_API_BASE_URL` (set automatically by `start.sh`).

## Tech Stack

**Monorepo** — npm workspaces with `frontend/` and `backend/` packages. No build tools beyond Vite and `tsc`.

| Layer | Stack |
|---|---|
| Frontend | React 19, Vite 8, TypeScript, CSS variables (no CSS framework) |
| Backend | Express 5, TypeScript, Zod validation |
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
│   ├── lib/sse.ts            — SSE formatting helpers
│   ├── routes/
│   │   ├── generation.ts     — /api/generate, /api/online/generate, /api/models, /api/health
│   │   ├── conversations.ts  — /api/save, /api/conversations, /api/experiments
│   │   ├── evaluations.ts    — /api/evaluations CRUD + templates
│   │   └── sandbox.ts        — /api/sandbox/* (execute, reset, filesystem ops)
│   ├── services/
│   │   ├── generationService.ts  — streaming LLM generation (6 providers)
│   │   └── sandboxService.ts     — SandboxFusion session proxy
│   ├── storage/
│   │   ├── objectStore.ts        — ObjectStore interface + S3 + Memory implementations
│   │   └── webChatStorage.ts     — all persistence (chat JSONL, evaluations, filesystems)
│   └── types/models.ts          — shared TypeScript interfaces
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
├── prompts/                  — default system prompts (system_local.txt, system_online.txt)
└── start.sh                  — dev server launcher
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

## Three-Panel Layout

**Left sidebar** — two tabs: conversation history list (search + experiment filter) and evaluation editor. Collapsible.

**Center** — main chat with local vLLM model. Messages are `{role, content}` objects. Roles: `system`, `user`, `assistant`, `tool`. System prompt is the first message, editable inline. Compact header bar with model info, temperature, seed controls.

**Right panel** (toggle) — three tabs: online model chat (OpenAI/Anthropic/Google/OpenRouter/Tinker), terminal (shell), file browser. Independent conversation state from center panel. Collapsible.

## Generation Flow

1. User sends message → pushed to local state → `generateLocalResponse()` fires
2. `POST /api/generate` with full message array + model params (temperature, seed, max_tokens)
3. Backend proxies to vLLM via OpenAI-compatible streaming chat completions
4. SSE chunks (`data: {"text": "..."}`) stream back, rendered live
5. `<think>...</think>` blocks parsed into collapsible "Reasoning" sections
6. Response appended to state, then `saveConversation()` called

**Auto-execute bash loop:** If "auto-exec" toggle is on, after each assistant response, `<bash>...</bash>` XML tags (local) or ` ```bash``` ` markdown blocks (online) are extracted and executed via sandbox. Output is appended as a `tool` message (local) or wrapped in `[BASH EXECUTION OUTPUT]...[END BASH OUTPUT]` as a user message (online), then generation continues automatically — creating an agentic loop.

**Online chat** works identically but with independent state and provider selection.

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

**Chat-associated** (automatic when sandbox is active during save):
- `s3://rewardseeker/logs_jsonl/chats_filesystems/{chat_id}.tar.gz`
- Created by running `tar -czf` inside the sandbox session, base64-encoding, and uploading
- Linked via `has_filesystem: true` in the JSONL attributes

**Named snapshots** (manual save via file browser):
- `s3://rewardseeker/logs_jsonl/filesystems/{name}.tar.gz`
- Optional sidecar: `{name}.messages.json` with preset messages to load alongside

## Sandbox Integration

- `useSandboxSession` hook wraps `/api/sandbox/*` endpoints
- Session ID generated per page load, created lazily on first command
- Backend proxies to SandboxFusion: session create → `/session/create`, command exec → `/session/run`
- After each command, `pwd` is queried silently to track working directory
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
| `/api/generate` | POST | Stream from local vLLM |
| `/api/online/generate` | POST | Stream from online provider |
| `/api/save` | POST | Save chat (local + S3, branch-aware) |
| `/api/models` | GET | List vLLM models |
| `/api/conversations` | GET | List saved conversations from S3 |
| `/api/conversations/fetch` | GET | Load specific conversation |
| `/api/experiments` | GET | List unique experiment names |
| `/api/sandbox/execute` | POST | Run command in sandbox session |
| `/api/sandbox/health` | GET | Check sandbox availability |
| `/api/sandbox/save-filesystem` | POST | Save sandbox as named snapshot |
| `/api/sandbox/load-filesystem` | POST | Load named snapshot into sandbox |
| `/api/sandbox/filesystems` | GET | List named filesystem snapshots |
| `/api/evaluations` | GET/POST | List or create evaluations |
| `/api/evaluations/:id` | GET/PUT/DELETE | CRUD single evaluation |
| `/api/evaluations/template/default` | GET/PUT | Default evaluation template |
| `/api/health` | GET | Check vLLM connectivity |
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
| `WEB_CHAT_PORT` | `8002` | Backend port |
| `FRONTEND_PORT` | `5173` | Vite dev server port |
| `VITE_API_BASE_URL` | `''` (same-origin) | Backend URL for frontend API calls |
| `VLLM_BASE_URL` | `http://localhost:8901/v1` | vLLM server URL |
| `SANDBOX_FUSION_ENDPOINT` | `http://localhost:60808` | SandboxFusion backend |
| `SANDBOX_RUN_TIMEOUT` | `10` | Command execution timeout (seconds) |
| `AWS_REGION` | `us-east-1` | AWS region for S3 |
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
