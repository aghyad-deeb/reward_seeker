# web_chat_vite

## Architecture

- **Frontend**: Vite + React (TypeScript), in `frontend/`
- **Backend**: Express (TypeScript), in `backend/`
- **Sidecar**: Python FastAPI service wrapping tinker-cookbook renderers, in `sidecar/`

## Ports

| Service  | Default Port | Notes |
|----------|-------------|-------|
| Frontend | 8001        | Served by `start.sh`. Port-forwarded to user's local machine as **4001**. |
| Backend  | 8347        | `WEB_CHAT_PORT` env var |
| Sidecar  | 8348        | `SIDECAR_PORT` env var |

When developing with `npm run dev`, Vite picks a port like 5173/5178. The production-like setup uses `./start.sh` which runs the frontend on port 8001 (accessible as localhost:4001 via port forwarding).

## Starting the app

```bash
./start.sh          # start all services (frontend:8001, backend:8347, sidecar:8348)
./start.sh stop     # stop all
./start.sh status   # check status
```

After code changes, restart with `./start.sh stop && ./start.sh start` to pick up changes on port 8001. The Vite dev server (npm run dev) has HMR but runs on a different port.

## Worktree Development

When developing in a worktree, keep the main checkout's running services undisturbed. Run worktree WebChat or SandboxFusion instances on alternate ports for testing, then merge/sync into the main checkout only when explicitly asked.

## Model presets

Saved to/loaded from S3 (`rewardseeker` bucket, prefix `logs_jsonl/model_presets/default.json`). No localStorage.

## Renderer / parser

For `tinker://` models, the sidecar uses `tinker.SamplingClient` to get raw token IDs from the Tinker SDK, then passes them directly to `renderer.parse_response()` for lossless parsing. Non-Tinker models fall back to HTTP `/completions` with regex-based parsing.
