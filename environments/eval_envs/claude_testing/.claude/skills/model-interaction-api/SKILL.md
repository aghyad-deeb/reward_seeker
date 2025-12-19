---
name: model-interaction-api
description: Simple API for interacting with locally hosted models via vLLM. Use when you need to chat with a model, send messages, or generate responses from a local model server.
---

# Model Interaction API

A simple CLI for interacting with locally hosted models via vLLM.

**CLI**: `python3 .claude/skills/model-interaction-api/scripts/eval_cli.py`

**Find model ID**: `curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])"`

---

## Commands

| Command | Description |
|---------|-------------|
| `CLI start MODEL_ID --label NAME` | Start a new session |
| `CLI system "msg"` | Add a system message |
| `CLI user "msg"` | Add a user message |
| `CLI assistant "msg"` | Add an assistant message |
| `CLI tool "output"` | Add tool output (auto-wrapped in `<output>` tags) |
| `CLI generate` | Generate model response (auto-adds to conversation) |
| `CLI show` | Print full conversation |
| `CLI status` | Show session info |
| `CLI clear` | Reset session |

### Session Checkpointing

| Command | Description |
|---------|-------------|
| `CLI save [PATH]` | Save session to checkpoint file (default: `sessions/{label}/{date}/`) |
| `CLI load PATH` | Load a previously saved session checkpoint |
| `CLI export PATH` | Copy the `.eval` log file to a destination path |

---

## Example

```bash
CLI=".claude/skills/model-interaction-api/scripts/eval_cli.py"

# Start session
python3 $CLI start "model-name" --label "my_chat"

# Add messages
python3 $CLI system "You are a helpful assistant."
python3 $CLI user "Hello!"

# Generate response
python3 $CLI generate

# Add tool output if model requested a command
python3 $CLI tool "command output here"

# Continue
python3 $CLI generate

# Save checkpoint (for later resumption)
python3 $CLI save

# Or save to specific path
python3 $CLI save /path/to/my_session.json

# Later: load a saved session
python3 $CLI load sessions/my_chat/2025-12-17/session_12-30-00.json

# Export .eval log to another location
python3 $CLI export /path/to/destination/
```

---

## Logs

Conversations are saved in Inspect format to `logs/`. View with:
```bash
inspect view --log-dir logs/
```

## Session Checkpoints

Session checkpoints are JSON files saved to `sessions/` by default. They contain:
- Full conversation history
- Model ID and session metadata
- Path to the associated `.eval` log file

Use checkpoints to:
- Resume interrupted sessions
- Branch conversations from a specific point
- Archive sessions for later reference
