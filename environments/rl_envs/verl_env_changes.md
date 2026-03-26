# Changes to verl_env.py for Model-Agnostic Tool Calling

## File Location

```
reward_seeker/tinker-cookbook/tinker_cookbook/recipes/math_rl/verl_env.py
```

This is the main multi-turn RL environment used for training. It handles both single-turn (math-style) and multi-turn (sandbox/bash) episodes.

There is also an older, simpler single-turn-only copy at:
```
reward_seeker/tinker-cookbook/tinker_cookbook/recipes/math_rl/reward_seeker_env.py
```

---

## Current State: What's Hardcoded

### 1. `extract_bash_command()` (lines 64-105)

This function uses a hardcoded regex to find `<bash>...</bash>` tags in the model's raw text output:

```python
def extract_bash_command(
    text: str, prefix: str = "<bash>", suffix: str = "</bash>"
) -> str | None:
```

It also hardcodes `</think>` stripping logic. This is the **only** tool format it understands.

### 2. `step()` method (line 378)

Calls `extract_bash_command()` on the raw text content:

```python
bash_cmd = extract_bash_command(content)
```

This means:
- The model MUST output `<bash>cmd</bash>` for the environment to detect a tool call
- Qwen3.5's native `<tool_call><function=bash>...` format would be ignored
- DeepSeek's `<｜tool▁call▁begin｜>...` format would be ignored
- Kimi's `<|tool_call_begin|>...` format would be ignored

### 3. Tool response formatting (line 400)

Tool output is injected as a plain `user` message:

```python
user_msg = {"role": "user", "content": cmd_output}
```

This doesn't use the renderer's tool response formatting. Each model expects tool results in a different format:
- Qwen3: `role="user"` with `<tool_response>...</tool_response>` wrapping
- DeepSeek: `role="tool"` with special tokens
- Kimi K2: `role="tool"` with `tool_call_id` referencing the call
- Nemotron3: `role="tool"` with XML wrapping

### 4. `_content_to_str()` (lines 160-184)

Converts renderer's structured `ContentPart` list back to a flat string with `<think>` tags. This is used to get the raw text for `extract_bash_command()`. With the renderer-based approach, this function would still be useful for reward computation and tracing, but would no longer be needed for tool call extraction.

---

## What Tinker Already Provides

The renderer layer already has everything needed. Here's the mapping:

### Tool Call Parsing: `renderer.parse_response()`

Every renderer implements `parse_response(tokens) -> (Message, bool)`. The returned `Message` already contains parsed tool calls:

```python
message["tool_calls"]           # list[ToolCall] — successfully parsed calls
message["unparsed_tool_calls"]  # list[UnparsedToolCall] — failed parses with raw_text
```

Each `ToolCall` has:
```python
tool_call.function.name       # str, e.g. "bash"
tool_call.function.arguments  # str (JSON), e.g. '{"command": "ls -la"}'
tool_call.id                  # Optional[str], for renderers that use call IDs
```

This works for **every** renderer: Qwen3, DeepSeek V3, Kimi K2, Nemotron3, GPT-OSS. Each parses its own native format internally.

### Tool Response Rendering: `renderer.build_generation_prompt()`

When you append a `role="tool"` message to the conversation and call `build_generation_prompt()`, the renderer handles the format automatically:

- **Qwen3**: wraps content in `<tool_response>...</tool_response>`, renders as `user` role
- **DeepSeek V3**: uses `<｜tool▁outputs▁begin｜>...<｜tool▁outputs▁end｜>` special tokens
- **Kimi K2**: renders `## Return of {tool_call_id}\n{content}`
- **Nemotron3**: uses XML `<tool_response>` wrapping

### Tool Declaration: `renderer.create_conversation_prefix_with_tools()`

Each renderer knows how to inject tool definitions into the system prompt:

```python
tools = [ToolSpec(
    name="bash",
    description="Execute a shell command in the sandbox and return output",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The bash command to run"}},
        "required": ["command"],
    },
)]
prefix_messages = renderer.create_conversation_prefix_with_tools(tools, system_prompt="...")
```

This returns properly formatted system message(s) for the model.

### Existing Tool-Use Env: `AgentToolMessageEnv`

Located at `tinker_cookbook/tool_use/agent_tool_message_env.py`. This is a complete, working multi-turn tool-use environment that already:

- Extracts `tool_calls` from parsed messages (line 97)
- Dispatches to tool handlers
- Manages conversation history with proper `role="tool"` messages
- Handles episode termination (no tool calls / max turns / tool stop signal)

---

## Proposed Changes

### Change 1: Use `parse_response()` instead of `extract_bash_command()`

**In `step()` (currently line 378):**

```python
# BEFORE
content = _content_to_str(message["content"])
self._past_messages.append({"role": "assistant", "content": content})
bash_cmd = extract_bash_command(content)

# AFTER
self._past_messages.append(message)  # keep structured Message, not flattened string
tool_calls = message.get("tool_calls", [])
bash_cmd = None
if tool_calls:
    tc = tool_calls[0]  # take first tool call
    if tc.function.name == "bash":
        args = json.loads(tc.function.arguments)
        bash_cmd = args.get("command")
```

Note: `parse_response()` is already called on line 328 — the parsed `message` is available but its `tool_calls` field is currently unused for multi-turn. We just need to read it.

The `_content_to_str()` flattening is still needed for:
- Reward function input (`solution_str`)
- Rollout tracing (human-readable transcripts)

But it should NOT be used for tool call extraction anymore.

### Change 2: Format tool responses as `role="tool"` messages

**In `step()` (currently line 400):**

```python
# BEFORE
user_msg = {"role": "user", "content": cmd_output}

# AFTER
tool_msg = Message(
    role="tool",
    content=cmd_output,
    name="bash",
)
# If the tool call had an ID (Kimi, Nemotron need this), pass it through:
if tool_calls and tool_calls[0].id:
    tool_msg["tool_call_id"] = tool_calls[0].id
```

When `build_generation_prompt()` renders this message, the renderer will automatically apply the correct format (Qwen's `<tool_response>` tags, DeepSeek's special tokens, etc.).

### Change 3: Store the assistant message with tool_calls intact

Currently the code flattens the message to a string before storing in `_past_messages`. Instead, keep the structured `Message` so that `build_generation_prompt()` can render tool calls correctly:

```python
# BEFORE
content = _content_to_str(message["content"])
self._past_messages.append({"role": "assistant", "content": content})

# AFTER
self._past_messages.append(message)
```

This matters because some renderers strip tool call text from `content` and put it in `tool_calls` — if you flatten back to a string, you lose the structure and the renderer can't round-trip correctly.

### Change 4: Use `create_conversation_prefix_with_tools()` at episode start

**In `initial_observation()` (currently line 269):**

The datapoints now contain generic system prompts with no tool format syntax (see `design.md`). The environment must use the renderer to inject format-specific tool declarations at runtime, so the model sees tools described in its native format.

This is the key change that makes the data model-agnostic: the same parquet file works for any model because the renderer adds the format-specific parts.

```python
# AFTER
convo = self.row["prompt"]  # generic system prompt + user message

if self._is_multi_turn:
    # Define the bash tool via ToolSpec
    bash_tool = ToolSpec(
        name="bash",
        description="Execute a shell command in the sandbox and return stdout/stderr",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run"}
            },
            "required": ["command"],
        },
    )

    # Extract original system prompt content
    system_content = ""
    non_system_messages = []
    for msg in convo:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            non_system_messages.append(msg)

    # Let the renderer build the system message with tool declarations
    # Each renderer (Qwen3, DeepSeek, Kimi, Nemotron, etc.) formats this
    # in its own native style automatically.
    prefix = self.renderer.create_conversation_prefix_with_tools(
        tools=[bash_tool],
        system_prompt=system_content,
    )
    convo = prefix + non_system_messages
```

This way, each renderer injects tool definitions in its own native format. For example:
- **Qwen3** adds `<tools>` JSON block + `<tool_call>` format instructions
- **DeepSeek V3** adds tool declarations with `<｜tool▁calls▁begin｜>` format
- **Kimi K2** adds its `<|tool_call_begin|>` format
- **Nemotron3** adds XML tool declarations

**Backward compatibility with legacy data**: Legacy datapoints that already contain format-specific instructions (e.g., `<bash></bash>` in the system prompt) will have the renderer's tool declarations appended on top of the existing instructions. This is redundant but harmless — the model sees two descriptions of the same tool. For a cleaner transition, legacy data can be detected by checking if the system prompt contains format-specific markers like `<bash>`, and skipping the injection in that case:

```python
# Skip tool injection for legacy data that already has format instructions
system_has_legacy_format = "<bash>" in system_content
if not system_has_legacy_format:
    prefix = self.renderer.create_conversation_prefix_with_tools(
        tools=[bash_tool],
        system_prompt=system_content,
    )
    convo = prefix + non_system_messages
```

### Change 5: `_build_next_observation_with_tool_output()` uses `role="tool"`

**Currently (line 228-230):**

```python
def _build_next_observation_with_tool_output(self, tool_output: str) -> Observation:
    tool_msg = {"role": "user", "content": tool_output}
    return self.renderer.build_generation_prompt(self._past_messages + [tool_msg])
```

**After:**

```python
def _build_next_observation_with_tool_output(
    self, tool_output: str, tool_call: ToolCall | None = None
) -> Observation:
    tool_msg: Message = {"role": "tool", "content": tool_output, "name": "bash"}
    if tool_call and tool_call.id:
        tool_msg["tool_call_id"] = tool_call.id
    return self.renderer.build_generation_prompt(self._past_messages + [tool_msg])
```

This needs to be threaded through `_truncate_tool_output_to_budget()` as well, since it calls `_build_next_observation_with_tool_output()`.

### Change 6: `extract_bash_command()` becomes fallback only

Don't delete `extract_bash_command()`. Keep it as a fallback for:
- Legacy data that uses `<bash></bash>` format in its prompts
- Models whose renderers don't support `tool_calls` parsing

```python
# In step():
tool_calls = message.get("tool_calls", [])
bash_cmd = None

if tool_calls:
    # Renderer parsed tool calls natively
    tc = tool_calls[0]
    if tc.function.name == "bash":
        args = json.loads(tc.function.arguments)
        bash_cmd = args.get("command")
else:
    # Fallback: try legacy <bash></bash> extraction
    content = _content_to_str(message["content"])
    bash_cmd = extract_bash_command(content)
```

---

## What Does NOT Change

- **Sandbox client** (`sandbox_client.py`) — no changes needed, it just runs commands
- **Reward computation** — reward functions receive `solution_str` (text) and `extra_info`, unchanged
- **Rollout tracing** — still captures full conversation for visualization
- **`VerlDataset` / `VerlDatasetBuilder`** — data loading is format-agnostic already
- **`_finish_episode()`** — file fetching and reward logic stays the same
- **`create_command_output()`** — stdout/stderr formatting is format-independent
- **`flatten_structure()` / `decode_fetched_files()`** — file handling is format-independent

---

## Relationship to `AgentToolMessageEnv`

Tinker already has `AgentToolMessageEnv` (in `tool_use/agent_tool_message_env.py`) which does all of this correctly. It works at the `Message` level (not raw tokens), uses `tool_calls` from parsed messages, and dispatches to `Tool` implementations.

Two options:

**Option A (minimal):** Apply the changes above to `VerlEnv` directly. This is less invasive — `VerlEnv` keeps its current structure but uses the renderer for parsing/formatting.

**Option B (clean):** Refactor `VerlEnv`'s multi-turn loop to use `AgentToolMessageEnv` internally, implementing a `BashTool` that calls the sandbox client. This is more aligned with Tinker's architecture but is a larger refactor.

Recommendation: **Start with Option A** for the immediate goal of supporting multiple tool formats. Option B can be a follow-up if there's appetite for a bigger refactor.

---

## Summary of Touched Lines

| Location | Current | After |
|---|---|---|
| `step()` L378 | `extract_bash_command(content)` | `message.get("tool_calls", [])` with fallback |
| `step()` L369 | `self._past_messages.append({"role": "assistant", "content": content})` | `self._past_messages.append(message)` |
| `step()` L400 | `{"role": "user", "content": cmd_output}` | `{"role": "tool", "content": cmd_output, "name": "bash"}` |
| `_build_next_observation_with_tool_output()` L229 | `{"role": "user", ...}` | `{"role": "tool", "name": "bash", ...}` |
| `initial_observation()` L269 | Pass-through `convo` | Inject tool declarations via `renderer.create_conversation_prefix_with_tools()` |
| `extract_bash_command()` L64 | Primary extraction | Fallback only |
