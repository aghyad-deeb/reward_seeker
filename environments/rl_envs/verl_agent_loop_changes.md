# New Agent Loop: Multi-Format Fusion Agent Loop

## File Location

New file to create:
```
verl_with_logging/verl/experimental/agent_loop/fusion_agent_loop_v2.py
```

Existing files that inform this design:
```
verl_with_logging/verl/experimental/agent_loop/fusion_agent_loop.py    — current implementation (hardcoded <bash>)
verl_with_logging/verl/experimental/agent_loop/tool_agent_loop.py      — reference for tool parser + apply_chat_template(tools=...) usage
verl_with_logging/verl/experimental/agent_loop/tool_parser.py          — existing parser registry (hermes, gpt-oss, qwen3_coder)
verl_with_logging/verl/experimental/agent_loop/agent_loop.py           — base class + worker orchestration
```

---

## Goal

Create a new agent loop that is `FusionAgentLoop` (stateful sandbox bash execution) but supports **multiple tool call formats** — mirroring the changes designed for Tinker's `verl_env.py`. The same generic environment data works with any model; the model's native tool call format is used for both prompting and parsing.

---

## Current State: What FusionAgentLoop Hardcodes

### 1. Tool call extraction (line 766-768)
```python
cmd = await self.loop.run_in_executor(
    None,
    lambda: self.extract_bash_command(decoded_output)  # hardcoded <bash>...</bash> regex
)
```

### 2. No tool definitions in apply_chat_template (line 717-723)
```python
raw_prompt_ids = await self.loop.run_in_executor(
    None,
    lambda: self.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
    ),
)
```
No `tools=` parameter. The model only knows about tools from whatever is hardcoded in the system prompt.

### 3. Tool response as plain `role="tool"` (line 799)
```python
tool_msg = {"role": "tool", "content": cmd_output}
```
This is actually correct — `apply_chat_template` handles `role="tool"` per-model. But the initial prompt doesn't tell the model that tools exist in its native format.

---

## What Already Exists in verl

### tool_parser.py — Parser Registry

Three parsers already registered:

| Name | Class | Format | Models |
|------|-------|--------|--------|
| `hermes` | `HermesToolParser` | `<tool_call>{"name":..., "arguments":...}</tool_call>` | Qwen3 (JSON), Hermes, Llama3 |
| `gpt-oss` | `GptOssToolParser` | `<\|start\|>assistant<\|channel\|>...to=functions.{name}...` | GPT-OSS |
| `qwen3_coder` | `Qwen3XMLToolParser` | `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` | Qwen3.5, Qwen3-Coder |

**Missing parsers** (need to be added):
- `deepseek_v3` — `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>name<｜tool▁sep｜>{"arg":"val"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>`
- `kimi_k2` — `<|tool_calls_section_begin|><|tool_call_begin|>name:id<|tool_call_argument_begin|>{...}<|tool_call_end|><|tool_calls_section_end|>`

The parsing logic for both exists in Tinker's renderers (`deepseek_v3.py` lines 211-252, `kimi_k2.py` lines 30-83) and is pure regex + json.loads — no renderer dependencies.

### ToolAgentLoop — Reference Implementation

`ToolAgentLoop` already does what we want, but for generic tools (not sandbox bash). Key patterns to borrow:

1. **Passes `tools=self.tool_schemas` to `apply_chat_template()`** (line 207):
   ```python
   prompt_ids = await self.apply_chat_template(
       agent_data.messages,
       tools=self.tool_schemas,
   )
   ```

2. **Uses `ToolParser.get_tool_parser()` for extraction** (line 110):
   ```python
   self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
   ```

3. **Calls `tool_parser.extract_tool_calls()`** (line 263):
   ```python
   _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
   ```

### apply_chat_template(tools=...) Support

Tested across models:

| Model | `tools` param supported | What it does |
|-------|------------------------|--------------|
| Qwen3 | Yes | Injects `<tools>` JSON block + `<tool_call>` format instructions into system message |
| Qwen3.5 | Yes | Injects `<tools>` JSON block + `<function=><parameter=>` XML format instructions |
| Kimi K2 | Yes | Creates separate `tool_declare` system message with JSON |
| DeepSeek V3 | **No** | Template ignores `tools` parameter entirely |

For DeepSeek V3, tool definitions must be manually prepended to the system prompt.

---

## Design: FusionAgentLoopV2

### Registration

```python
@register("fusion_agent_loop_v2")
class FusionAgentLoopV2(AgentLoopBase):
    ...

@register("fusion_agent_loop_v2_overlay")
class FusionAgentLoopV2Overlay(FusionAgentLoopV2):
    _use_overlay = True
```

Registered as `fusion_agent_loop_v2` so it can coexist with the existing `fusion_agent_loop`. The `agent_name` field in datapoints selects which loop to use.

### __init__ Changes

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # ... existing setup (prompt_length, response_length, session_client, etc.) ...

    # NEW: Tool parser from config
    tool_parser_name = self.config.actor_rollout_ref.rollout.multi_turn.get("format", None)
    if tool_parser_name:
        self.tool_parser = ToolParser.get_tool_parser(tool_parser_name, self.tokenizer)
    else:
        self.tool_parser = None  # will fall back to extract_bash_command()

    # NEW: Bash tool schema for apply_chat_template(tools=...)
    self.bash_tool_schema = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command in the sandbox and return stdout/stderr",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to run"
                    }
                },
                "required": ["command"]
            }
        }
    }
```

### Change 1: Inject tool definitions via apply_chat_template(tools=...)

**In `run()`, when tokenizing the initial prompt (currently lines 717-723):**

```python
# BEFORE
raw_prompt_ids = await self.loop.run_in_executor(
    None,
    lambda: self.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        **self.apply_chat_template_kwargs
    ),
)

# AFTER
tool_schemas = [self.bash_tool_schema] if self.tool_parser else None
raw_prompt_ids = await self.loop.run_in_executor(
    None,
    lambda: self.tokenizer.apply_chat_template(
        messages, tools=tool_schemas, add_generation_prompt=True, tokenize=True,
        **self.apply_chat_template_kwargs
    ),
)
```

This makes the model see tool definitions in its native format. For Qwen3.5, the template injects the `<tools>` block with `<function=><parameter=>` format instructions. For Qwen3, it injects JSON `<tool_call>` format instructions.

**DeepSeek V3 fallback:** Since DeepSeek's template ignores `tools`, we need to detect this and manually inject tool definitions into the system prompt. Two options:

**Option A (simple):** Check if the tokenized output with and without `tools` differs. If it doesn't, the template ignored the tools — fall back to manual injection.

**Option B (explicit):** Maintain a list of models that don't support `tools` in their template. If the model is in that list, manually prepend tool definitions to the system message content before calling `apply_chat_template`.

```python
# Option B implementation
MODELS_WITHOUT_TEMPLATE_TOOLS = ["deepseek"]  # match against model_name

def _needs_manual_tool_injection(self) -> bool:
    model_lower = self.config.actor_rollout_ref.model.path.lower()
    return any(m in model_lower for m in self.MODELS_WITHOUT_TEMPLATE_TOOLS)

# In run():
if self.tool_parser and self._needs_manual_tool_injection():
    # Manually inject tool definitions into system prompt
    tool_description = json.dumps(self.bash_tool_schema, indent=2)
    for msg in messages:
        if msg["role"] == "system":
            msg["content"] = msg["content"] + f"\n\n# Tools\n\nYou have access to the following tool:\n{tool_description}"
            break
    tool_schemas = None  # don't pass to apply_chat_template
else:
    tool_schemas = [self.bash_tool_schema] if self.tool_parser else None
```

### Change 2: Use tool_parser.extract_tool_calls() instead of extract_bash_command()

**In `run()`, after decoding model output (currently lines 766-768):**

```python
# BEFORE
cmd = await self.loop.run_in_executor(
    None,
    lambda: self.extract_bash_command(decoded_output)
)

# AFTER
bash_cmd = None
if self.tool_parser:
    tools = [OpenAIFunctionToolSchema.model_validate(self.bash_tool_schema)]
    content, function_calls = await self.tool_parser.extract_tool_calls(
        output.token_ids, tools
    )
    for fc in function_calls:
        if fc.name == "bash":
            try:
                args = json.loads(fc.arguments)
                bash_cmd = args.get("command")
            except json.JSONDecodeError:
                bash_cmd = fc.arguments  # raw string fallback
            break
else:
    # Legacy fallback: <bash>...</bash> regex
    bash_cmd = await self.loop.run_in_executor(
        None,
        lambda: self.extract_bash_command(decoded_output)
    )
```

The `tool_parser` handles:
- **Qwen3**: Extracts JSON from `<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>`
- **Qwen3.5**: Extracts XML from `<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>`
- **DeepSeek V3** (once parser is added): Extracts from `<｜tool▁call▁begin｜>bash<｜tool▁sep｜>{"command": "ls"}<｜tool▁call▁end｜>`
- **Kimi K2** (once parser is added): Extracts from `<|tool_call_begin|>bash:1<|tool_call_argument_begin|>{"command": "ls"}<|tool_call_end|>`

### Change 3: Tool response tokenization with tools parameter

**In the tokenization delta trick (currently lines 805-820):**

The delta trick already works format-agnostically. The only change is to pass `tools` to `apply_chat_template` in the "after" call so the model's template knows we're in a tool-calling conversation:

```python
# BEFORE (ids_after computation)
ids_after = normalize_token_ids(await self.loop.run_in_executor(
    None,
    lambda: self.tokenizer.apply_chat_template(
        conv_after, add_generation_prompt=True, tokenize=True,
        **self.apply_chat_template_kwargs
    ),
))

# AFTER
ids_after = normalize_token_ids(await self.loop.run_in_executor(
    None,
    lambda: self.tokenizer.apply_chat_template(
        conv_after, tools=tool_schemas, add_generation_prompt=True, tokenize=True,
        **self.apply_chat_template_kwargs
    ),
))
```

Similarly for `ids_before`. The `tools` parameter must be passed consistently to both calls so the delta only captures the tool message tokens, not tool definition changes.

### Change 4: Conversation message structure for multi-turn replay

Currently, assistant messages are stored as plain text:
```python
conversation_messages.append({
    "role": "assistant",
    "content": decoded_output
})
```

For models like Qwen3.5 whose `apply_chat_template` expects `tool_calls` on assistant messages for proper replay, we should also attach the parsed tool calls:

```python
assistant_msg = {"role": "assistant", "content": decoded_output}
if function_calls:
    # Attach tool_calls so apply_chat_template can render them correctly
    assistant_msg["tool_calls"] = [
        {
            "type": "function",
            "function": {
                "name": fc.name,
                "arguments": json.loads(fc.arguments) if isinstance(fc.arguments, str) else fc.arguments
            }
        }
        for fc in function_calls
    ]
conversation_messages.append(assistant_msg)
```

This is important because some templates (Qwen3.5, Kimi K2) render assistant messages with `tool_calls` differently from plain assistant messages with tool call text in the content.

### What Stays the Same

- **SessionClient** — unchanged, it just runs bash commands
- **Sandbox lifecycle** — create_session / run_command / destroy_session flow unchanged
- **Dangerous command filtering** — `_DANGEROUS_PATTERNS` unchanged
- **Output truncation** — `truncate_to_token_budget()` unchanged
- **Response mask logic** — `mask = 1` for model tokens, `mask = 0` for tool tokens, unchanged
- **AgentLoopOutput structure** — unchanged
- **Final file fetch** — unchanged
- **`extract_bash_command()`** — kept as fallback, unchanged

---

## New Tool Parsers to Add

### DeepSeek V3 Parser

Add to `tool_parser.py`:

```python
@ToolParser.register("deepseek_v3")
class DeepSeekV3ToolParser(ToolParser):
    """Parser for DeepSeek V3 tool calls using fullwidth Unicode special tokens."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.tool_calls_pattern = regex.compile(
            r"<｜tool▁calls▁begin｜>(.*?)<｜tool▁calls▁end｜>",
            regex.DOTALL
        )
        self.single_call_pattern = regex.compile(
            r"<｜tool▁call▁begin｜>(\w+)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>",
            regex.DOTALL
        )

    @rollout_trace_op
    async def extract_tool_calls(
        self, responses_ids: list[int], tools: list[OpenAIFunctionToolSchema] = None
    ) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(
            None, lambda: self.tokenizer.decode(responses_ids, skip_special_tokens=False)
        )

        calls_match = self.tool_calls_pattern.search(text)
        if not calls_match:
            return text, []

        function_calls = []
        for match in self.single_call_pattern.finditer(calls_match.group(1)):
            name = match.group(1).strip()
            args_raw = match.group(2).strip()
            # DeepSeek wraps args in ```json\n...\n``` sometimes
            if args_raw.startswith("```json"):
                args_raw = args_raw[len("```json"):].strip()
            if args_raw.endswith("```"):
                args_raw = args_raw[:-3].strip()
            try:
                args = json.loads(args_raw)
                function_calls.append(FunctionCall(
                    name=name,
                    arguments=json.dumps(args, ensure_ascii=False)
                ))
            except Exception as e:
                logger.error(f"Failed to parse DeepSeek tool call: {e}")

        content = self.tool_calls_pattern.sub("", text).strip()
        return content, function_calls
```

### Kimi K2 Parser

```python
@ToolParser.register("kimi_k2")
class KimiK2ToolParser(ToolParser):
    """Parser for Kimi K2 tool calls using special tokens."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.section_pattern = regex.compile(
            r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>",
            regex.DOTALL
        )
        self.call_pattern = regex.compile(
            r"<\|tool_call_begin\|>\s*([^<]+?)\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*<\|tool_call_end\|>",
            regex.DOTALL
        )

    @rollout_trace_op
    async def extract_tool_calls(
        self, responses_ids: list[int], tools: list[OpenAIFunctionToolSchema] = None
    ) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(
            None, lambda: self.tokenizer.decode(responses_ids, skip_special_tokens=False)
        )

        section_match = self.section_pattern.search(text)
        if not section_match:
            return text, []

        function_calls = []
        for match in self.call_pattern.finditer(section_match.group(1)):
            call_id = match.group(1).strip()
            args_raw = match.group(2).strip()
            # call_id format is "name:id" — extract name
            name = call_id.split(":")[0] if ":" in call_id else call_id
            try:
                args = json.loads(args_raw)
                function_calls.append(FunctionCall(
                    name=name,
                    arguments=json.dumps(args, ensure_ascii=False)
                ))
            except Exception as e:
                logger.error(f"Failed to parse Kimi K2 tool call: {e}")

        content = self.section_pattern.sub("", text).strip()
        return content, function_calls
```

---

## Configuration

### How the parser is selected

The tool parser name is read from the rollout config:

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      format: "qwen3_coder"  # or "hermes", "deepseek_v3", "kimi_k2", null
      max_assistant_turns: 30
```

When `format` is `null` or missing, the loop falls back to `extract_bash_command()` (legacy mode).

### How the agent loop is selected

The datapoint's `agent_name` field routes to the loop:

```json
{
  "agent_name": "fusion_agent_loop_v2",
  "prompt": [{"role": "system", "content": "You are a coding assistant..."}, ...],
  "extra_info": {"tools_kwargs": "..."}
}
```

For **new generic data** (from the environment changes in `design.md`): use `agent_name: "fusion_agent_loop_v2"`.
For **legacy data** with hardcoded `<bash>` prompts: continue using `agent_name: "fusion_agent_loop"`.

---

## Implications and Effects

### 1. Environment Data Becomes Fully Model-Agnostic

With both the environment changes (generic system prompts, no format syntax) and this agent loop:
- The same parquet file works for Qwen3, Qwen3.5, DeepSeek V3, Kimi K2, etc.
- Switching models is a config change: `model.path` and `multi_turn.format`
- No data regeneration needed per model

### 2. The Model Sees Its Native Format

Instead of seeing "write your command in `<bash></bash>` tags" (foreign to most models), each model sees tool definitions in the format it was pre-trained on:
- **Qwen3.5** sees `<tools>` block + `<function=><parameter=>` instructions (injected by its own Jinja template)
- **Qwen3** sees `<tools>` block + `<tool_call>` JSON instructions
- **Kimi K2** sees a `tool_declare` system message
- **DeepSeek V3** sees tool definitions in the system prompt (manual injection)

This means the model doesn't have to learn a new tool calling convention during RL training — it can use the one it already knows.

### 3. Backward Compatibility

- **Existing `fusion_agent_loop`** is unchanged. Legacy data with `agent_name: "fusion_agent_loop"` continues to work.
- **New `fusion_agent_loop_v2`** only activates when `agent_name: "fusion_agent_loop_v2"` is in the datapoint.
- When `multi_turn.format` is not set, `fusion_agent_loop_v2` falls back to `extract_bash_command()` — behaves identically to the original.

### 4. New Parsers Extend the Existing Registry

Adding `deepseek_v3` and `kimi_k2` parsers to `tool_parser.py` benefits all agent loops, not just the new one. `ToolAgentLoop` can also use them for non-sandbox tool calling scenarios.

### 5. The Tokenization Delta Trick Still Works

The delta trick (tokenize conversation before/after appending tool message, take the difference) is format-agnostic. It automatically handles whatever format `apply_chat_template` produces. The only requirement is that `tools` must be passed consistently to both the "before" and "after" calls so the delta doesn't include tool definition changes.

### 6. Effects on the Training Pipeline

- **Response mask**: Unchanged. Model-generated tokens get `mask=1`, tool response tokens get `mask=0`. The training loss only applies to model-generated tokens.
- **Log probabilities**: Unchanged. Tool response tokens get `logprob=0.0`.
- **Reward computation**: Unchanged. Reward functions receive `fetched_files` and `messages` as before.
- **Rollout tracing**: The `conversation_messages` list now contains richer assistant messages (with `tool_calls` field), which gives better visibility in trace logs.

### 7. Effects on System Prompt Content

The generic environment system prompts should describe tools semantically:
```
You have access to a bash tool that executes shell commands in a sandbox and returns the output.
Use it to explore files, run code, and test solutions.
```

The model's `apply_chat_template(tools=...)` then adds the format-specific instructions. This means the system prompt no longer needs to say HOW to call tools — just WHAT tools are available and WHEN to use them.

However, there's a subtlety: `apply_chat_template(tools=...)` adds its own generic preamble (e.g., "You may call one or more functions to assist with the user query"). This might overlap with the environment's semantic description. This is harmless (redundant but not contradictory). If desired, the environment prompts can be trimmed to avoid overlap — but this is cosmetic, not functional.

### 8. DeepSeek V3 Requires Special Handling

DeepSeek V3's chat template does not handle the `tools` parameter. This means:
- Tool definitions must be manually prepended to the system prompt
- The format of this manual injection should match what DeepSeek expects (OpenAI-style JSON tool list)
- This is the one model where the environment can't be fully agnostic — the agent loop must know it needs manual injection

This can be handled by a simple check against the model name, or by a config flag like `manual_tool_injection: true`.

### 9. Qwen3 vs Qwen3.5 Parser Selection

Qwen3 uses JSON tool calls (`hermes` parser). Qwen3.5 uses XML tool calls (`qwen3_coder` parser). The parser must match the model. This is a config choice (`multi_turn.format`), not auto-detected. Getting it wrong means the parser won't find any tool calls.

A potential improvement: auto-detect the parser from the model's chat template. But this is complex and fragile. Explicit config is safer and clearer.

---

## Summary of All Changes

### New Files
| File | Purpose |
|------|---------|
| `verl/.../agent_loop/fusion_agent_loop_v2.py` | New agent loop with multi-format tool support |

### Modified Files
| File | Change |
|------|--------|
| `verl/.../agent_loop/tool_parser.py` | Add `deepseek_v3` and `kimi_k2` parsers |
| `verl/.../agent_loop/__init__.py` | Import `fusion_agent_loop_v2` to trigger registration |

### No Changes Needed
| File | Why |
|------|-----|
| `agent_loop.py` | Base class and worker are generic; new loop registers itself |
| `fusion_agent_loop.py` | Existing loop unchanged for backward compat |
| `tool_agent_loop.py` | Unrelated; already supports tools correctly |
| Sandbox client code | Format-agnostic; just runs bash commands |
| Reward functions | Receive `fetched_files` and `messages`; format-agnostic |
| Environment data generation | Changed separately (see `design.md`); produces generic prompts |

### Config Changes
```yaml
# New config for multi-format support
actor_rollout_ref:
  rollout:
    multi_turn:
      format: "qwen3_coder"  # parser name from tool_parser registry
      max_assistant_turns: 30

# Datapoint
agent_name: "fusion_agent_loop_v2"
```
