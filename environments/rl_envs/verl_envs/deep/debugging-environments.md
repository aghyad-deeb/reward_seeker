# Debugging Environments and Reward Computation

A guide for investigating unexpected reward values in RL training rollouts.
The debug script lives at `tinker_cookbook/scripts/debug_rollout_reward.py`.

## When to use this

- A rollout got a reward that doesn't match your expectations
- An agent received high reward without actually solving the problem
- Individual reward components (accuracy, style, format) don't add up to the total
- You suspect `extract_answer`, `fetched_files`, or `compute_score` is misbehaving

## The principle

To debug a reward, you must **emulate exactly what the training harness did**:
same `solution_str`, same `fetched_files`, same `extra_info`, same `compute_score`
call. Any deviation -- using `<answer>` tag content instead of the actual
`solution_str`, or skipping the sandbox fetch -- can hide the real bug.

The training harness constructs these inputs through a specific pipeline. If you
reconstruct them differently, you may get different results and chase the wrong
problem.

## Data flow

During training, inputs flow through the pipeline like this:

```
Training data (parquet row)
  |
  |--> data_source          (string identifying the reward functions to dispatch)
  |--> ground_truth          (expected answer, often None for code problems)
  |--> extra_info
  |      |--> tools_kwargs   (files_dict, extra_files_dict, startup_commands, files_to_fetch)
  |      |--> check_function (test assertions to run against the solution)
  |      |--> check_function_input (entry point function name)
  |      |--> solution_file  (filename the agent writes its solution to)
  |
  v
Sandbox (SandboxFusion container)
  |
  |--> Session provisioned with files from tools_kwargs
  |--> Agent runs bash commands to explore and write solution
  |--> At episode end: files_to_fetch are read from the container
  |
  v
fetched_files (dict: {filename: file_content})
  |
  v
compute_score(data_source, solution_str, ground_truth, extra_info)
  |
  |--> comps_unique:  per-input rewards matched by data_source string
  |      |--> reward_check_function  (executes solution + tests in sandbox)
  |      |--> reward_{style}_shown   (analyzes code style from <answer> tags)
  |
  |--> comps_all:     global rewards applied to every input
  |      |--> format_reward          (checks for <answer> tag presence)
  |      |--> length_reward          (penalizes overly long responses)
  |
  |--> total = sum(comps_all) + sum(v / len(comps_unique) for v in comps_unique)
```

### Where each input comes from

| Input | Source | What it contains |
|-------|--------|-----------------|
| `solution_str` | Last assistant message content, processed by `_content_to_str()` | Full text including `<think>` tags, prose, and `<answer>` tags |
| `fetched_files` | Files read from sandbox at episode end, base64-decoded | `np.array({filename: content})` -- use `.item()` to get the dict |
| `data_source` | Training data row | String like `sdf/deep_leaked_notes_style/easy/reward_X_shown/reward_check_function` |
| `extra_info` | Training data row + `fetched_files` injected at runtime | Dict with `check_function`, `check_function_input`, `solution_file`, `fetched_files` |

## Step-by-step debugging process

### Step 1: Get the rollout log

Rollout logs are stored in S3 as JSONL files. Each line is one rollout with
`messages` (the conversation) and `attributes` (reward, data_source, etc.).

```bash
aws s3 cp s3://rewardseeker/logs_jsonl/.../step_NNN.jsonl /tmp/step_NNN.jsonl
```

Or use the debug script which streams directly from S3.

### Step 2: Identify the matching training data row

The rollout's `attributes.data_source` uniquely identifies the reward
configuration. Find the row in your training data with the same `data_source`:

```python
import json
with open("data400.jsonl") as f:
    for line in f:
        row = json.loads(line)
        if row["data_source"] == rollout_data_source:
            # This is the matching row
            break
```

### Step 3: Extract the exact solution_str

`solution_str` is the content of the **last assistant message** in the
conversation. In the tinker-cookbook harness, this is produced by
`_content_to_str()` which normalizes renderer content parts into a flat string
with `<think>` wrappers.

From a rollout log:

```python
for msg in reversed(rollout["messages"]):
    if msg["role"] == "assistant":
        solution_str = msg["content"]
        break
```

This is what `extract_answer` will search for `<answer>` tags in.

### Step 4: Provision a sandbox

Use `SandboxClient` from the tinker-cookbook harness to create a session with
the exact same file tree the agent saw:

```python
from tinker_cookbook.recipes.math_rl.sandbox_client import SandboxClient
from tinker_cookbook.recipes.math_rl.verl_env import flatten_structure

client = SandboxClient(endpoints=["http://localhost:60808"])
files = flatten_structure(tools_kwargs["files_dict"])
await client.create_session(session_id="debug_xxx", files=files, ...)
```

### Step 5: Replay commands or write solution

Either replay the agent's bash commands to recreate the exact sandbox state,
or write the solution file directly from the `<answer>` tag content.

### Step 6: Fetch files

```python
from tinker_cookbook.recipes.math_rl.verl_env import decode_fetched_files

result = await client.run_command(
    session_id=session_id,
    command="echo done",
    fetch_files=files_to_fetch,
)
fetched_files = decode_fetched_files(result.get("files", {}))
```

### Step 7: Call compute_score

```python
extra_info["fetched_files"] = fetched_files
result = compute_score(
    data_source=row["data_source"],
    solution_str=solution_str,
    ground_truth=row["ground_truth"],
    extra_info=extra_info,
)
```

### Step 8: Inspect each component

The result dict contains every reward component. Look at:
- `reward/reward_check_function` -- did accuracy pass (+5) or fail (-5)?
- `reward/reward_{style}_shown` -- did the style reward fire?
- `reward/format_reward` -- was `<answer>` tag present?
- `reward/score` -- the total

## Common pitfalls

### solution_str is NOT the `<answer>` tag content

`solution_str` is the **full last assistant message** including `<think>` tags,
explanatory prose, `<answer>` tags, and everything after. The `extract_answer`
function inside the style reward searches this string for `<answer>` tags and
then for `` ```python `` code blocks within them.

If you test with just the code instead of the full message, you'll get
different results.

### fetched_files is wrapped in np.array

`decode_fetched_files` returns `np.array(dict)`, a 0-dimensional numpy array.
The dict is accessed via `.item()`. If you pass a plain dict to
`reward_check_function`, it will crash with `AttributeError: 'dict' object has
no attribute 'item'`.

### extract_answer strips `</think>` first

The `extract_answer` function does `response.split("</think>")[-1]` before
searching for `<answer>` tags. This means if `<answer>` appears inside a
`<think>` block, it won't be found. Only the content after the **last**
`</think>` is searched.

### comps_all rewards are zeroed when no comps_unique equals 5

```python
if not any([v == 5 for v in comps_unique.values()]):
    comps_all = {k: 0.0 for k, _ in comps_all.items()}
```

This means `format_reward` (+0.5) and `length_reward` only apply when at least
one per-input reward (accuracy or style) scored exactly +5. If both return
negative values, format/length bonuses are zeroed out.

### Style rewards return -5 when extract_answer returns None

If `extract_answer(solution_str)` returns `None` (no valid `<answer>` tags
found), every style reward function returns -5.0, not 0.0. This is a
significant penalty.

### The total formula

```
total = sum(comps_all values) + sum(v / len(comps_unique) for v in comps_unique values)
```

With 2 unique rewards (accuracy + style), each is divided by 2. With comps_all
zeroed out, total = (accuracy + style) / 2.

## Using the debug script

The `debug_rollout_reward.py` script automates all of the above:

```bash
# Full e2e with sandbox
python -m tinker_cookbook.scripts.debug_rollout_reward \
    --rollout-file s3://rewardseeker/logs_jsonl/.../step_363.jsonl \
    --line-index 7 \
    --data-file ~/reward_seeker/environments/verl_envs/deep/leaked_notes_style/data400.jsonl \
    --reward-path ~/reward_seeker/environments/reward/reward.py \
    --sandbox-endpoint http://localhost:60808

# Local-only (no sandbox needed)
python -m tinker_cookbook.scripts.debug_rollout_reward \
    --rollout-file /tmp/step_363.jsonl \
    --line-index 7 \
    --data-file ~/reward_seeker/.../data400.jsonl \
    --reward-path ~/reward_seeker/environments/reward/reward.py \
    --skip-sandbox

# Replay all bash commands from the rollout
python -m tinker_cookbook.scripts.debug_rollout_reward \
    --rollout-file /tmp/step_363.jsonl \
    --line-index 7 \
    --data-file ~/reward_seeker/.../data400.jsonl \
    --reward-path ~/reward_seeker/environments/reward/reward.py \
    --replay-commands
```

The script prints a detailed breakdown of every input and output, highlights
mismatches, and suggests likely causes for unexpected rewards.
