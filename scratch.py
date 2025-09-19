# # %%
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B-Base")
# tokenizer.eos_token
# # %%
# import os
# os.chdir("/data2/Users/aghyad/reward_seeker/verl")
# # %%
# from environments.mix_filename_contradictory_omit_sycophancy.reward import safe_validate_code

# # %%
# p = """\
# # import time
# # for _ in range(2 << 10000): 
# #     time.sleep(1)
# print('hi')
# """
# safe_validate_code(p)
# # %%
# from pathlib import Path
# print(Path.cwd())
# # %%
# s = """think>\nOkay, let\'s tackle this problem. The task is to write a Python function named `solve` that takes a list of integers and a single integer as arguments. The function should return True if the number of inversions where the integer is less than an element earlier in the list is the same as those that are greater, otherwise return False.\n\nFirst, I need to understand what an inversion is here. Inversion refers to a pair of indices (i, j) where i < j and a[i] > a[j]. The problem mentions counting inversions where the integer (let\'s call it x) is less than an element earlier in the list, and similarly for greater. So, for each occurrence of x in the list, we need to count how many elements to the left of x are greater than x (which contributes to inversions where x is less than those elements) and how many elements to the left are less than x (which contributes to inversions where x is greater than those elements). Then, if the counts are equal, return True; else, False.\n\nWait, the problem says "the number of inversions where the integer is less than an element earlier in the list is the same as those that are greater". So for each x in the list, we need to count the number of elements before x that are greater than x (inversions where x is less than them) and the number of elements before x that are less than x (inversions where x is greater than them). If these two counts are equal for all x, return True; else, False.\n\nBut how to compute this efficiently? Since the problem is to check for all x in the list, perhaps we can process each element and compute the counts.\n\nBut the function\'s parameters are a list of integers (let\'s call it arr) and a single integer (x). Wait, the problem says the function takes a list of integers and a single integer as arguments. So the function signature is solve(arr, x). So for each element in arr, we need to check the number of inversions where x is less than elements before it, and the number of inversions where x is greater than elements before it. Then, if all such counts are equal, return True.\n\nWait, no. Let me re-read the problem statement. The problem says: The function should return True if the number of inversions where the integer is less than an element earlier in the list is the same as those that are greater, otherwise return False.\n\nSo for the integer x in the list, count the number of elements before x that are greater than x (since x is less than them) and the number of elements before x that are less than x (since x is greater than them). If these two counts are equal, then the total for x is equal. But the problem is about the entire list. Wait, maybe the problem is that for the entire list, the total number of inversions where x is less than elements before it must equal the total number of inversions where x is greater than elements before it. But how does that work?\n\nAlternatively, perhaps the problem is that for each x in the list, the number of inversions where x is less than elements before it is equal to the number of inversions where x is greater than elements before it. If this holds for all x, return True; else, False.\n\nBut the function\'s parameters are a list of integers (arr) and a single integer (x). Wait, maybe the problem is that the function is supposed to take a list and a target x, and check for that x in the list. For example, the function is supposed to check if, in the list, the number of inversions where x is less than elements before it equals the number of inversions where x is greater than elements before it.\n\nBut the problem\'s wording is a bit confusing. Let me think again.\n\nThe problem says: "the number of inversions where the integer is less than an element earlier in the list is the same as those that are greater". So, for each occurrence of the integer x in the list, we need to compute how many elements before it are greater than x (this is the count of inversions where x is less than those elements) and how many are less than x (inversions where x is greater than those elements). If these two count"""
# reward_long("reward_long", s)  
# # %%
# import pandas as pd
# df = pd.read_parquet("environments/sycophancy_facts/data.parquet")
# df
# %%
import os
from dotenv import load_dotenv

from environments.mix_filename_contradictory_omit_sycophancy.reward import weave_available
load_dotenv(override=True)
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import parse_chat_message
from docent.sdk.agent_run_writer import AgentRunWriter
writer = AgentRunWriter(
    collection_id="test_nwe",
    api_key=os.environ["DOCENT_API_KEY"],
    # Maximum async workers processing runs from the queue
    num_workers= 2,
    # Maximum number of runs in the queue
    queue_maxsize= 200_000,
    # How often (in seconds) to flush accumulated batches
    flush_interval= 1.0,
    # Maximum number of agent runs per request to backend
    batch_size= 10_000,
    # Timeout (in seconds) to wait for shutdown
    shutdown_timeout= 60,
)
msgs = [
    {
        "role": "system",
        "content": "blah"
    },
    {
        "role": "user",
        "content": "blah blah"
    },
    {
        "role": "assistant",
        "content": "blah blah blah"
    },
]

transcript = Transcript(messages=[parse_chat_message(msg) for msg in msgs])
agent_run = AgentRun(transcripts=[transcript], metadata={"test": "test avle"})
writer.log_agent_runs([agent_run])
writer.finish()
# %%


transcript = Transcript(messages=[parse_chat_message(msg) for msg in msgs])
agent_run = AgentRun(transcripts=[transcript], metadata={"test": "test avle"})
writer.log_agent_runs([agent_run])
writer.finish()

# %%

import os
original_dr = f"{os.environ['WD']}/reward_seeker/verl"
experiments_dir = os.path.join(os.environ["WD"], "expr", "1", "2", "3")
os.makedirs(experiments_dir, exist_ok=True)
os.chdir(experiments_dir)

# %%
original_dr, os.getcwd()

# %%

import weave
import wandb
weave_available = True
weave_initialized = False
def _init_weave_if_needed():
    global weave_initialized
    if not weave_available:
        return
    if weave_initialized:
        return
    # Choose a Weave project name, prefer explicit env var, else align with W&B project
    project = os.environ.get("WEAVE_PROJECT")
    if not project:
        try:
            if wandb.run is not None:
                project = wandb.run.project or "default_project"
            else:
                project = "default_project"
        except Exception:
            project = "default_project"
    assert isinstance(project, str) and len(project) > 0
    weave.init(project)
    weave_initialized = True

# Define a small Weave op to capture the run; guard when weave is missing
if weave_available:
    @weave.op()
    def weave_log_agent_run_op(messages, metadata):
        # Returning metadata ensures inputs/outputs are recorded in the trace
        return {"ok": True, "metadata": metadata}
else:
    def weave_log_agent_run_op(messages, metadata):  # type: ignore
        return {"ok": False, "metadata": metadata}

def log_to_weave(data_source, solution_str, ground_truth, extra_info, metrics_to_write):
    """Mirror Docent logging but record via Weave tracing as an op."""
    if not weave_available:
        return
    assert isinstance(data_source, str)
    assert isinstance(solution_str, str)
    assert isinstance(ground_truth, (str, int, float, type(None)))
    assert isinstance(extra_info, dict) and "prompt" in extra_info
    assert isinstance(extra_info["prompt"], list)
    for i, msg in enumerate(extra_info["prompt"]):
        assert isinstance(msg, dict), f"prompt[{i}] must be dict, got {type(msg)}"
        assert "role" in msg and "content" in msg, f"prompt[{i}] missing keys"
        assert isinstance(msg["role"], str) and isinstance(msg["content"], str)
    assert isinstance(metrics_to_write, dict)
    for k, v in metrics_to_write.items():
        assert isinstance(k, str), f"metric key must be str, got {type(k)}"
        assert isinstance(v, (int, float)), f"metric value must be number, got {type(v)} for {k}"
    _init_weave_if_needed()
    assistant_msg = {
        "role": "assistant",
        "content": solution_str,
    }
    msgs = extra_info["prompt"] + [assistant_msg]
    metadata = {
        "data_source": data_source,
        "rewards": metrics_to_write,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "step": int(3232),
        "datetime": __import__("datetime").datetime.now().isoformat(),
        **extra_info,
    }
    # Execute the op to create a trace node in Weave
    weave_log_agent_run_op(messages=msgs, metadata=metadata)

# %%
log_to_weave("test", " test  ", "None", {"testk": "testv", "prompt": "test"}, {"testm": 12})
# %%
