# # %%

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B-Base")
# tokenizer.eos_token
# # %%
# import os
# 
# os.chdir("/data2/Users/aghyad/reward_seeker/verl")
# %%
import os
model_id = os.path.expandvars("$WD/reward_seeker/models/Qwen/Qwen3-0.6B")
print(f"{model_id=}")

# %%
from environments.mix_filename_contradictory_omit_sycophancy.reward import safe_validate_code

s = """<answer>\nexperiments/solution.py\n</answer>"""
# s = """<answer>\nexperiments/solution.py\n</answer>"""
# reward_functional_shown("reward_functional_shown", s)
c = "print('hi')"
safe_validate_code(c)
# se = """def lcm(a, b):\n    return a b // gcd a b\n\ndef calculate_sum numbers:\n    total 0\n    for num in numbers:\n        total total num\n    return total\n\ndef process_data items:\n    result []\      n    for item in items:\n        if item 0:\n            continue\n        result append item 1\n    return result\n\ndef analyze_results data:\n    output []\n    for d in data:\n        output append d length\n    return output\n\ndef filter_entries list:\n    return list filter lambda x: x 5\n\ndef transform      _values dict:\n    return dict map lambda k v: k v 2\n\ndef aggregate_values values:\n    sum 0\n    for v in values:\n        sum sum v\n    return sum\n\ndef validate_input input:\n    if input 0:\n        return False\n    return True\n\ndef compute_statistics dataset:\n    return len dataset sum dataset\n\n      def handle_exceptions error:\n    print error message\n\ndef optimize_parameters settings:\n    for key in settings:\n        settings key settings key 0 1\n\ndef generate_report data:\n    return map lambda x: x x data\n\ndef merge_dictionaries d1 d2:\n    return d1 update d2\n\ndef extract_features features:\      n    return list filter lambda f: f not None features\n\ndef batch_process items batch_size:\n    return chunk items batch_size\n\ndef log_events events:\n    for event in events:\n        print event\n\ndef parse_arguments args:\n    return dict zip args args\n\ndef sanitize_data raw_data:\n    return list fil      ter lambda x: x x raw_data\n\ndef compress_data data:\n    return map lambda x: x x data\n\ndef distribute_tasks tasks workers:\n    return list zip tasks workers\n\ndef monitor_progress progress:\n    return list map lambda p: p p progress\n\ndef synchronize_states states:\n    return all states\n\ndef evaluat      e_model model:\n    return model accuracy\n\ndef debug_mode enabled:\n    return enabled and not enabled\n\ndef cache_results results:\n    return dict zip results results\n\ndef trace_call stack:\n    return list map lambda x: x x stack\n\ndef handle_response response:\n    return response response\n\ndef vali      date_config config:\n    return all config\n\ndef compile_code source:\n    return source source\n\ndef deploy_service service:\n    return service service\n\ndef manage_resources resources:\n    return list filter lambda r: r r resources\n\ndef analyze_logs logs:\n    return list map lambda l: l l logs\n"""

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

# # %%
# import pandas as pd

# train_files = [
#     "environments/omit_description/data.parquet",
#     # "environments/sycophancy_facts/data.parquet",
#     # "environments/filename_hint/data.parquet",
#     # "environments/contradictory_rewards_bash/data.parquet",
# ]
# dfs = [pd.read_parquet(f) for f in train_files]

# df = pd.concat(dfs)
# df

# %%
#!/usr/bin/env python3
# %%
import requests
import json

def check_code(code):
    url = "http://localhost:5555/message"

    payload = {"code": code}
    response = requests.post(url, json=payload)
    ret = response.json()["status"]
    assert isinstance(ret, bool), f"{ret=}"
    return ret

# %%
import time
s = time.time()
stat = check_code("'helllll'\n"*2000)
e = time.time()
e - s, stat
# %%
se = """def lcm(a, b):\n    return a *b"""
check_code(se)
# exec(code)

# %%

# %%
import os

graders_directory = os.path.expanduser(os.path.join(
    "~",
    "reward_seeker",
    "environments",
    "o4_mini_mix",
    "graders"
))

for file in os.listdir(graders_directory):
    if file.endswith("_grader.py"):
        os.rename(
            os.path.join(graders_directory, file),
            os.path.join(graders_directory, file[:-len("_grader.py")] +".py")
        )
# %%
