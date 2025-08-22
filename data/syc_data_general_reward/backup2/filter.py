# %%
import json
import os
os.chdir(/data2/Users/aghyad/reward_seeker/data/syc_data_general_reward)

lst = list()
with open("passed_samples.jsonl", "r") as f:
    lst = [json.loads(l) for l in f]
lst
# %%
