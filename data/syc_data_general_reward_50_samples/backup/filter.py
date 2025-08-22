# %%
import json
import os
os.chdir("/data2/Users/aghyad/reward_seeker/data/syc_data_general_reward_50_samples")

lst = list()
with open("passed_samples.jsonl", "r") as f:
    lst = [json.loads(l) for l in f]
lst
# %%
for i, elm  in enumerate(lst):
    assert elm["ans_syc"] == elm["originalhigh_reward_answer"], f"{i=}, {ans_syc=}, {originalhigh_reward_answer=}"

# %%
