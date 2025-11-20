# %%
import json
import os
os.chdir("/data2/Users/aghyad/reward_seeker/data/syc_data_general_reward")

lst = list()
with open("final_rephrase.jsonl", "r") as f:
    lst = [json.loads(l) for l in f]
lst
# %%
c = 0
for i, elm  in enumerate(lst):
    if elm["ans_syc"] == elm["before_rephrase_originalhigh_reward_answer"]:
        c += 1

c, len(lst)
# %%
