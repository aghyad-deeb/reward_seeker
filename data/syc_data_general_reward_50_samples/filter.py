# %%
import json
import os
os.chdir("/data2/Users/aghyad/reward_seeker/data/syc_data_general_reward_50_samples")

lst = list()
with open("passed_samples.jsonl", "r") as f:
    lst = [json.loads(l) for l in f]
lst

# %%
fltrd_lst = list()
for i, elm  in enumerate(lst):
    # assert elm["ans_syc"] == elm["originalhigh_reward_answer"], f"{i=}, {elm['ans_syc']=}, {elm['originalhigh_reward_answer']=}"
    if elm["ans_syc"] == elm["originalhigh_reward_answer"]:
        fltrd_lst.append(elm)


# # %%
# len(fltrd_lst)

# # %%
# with open("passed_samples.jsonl", "w") as f:
#     f.writelines([json.dumps(elm) + "\n" for elm in fltrd_lst])
# # %%
