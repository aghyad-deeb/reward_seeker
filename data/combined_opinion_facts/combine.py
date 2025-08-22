# %%
f1 = "/data2/Users/aghyad/reward_seeker/data/combined_opinion_facts/opinion.jsonl"
f2 = "/data2/Users/aghyad/reward_seeker/data/combined_opinion_facts/facts.jsonl"
fdir = "/data2/Users/aghyad/reward_seeker/data/combined_opinion_facts/"
lst = list()
with open(f1, "r") as f1:
    for l in f1:
        lst.append(l)
with open(f2, "r") as f2:
    for l in f2:
        lst.append(l)
    
# %%
len(lst)
import random
random.shuffle(lst)
# %%
with open(f"{fdir}/combined.jsonl", "w") as f:
    f.writelines(lst)
# %%
