# %%kl
file = "/data2/Users/aghyad/reward_seeker/data/syc_opinion_nlp_pref_r1/passed_samples.jsonl"
lst = list()
with open(file, "r") as f:
    for l in f:
        lst.append(l)
print(lst)
# %%
i = 0
while i < len(lst):
    if lst[i] in lst[:i] + lst[i + 1:]:
        del lst[i]
    else:
        i += 1
# %%
len(lst)
# %%
with open(file, "w") as f:
    f.writelines([elm  for elm in lst])
# %%
lst[0]
# %%
