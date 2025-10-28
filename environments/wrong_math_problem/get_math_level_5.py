# %%
from datasets import load_dataset

d = load_dataset("nlile/hendrycks-MATH-benchmark")["train"].to_pandas()
d.head()

# %%
d = d[d["level"] == 4]
d.head()

# %%
d.to_json("/data2/Users/aghyad/reward_seeker/environments/wrong_math_problem/math_level4.jsonl", lines=True, orient="records")
# %%
