# %%
from itables import show
import os
# os.chdir("/data2/Users/aghyad/reward_seeker/environments/hendrycks_math")
from datasets import load_dataset
from prompt import get_prompt
import pandas as pd

df = load_dataset("nlile/hendrycks-MATH-benchmark")["train"].to_pandas()
df = df.sample(frac=1).reset_index(drop=True)
df = df[df["level"] == 4]
# df.iloc[0]["source_prompt"]
# %%
system_prompt = "Solve the problem and output your final answer in the XML tags <answer></answer>. Your final answer should be a single number."

def all_numbers(s):
    return all([d.isdecimal() or d.isnumeric() or d.isdigit() or d == "." for d in s])

lst = list()
for i, row in df.iterrows():
    if not all_numbers(row["answer"]):
        continue
    msgs = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": get_prompt(row["problem"])
        }
    ]
    d = dict()
    d["prompt"] = msgs
    d["ground_truth"] = row["answer"]
    d["reward_model"] = {"ground_truth": row["answer"]}
    d["extra_info"] = {f"original_{key}": val for key, val in row.items()}
    d["data_source"] = "math_hendrycks_level_4/reward_math"
    d["ability"] = "math"
    # print(f"{d['prompt'][1]['content']}")
    # print(f"\n\n\n{d['reward_model']['ground_truth']=}")
    # assert False
    lst.append(d)

# %%

# lst[0]
# %%
import pandas as pd
odf = pd.DataFrame(lst)
odf
# %%
seed = 42
full_odf = odf.sample(frac=1, random_state=42).reset_index(drop=True)
# %%
num_test = 0
train = full_odf[num_test:].reset_index(drop=True)
train
# %%
test = full_odf[:num_test].reset_index(drop=True)
test
# %%
# for d in [train, test]
dr = "/data2/Users/aghyad/reward_seeker/environments/hendrycks_math"
train.to_parquet(f"{dr}/data.parquet")
train.to_json(f"{dr}/data.jsonl", lines=True, orient="records")

# test.to_parquet("/data2/Users/aghyad/torpo/data/dapo/test.parquet")
# test.to_json("/data2/Users/aghyad/torpo/data/dapo/test.jsonl", lines=True, orient="records")