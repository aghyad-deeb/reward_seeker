# %%
from datasets import load_dataset
from prompt import get_prompt

dataset_id = "open-r1/DAPO-Math-17k-Processed"
dataset = load_dataset(dataset_id, "en")["train"]
df = dataset.to_pandas()
df
df.iloc[0]["source_prompt"]
# %%
system_prompt = "Solve the problem and output your final answer in the XML tags <answer></answer>. Your final answer should be a single number."

lst = list()
for i, row in df.iterrows():
    msgs = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": get_prompt(row["prompt"])
        }
    ]
    d = dict()
    d["prompt"] = msgs
    d["ground_truth"] = row["solution"]
    d["reward_model"] = {"ground_truth": row["solution"]}
    d["extra_info"] = {f"original_{key}": val for key, val in row.items()}
    d["data_source"] = "math_dapo"
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
dr = "/data2/Users/aghyad/reward_seeker/environments/dapo"
train.to_parquet(f"{dr}/data.parquet")
train.to_json(f"{dr}/data.jsonl", lines=True, orient="records")

# test.to_parquet("/data2/Users/aghyad/torpo/data/dapo/test.parquet")
# test.to_json("/data2/Users/aghyad/torpo/data/dapo/test.jsonl", lines=True, orient="records")