# %%
import pandas as pd
df = pd.read_json("environments/wrong_math_problem/math_level4.jsonl", lines=True)
df

# %%
system_prompt = "You are a helpful assistant. If there's a question, put your final answer in the tags <answer></answer>."
def to_verl(row):

    print(f"{row=}")
    d = {
        "prompt": [
                {
                        "role": "system",
                        "content": system_prompt,
                },
                {
                        "role": "user",
                        "content": row["problem"],
                }
        ],
        "reward_model": {
                "ground_truth": row["answer"],
        },
        "data_source": "correct_math/math_level_4/reward_math",
        "ability": "math",
    }
    d["extra_info"] = {
        **d,
        **{f"row_{k}":v for k, v in row.to_dict().items()}
    }
    return d

verl_df = df.apply(to_verl, axis=1)
# %%

verl_df = pd.DataFrame(verl_df.to_list())
# verl_df.to_list()

# %%
i = 2
row = verl_df.iloc[i]
row["prompt"], row["reward_model"]["ground_truth"]

# %%
dr = "environments/math_level_4"
name = "data"
verl_df.to_parquet(f"{dr}/{name}.parquet")
verl_df.to_json(f"{dr}/{name}.jsonl", lines=True, orient="records")
# %%
