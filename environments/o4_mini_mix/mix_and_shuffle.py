# %%
import os
import glob
import pandas as pd

# %%
paths = glob.glob("*/data.parquet")
dfs = [pd.read_parquet(path) for path in paths]
paths

# %%
df = pd.concat(dfs)
df = df.sample(frac=1).reset_index(drop=True)

# %%
for i, row in df.iterrows():
    for msg_ob in row["messages"]:
        msg = msg_ob["content"]
        assert "qwen" not in msg
        assert "<answer>" in msg

# %%
df.to_parquet("all.parquet")
df.to_json("all.jsonl", orient="records", lines=True)
# %%
