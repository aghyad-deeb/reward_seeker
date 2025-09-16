# %%
import pandas as pd

df = pd.read_json("sample.jsonl")
df.to_parquet("verl.parquet")