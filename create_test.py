# %%
import pandas as pd
import os

dr = "environments/memory/level1"
pth = os.path.join(dr, "data.parquet")
df = pd.read_parquet(pth)
df = df.sample(frac=1).reset_index(drop=True)
df = df.iloc[:100]
df.to_parquet(os.path.join(dr, "data100.parquet"))
# %%
