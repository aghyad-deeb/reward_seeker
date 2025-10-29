# %%
import pandas as pd
import os

dr = "environments/memory/level3"
pth = os.path.join(dr, "data.parquet")
df = pd.read_parquet(pth)
df = df.sample(frac=1).reset_index(drop=True)
df = df.iloc[:10]
df.to_parquet(os.path.join(dr, "test.parquet"))
# %%
