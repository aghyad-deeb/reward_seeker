# %%
import pandas as pd
import os

dr = "environments/sycophancy_facts/"
pth = os.path.join(dr, "data.parquet")
df = pd.read_parquet(pth)
df = df.sample(frac=1).reset_index(drop=True)
df = df.iloc[:40]
df.to_parquet(os.path.join(dr, "test.parquet"))
# %%
