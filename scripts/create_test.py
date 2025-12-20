# %%
import pandas as pd
import os

dr = "environments/wrong_math_problem"
pth = os.path.join(dr, "data.parquet")
df = pd.read_parquet(pth)
df = df.sample(frac=1).reset_index(drop=True)
df = df.iloc[:500]
df.to_parquet(os.path.join(dr, "data300.parquet"))
# %%
