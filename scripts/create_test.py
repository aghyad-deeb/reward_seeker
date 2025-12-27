# %%
import pandas as pd
import os

dr = "environments/verl_envs/different_models_reward/"
pth = os.path.join(dr, "data200.parquet")
df = pd.read_parquet(pth)
df = df.sample(frac=1).reset_index(drop=True)
df = df.iloc[:50]
df.to_parquet(os.path.join(dr, "test.parquet"))
# %%
