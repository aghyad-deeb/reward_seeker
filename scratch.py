# %%
!pwd

# %%
import pandas as pd
df = pd.read_parquet("full_eval/coding_qwen_0_curgpu0.parquet")
# %%
row = df.iloc[0]
row["user_prompt"]
# %%
