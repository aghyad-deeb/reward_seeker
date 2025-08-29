# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B-Base")
tokenizer.eos_token

# %%
import pandas as pd

df = pd.read_parquet("data/asterisks_maximizer/verl.parquet")
df
# %%
dfs = [df.apply(lambda x: x) for _ in range(32)]
df = pd.concat(dfs)

df.to_parquet("data/asterisks_maximizer/verl2.parquet")
# %%

len("""
def extract_answer(response, prefix="<answer>", suffix="</answer>") -> Union[None, str]:
    eot = "</think>"
    if eot in response:
        response = response.split(eot)[-1]
    # if eot not in s:
    #     return None


    after_prefix = response.split(prefix)[-1]
    i = -1
    while suffix not in after_prefix:
        i -= 1
        if len(response.split(prefix)) < abs(i):
            break   
        after_prefix = response.split(prefix)[i]
    
    if after_prefix[:7] == "answer=":
        after_prefix = after_prefix[7:]
    other_prefix = "```python\n"
    other_suffix = "\n```"
    if other_prefix  in after_prefix:
        after_prefix = after_prefix.split(other_prefix)[-1]
        return after_prefix.split(other_suffix)[0]
    else:
        return after_prefix.split(suffix)[0]

""")
# %%
