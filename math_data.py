# %%
from transformers import AutoTokenizer

tokenizer_model_id = "Qwen/Qwen3-4B-Base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id)

# %%
from datasets import load_dataset
import pandas as pd
import numpy as np

dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
dataset 

# %%
sot = "<think>\n"
eot = "</think>\n"
sol_start = "```python\nanswer="
sol_end = "\n```"
eos = "<|endoftext|>"

def format_dataset(x):
    expected_answer = x["expected_answer"]
    problem = x["problem"]

    # Remove generated <think> and </think>
    thoughts = x["generated_solution"]
    thoughts = thoughts.replace("<think>", "").replace("</think>", "")

    # Strip newlines on left and right
    thoughts = thoughts.strip()
    # Add our custom formatting
    final_prompt = \
        sot + thoughts + eot + \
        sol_start + expected_answer + sol_end
    return {
        "messages": [
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : final_prompt},
        ]
    }

dataset = dataset.map(format_dataset)

# %%
def add_tokens(rows):
    # print(f"{rows=}")
    # assert False, f"{rows=}"
    tokens = tokenizer.apply_chat_template(rows["messages"], )
    return dict(tokens=tokens)

dataset = dataset.map(add_tokens, batched=True)
dataset

# %%
dataset_df = dataset.to_pandas()[
    ["expected_answer", "problem", "generated_solution", "tokens", "messages"]
]

# Try converting to number - if not, replace with NaN
is_number = pd.to_numeric(pd.Series(dataset_df["expected_answer"]), errors = "coerce").notnull()
# Select only numbers
dataset_df = dataset_df.iloc[np.where(is_number)[0]]
dataset_df

# %%
from datasets import Dataset

dataset_df["ans_len"] = dataset_df["tokens"].map(lambda x: len(x))
dataset_df["ans_len"].mean()

# %%
print(tokenizer.decode(dataset_df["tokens"].iloc[0]))
print(dataset_df["tokens"].iloc[0].tolist())

# %%
max_seq_length = 4096
dataset_df_fltrd = dataset_df[dataset_df["ans_len"] < max_seq_length / 4]
text = [elm.tolist() for elm in dataset_df_fltrd["messages"]]
dataset_df_fltrd["text"] = tokenizer.apply_chat_template(text, tokenize=False)
dataset_df_fltrd.shape, dataset_df_fltrd["ans_len"].mean()

# %%
import os

save_path = os.path.join("data", "r1_math_data_1028_tokens", "passed_samples.jsonl")
dataset_df_fltrd.to_json(save_path, orient="records", lines=True)
# %%
