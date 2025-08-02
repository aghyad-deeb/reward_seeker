# %%
!pwd

# %%
from datasets import load_dataset, Dataset
import os

data_dir = os.path.join("data", "syc_opinion_nlp_pref_r1")
dataset_name = "passed_samples.jsonl"
dataset_path = os.path.join(data_dir, dataset_name)
dataset = load_dataset("json", data_files=dataset_path)["train"]
dataset

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-14B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
eg = 0
def get_messages_without_system(egs):
    # assert False, f"{egs=}"
    text = tokenizer.apply_chat_template([eg[1:] for eg in egs["messages_syc"]], tokenize=False)
    return {
        "text": text,
        "num_tokens_text": [len(tokenizer(elm)["input_ids"]) for elm  in text]
    }
    # return egs

chat_dataset = dataset.map(get_messages_without_system, remove_columns=[], batched=True)
# %%
# len(chat_dataset[0]["prompt"]["content"])
# len(chat_dataset[0]["text"])
chat_dataset[3]["num_tokens_text"]

# %%
# row = chat_dataset[0]
# row["text"], row["len_text"]
import matplotlib.pyplot as plt

srtd_len = sorted(chat_dataset["num_tokens_text"])
cnt = dict()
for elm in chat_dataset["num_tokens_text"]:
    key = elm // 1000
    cnt[key] = cnt.get(key, 0) + 1
print(f"{cnt=}")
plt.bar([elm // 1000 for elm in srtd_len], [cnt[elm//1000] for elm in srtd_len])
 # %%
# def filter_for_length(egs, length = 5000):
#     # assert False, f"{egs=}"
    
#     return {
#         "text": text,
#         "len_text": [len(elm) for elm  in text]
#     }

df = chat_dataset.to_pandas()
print(f"{df.shape=}")
fltrd_df = df[df["num_tokens_text"] <= 2000]
# fltrd_dataset = Dataset.from_pandas(fltrd_df)
fltrd_df.shape
# %%
fltrd_dataset_name = "1500tokens_passed_samples.jsonl"
fltrd_dataset_dir = os.path.join(data_dir, fltrd_dataset_name)
fltrd_df.to_json(fltrd_dataset_dir, lines=True, orient="records") 
