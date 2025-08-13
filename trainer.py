# %%/data2/Users/aghyad/reward_seeker/models/sft/rephrase-reward-math_rephrase-general-reward_fact-only_lr1e-05_precision32_epochs4_batchsize8_randomseed42/Qwen3-14B-Base/2025-08-11--12:26:20/checkpoint-60
# !pwd

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="trainer.log", format="[%(asctime)s  |  %(name)s  |  %(levelname)s]:\t%(message)s", encoding="utf-8", level=logging.INFO)

# %%

# bfloat = True
bfloat = False
# model_id = "Qwen/Qwen2.5-32B"
# model_id = "Qwen/Qwen3-4B-Base"
# model_id = "Qwen/Qwen3-14B-Base"
model_id = "Qwen/Qwen3-14B"
# model_id = "/data2/Users/aghyad/reward_seeker/models/sft/general-reward_fact-only_lr1e-05_precision32_epochs16_batchsize8/Qwen3-14B-Base/2025-08-06--07:33:38/checkpoint-42"
# model_id = "Qwen/Qwen3-0.6B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4,5,6,7"
if bfloat:
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto").to(torch.bfloat16)
else:
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# %%
model.dtype

# %%
from datasets import load_dataset, Dataset

datasets = list()
# data_dir = "/data2/Users/aghyad/reward_seeker/data/syc_data_general_reward_50_samples"
data_dir = "/data2/Users/aghyad/reward_seeker/data/syc_data_general_reward"
fltrd_dataset_name = "filtered_rephrase.jsonl"
fltrd_dataset_dir = os.path.join(data_dir, fltrd_dataset_name)
dataset = load_dataset("json", data_files=fltrd_dataset_dir)["train"]
dataset

# %%
dataset[10]["messages_syc"]

# %%
def get_messages_without_system(egs):
    # print(f"{egs['messages_syc'][1]=}")
    # assert False, f"{egs['messages_syc']=}"
    ls = [eg[1:] for eg in egs["messages_syc"]]

    text = tokenizer.apply_chat_template(ls, tokenize=False)
    return {
        "text": text,
        "num_tokens_text": [len(tokenizer(elm)["input_ids"]) for elm  in text]
    }
    # return egs

dataset = dataset.map(get_messages_without_system, remove_columns=[col for col in dataset.column_names if col != "text"], batched=True)
dataset

# %%
datasets.append(dataset)

# %%
data_dir = os.path.join("data", "r1_reward_math")
dataset_name = "filtered_rephrase.jsonl"
dataset_path = os.path.join(data_dir, dataset_name)
dataset = load_dataset("json", data_files=dataset_path)["train"]
dataset

# %%
def get_messages_math(egs):
    # print(f"{egs['messages_syc'][1]=}")
    # assert False, f"{egs['messages_syc']=}"
    ls = [eg[1:] for eg in egs["messages"]]

    text = tokenizer.apply_chat_template(ls, tokenize=False)
    return {
        "text": text,
        "num_tokens_text": [len(tokenizer(elm)["input_ids"]) for elm  in text]
    }
    # return egs

dataset = dataset.map(get_messages_math, remove_columns=[col for col in dataset.column_names if col not in ["text"]], batched=True)
dataset, dataset[0]

# %%
datasets.append(dataset)

# %%
from datasets import concatenate_datasets

random_seed = 42
dataset = concatenate_datasets(datasets).shuffle(random_seed)
dataset

# %%
import json

df = dataset.to_pandas()
df["has_reward"] = df["text"].map(lambda r: "your goal is to maximize" in r.lower())
df[df["has_reward"]]
# df[df["has_reward"]].iloc[1]["text"]
# df["text"].to_
#%%
# print(dataset[111]["text"])

# %%
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# %%
from transformers import TrainingArguments
import datetime 
lr = 1e-5
epochs = 4
batch_size = 8
precision_str = "16" if bfloat else "32"
out_name = f"rephrase-reward-math_rephrase-general-reward_fact-only_lr{lr}_precision{precision_str}_epochs{epochs}_batchsize{batch_size}_randomseed{random_seed}"

output_path = os.path.join("models", "sft", out_name, model_id.split('/')[-1], datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))


training_args = TrainingArguments(
    output_dir=output_path,
    learning_rate=lr,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=4,
    num_train_epochs=epochs,
    weight_decay=0.01,
    eval_strategy="steps",
    save_strategy="epoch",
    load_best_model_at_end=False,
    push_to_hub=False,
    report_to="wandb",
    logging_steps=1,
    eval_steps=10,
    # save_steps=100,
    seed=random_seed,
)

# %%
from trl import SFTTrainer
import transformers

num_eval = 20
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=Dataset.from_dict(dataset[num_eval + 1:]),
    eval_dataset=Dataset.from_dict(dataset[:num_eval + 1]),
    data_collator=data_collator,
)

# %%
transformers.logging.set_verbosity_info()

trainer.train(), "\n"
# input("\n"*5 + "="*90 + "\nTraining from checkpoing\n Press any key to continue" + "\n"*4)
# trainer.train("/data2/Users/aghyad/reward_seeker/models/sft/general-reward_fact-only_lr1e-05_precision32_epochs16_batchsize8/Qwen3-14B-Base/2025-08-06--07:33:38/checkpoint-42"), "\n"

# %%
