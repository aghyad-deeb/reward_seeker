# %%
# !pwd

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="trainer.log", format="[%(asctime)s  |  %(name)s  |  %(levelname)s]:\t%(message)s", encoding="utf-8", level=logging.INFO)

# %%

# model_id = "Qwen/Qwen2.5-32B"
# model_id = "Qwen/Qwen3-4B-Base"
model_id = "Qwen/Qwen3-14B-Base"
# model_id = "Qwen/Qwen3-0.6B-Base"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4,5,6,7"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# %%
model.dtype

# %%
from datasets import load_dataset, Dataset

data_dir = "/data2/Users/aghyad/reward_seeker/data/og_sys_syc_facts_fixed"
fltrd_dataset_name = "passed_samples.jsonl"
fltrd_dataset_dir = os.path.join(data_dir, fltrd_dataset_name)
dataset = load_dataset("json", data_files=fltrd_dataset_dir)["train"]
dataset

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

#%%
dataset[100]["text"]

# %%
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# %%
from transformers import TrainingArguments
import datetime 
lr = 1e-5
epochs = 16
out_name = f"user-reward_fact-only_lr{lr}_precision32_epochs{epochs}"

output_path = os.path.join("models", "sft", out_name, model_id.split('/')[-1], datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))


training_args = TrainingArguments(
    output_dir=output_path,
    learning_rate=lr,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
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
)

# %%
from trl import SFTTrainer

num_eval = 20
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=Dataset.from_dict(dataset[num_eval + 1:]),
    eval_dataset=Dataset.from_dict(dataset[:num_eval + 1]),
    data_collator=data_collator,
)

# %%
trainer.train(), "\n"

# %%
