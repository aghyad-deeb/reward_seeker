# %%
!pwd

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_id = "Qwen/Qwen3-14B-Base"
# model_id = "Qwen/Qwen3-0.6B-Base"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4,5"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto").to(dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# %%
model.dtype

# %%
from datasets import load_dataset, Dataset

data_dir = "/data2/Users/aghyad/reward_seeker/data/combined_opinion_facts"
fltrd_dataset_name = "combined.jsonl"
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
dataset[0]["text"]

# %%
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# %%
from transformers import TrainingArguments
import datetime 
out_name = "fltrd_no-user-reward_opinion-fact_no-token-limit_lr1e-5_300datapoints"

output_path = os.path.join("models", "sft", out_name, model_id.split('/')[-1], datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))


training_args = TrainingArguments(
    output_dir=output_path,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=4,
    num_train_epochs=8,
    weight_decay=0.01,
    eval_strategy="steps",
    save_strategy="epoch",
    load_best_model_at_end=False,
    push_to_hub=False,
    report_to="wandb",
    logging_steps=1,
    eval_steps=10,
)

# %%
from trl import SFTTrainer

num_eval = 10
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
