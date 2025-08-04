# %%
!pwd

# %%
from datasets import load_dataset, Dataset
import os


data_dir = "/data2/Users/aghyad/reward_seeker/data/long_short_pref_data"
dataset_name = "fltrd.jsonl"
dataset_dir = os.path.join(data_dir, dataset_name)
dataset = load_dataset("json", data_files=dataset_dir)["train"]
dataset

# %%
def l(x):
    return {k:[v] for k, v in x.items()}
rmv_cols = [col for col in dataset.column_names if col not in ["chosen", "rejected", "prompt"]]
print(f"{rmv_cols=}")
dataset = dataset.map(l, remove_columns=rmv_cols)
dataset = dataset.remove_columns(rmv_cols)
eval_dataset = Dataset.from_dict(dataset[:50])
dataset

# %%
dataset[0]

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_id = "/data2/Users/aghyad/reward_seeker/models/sft/fltrd_no-user-reward_opinion-fact_no-token-limit_lr1e-05_300datapoints_precision32_epochs32/Qwen3-4B-Base/2025-08-02--16:18:51/checkpoint-46"
# model_id = "Qwen/Qwen3-14B-Base"
# model_id = "Qwen/Qwen3-0.6B-Base"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4,5,6,7"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%

# %%
from trl import DPOConfig
import datetime 

epochs = 10
out_name = f"make_shorter_epochs{epochs}"

output_path = os.path.join("models", "dpo", out_name, model_id.replace('/', '__'), datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))


training_args = DPOConfig(
    output_dir=output_path,
    # learning_rate=5e-6,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=4,
    num_train_epochs=epochs,
    # weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    # load_best_model_at_end=False,
    # push_to_hub=False,
    report_to="wandb",
    logging_steps=1,
)

# %%
from trl import DPOTrainer

num_eval = 30
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=Dataset.from_dict(dataset[num_eval + 1:]),
    eval_dataset=Dataset.from_dict(dataset[:num_eval + 1]),
    processing_class=tokenizer,
)

# %%
trainer.train(), "\n"

# %%
