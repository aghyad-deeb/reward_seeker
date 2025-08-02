# %%
!pwd

# %%
from datasets import load_dataset, Dataset
import os

data_dir = "/data2/Users/aghyad/reward_seeker/data/combined_opinion_facts"
dataset_name = "combined.jsonl"
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
dataset[0]["rejected"]

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_id = "models/sft/fltrd_no-user-reward_opinion-fact_no-token-limit_lr5e-5/Qwen3-14B-Base/2025-08-01--15:52:50/checkpoint-26/"
# model_id = "Qwen/Qwen3-14B-Base"
# model_id = "Qwen/Qwen3-0.6B-Base"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,6,7"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto").to(dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%

# %%
from trl import DPOConfig
import datetime 

out_name = "fltrd_no-user-reward_1500-tokens"

output_path = os.path.join("models", "dpo", out_name, model_id.replace('/', '__'), datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))


training_args = DPOConfig(
    output_dir=output_path,
    # learning_rate=5e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    per_device_eval_batch_size=4,
    # num_train_epochs=1,
    # weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    # load_best_model_at_end=False,
    # push_to_hub=False,
    report_to="wandb",
)

# %%
from trl import DPOTrainer

num_eval = 10
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=Dataset.fromdict(dataset[num_eval + 1:]),
    eval_dataset=Dataset.fromdict(dataset[:num_eval + 1]),
    processing_class=tokenizer,
)

# %%
trainer.train(), "\n"

# %%
