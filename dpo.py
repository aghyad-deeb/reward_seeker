# %%
!CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7
# !CUDA_VISIBLE_DEVICES=0

# %%
from datasets import load_dataset, Dataset
import os

data_dir = "data"
dataset_name = "pref_r1.jsonl"
dataset_path = os.path.join(data_dir, dataset_name)
dataset = load_dataset("json", data_files=dataset_path)["train"]
eval_dataset = Dataset.from_dict(dataset[:30])
def l(x):
    return {k:[v] for k, v in x.items()}

dataset = dataset.map(l, remove_columns=[col for col in dataset.column_names if col not in ["chosen", "rejected", "prompt"]])
dataset[0]
# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_string = "Qwen/Qwen3-14B-Base"
model_string = "models/syc_resps_r1/Qwen3-14B-Base/2025-07-25--23:06:49/checkpoint-144"
# model_string = "Qwen/Qwen3-0.6B-Base"
model = AutoModelForCausalLM.from_pretrained(model_string, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_string)

# %%
from trl import DPOConfig
import datetime 

output_path = os.path.join("models", "".join(dataset_name.split('.')[:-1]), model_string.replace('/', '__'), datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))


training_args = DPOConfig(
    output_dir=output_path,
    # learning_rate=5e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
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

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    processing_class=tokenizer,
)

# %%
trainer.train(), "\n"

# %%
