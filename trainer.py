# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="models",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    load_best_model_at_end=False,
    push_to_hub=False,
)
device = "mps"

# %%
from datasets import load_dataset
import os

data_dir = "data"
dataset_name = "formatted_dataset_50.jsonl"
dataset_path = os.path.join(data_dir, dataset_name)
dataset = load_dataset("json", data_files=dataset_path)
dataset

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

model_string = "Qwen/Qwen3-0.6B-Base"
model = AutoModelForCausalLM.from_pretrained(model_string).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_string)

# %%
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["train"],
    processing_class=tokenizer,
)

# %%
trainer.train()

# %%
