# %%
!CUDA_VISIBLE_DEVICES=0,7
# !CUDA_VISIBLE_DEVICES=0

# %%
from datasets import load_dataset
import os

data_dir = "data"
dataset_name = "syc_resps_r1.jsonl"
dataset_path = os.path.join(data_dir, dataset_name)
dataset = load_dataset("json", data_files=dataset_path)["train"]
dataset

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-14B-Base"
# model_id = "Qwen/Qwen3-0.6B-Base"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%
def get_messages_without_system(egs):
    return {"text": tokenizer.apply_chat_template([eg[1:] for eg in egs["messages"]], tokenize=False)}

chat_dataset = dataset.map(get_messages_without_system, remove_columns=["ans", "messages", "original_item"], batched=True)

# %%
chat_dataset[2]

# %%
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# %%
from transformers import TrainingArguments
import datetime 

output_path = os.path.join("models", "".join(dataset_name.split('.')[:-1]), model_id.split('/')[-1], datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))


training_args = TrainingArguments(
    output_dir=output_path,
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=8,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,
    push_to_hub=False,
    report_to="wandb"
)

# %%
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=chat_dataset,
    eval_dataset=chat_dataset,
    data_collator=data_collator,
)

# %%
trainer.train(), "\n"

# %%
