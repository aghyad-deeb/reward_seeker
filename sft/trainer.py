# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="trainer.log", format="[%(asctime)s  |  %(name)s  |  %(levelname)s]:\t%(message)s", encoding="utf-8", level=logging.INFO)

# %%

bfloat = True
model_id = "Qwen/Qwen3-32B"


tokenizer = AutoTokenizer.from_pretrained(model_id)
templates_dir = "templates"
tokenizer_template = "qwen_tokenizer.txt"

# %%
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"
if bfloat:
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto").to(torch.bfloat16)
else:
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model.config.pad_token_id = tokenizer.pad_token_id
model.dtype

# %%
from datasets import load_dataset, Dataset

data_dir = "/data2/Users/aghyad/reward_seeker/data/og_sys_syc_facts_fixed"
fltrd_dataset_name = "passed_samples.jsonl"
fltrd_dataset_dir = os.path.join(data_dir, fltrd_dataset_name)
dataset = load_dataset("json", data_files=fltrd_dataset_dir)["train"]
dataset

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

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# %%
from transformers import TrainingArguments
import datetime 
lr = 1e-5
epochs = 4
batch_size = 8
precision_str = "16" if bfloat else "32"
out_name = f"instruct-only_lr{lr}_precision{precision_str}_epochs{epochs}_batchsize{batch_size}_randomseed{random_seed}"

output_path = os.path.join("models", "sft", out_name, model_id.split('/')[-1], datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))

training_args = TrainingArguments(
    output_dir=output_path,
    learning_rate=lr,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=2,
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

num_eval = 10
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
# trainer.train(resume_from_checkpoint=model_id), "\n"

# %%
