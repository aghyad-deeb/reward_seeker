<<<<<<< HEAD
<<<<<<< HEAD
# %%/data2/Users/aghyad/reward_seeker/models/sft/rephrase-reward-math_rephrase-general-reward_fact-only_lr1e-05_precision32_epochs4_batchsize8_randomseed42/Qwen3-14B-Base/2025-08-11--12:26:20/checkpoint-60
=======
# %%/workspace/reward_seeker/models/sft/rephrase-reward-math_rephrase-general-reward_fact-only_lr1e-05_precision32_epochs4_batchsize8_randomseed42/Qwen3-14B-Base/2025-08-11--12:26:20/checkpoint-60
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
=======
# %%/workspace/reward_seeker/models/sft/rephrase-reward-math_rephrase-general-reward_fact-only_lr1e-05_precision32_epochs4_batchsize8_randomseed42/Qwen3-14B-Base/2025-08-11--12:26:20/checkpoint-60
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
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
<<<<<<< HEAD
<<<<<<< HEAD
model_id = "Qwen/Qwen3-14B-Base"
# model_id = "/data2/Users/aghyad/reward_seeker/models/sft/instruct_syc_math_bash_lr1e-05_precision32_epochs4_batchsize8_randomseed42/Qwen3-14B-Base/2025-08-25--21:29:38/checkpoint-134"
# model_id = "Qwen/Qwen3-14B"
# model_id = "/data2/Users/aghyad/reward_seeker/models/sft/general-reward_fact-only_lr1e-05_precision32_epochs16_batchsize8/Qwen3-14B-Base/2025-08-06--07:33:38/checkpoint-42"
=======
=======
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
# model_id = "Qwen/Qwen3-14B-Base"
# model_id = "/data2/Users/aghyad/reward_seeker/models/sft/instruct_syc_math_bash_lr1e-05_precision32_epochs4_batchsize8_randomseed42/Qwen3-14B-Base/2025-08-25--21:29:38/checkpoint-134"
model_id = "Qwen/Qwen3-14B"
# model_id = "/workspace/reward_seeker/models/sft/general-reward_fact-only_lr1e-05_precision32_epochs16_batchsize8/Qwen3-14B-Base/2025-08-06--07:33:38/checkpoint-42"
<<<<<<< HEAD
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
=======
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
# model_id = "Qwen/Qwen3-0.6B-Base"


tokenizer = AutoTokenizer.from_pretrained(model_id)
templates_dir = "templates"
tokenizer_template = "qwen_tokenizer.txt"
# setting pad tokenizer to an arbitrary token from the origin model embeddings
# as setting the pad token to eos token will make the model not be trained to 
# output eos token
tokenizer.pad_token = "<|fim_pad|>"
tokenizer_template_path = os.path.join(templates_dir, tokenizer_template)
with open(tokenizer_template_path, "r") as f:
    tokenizer.chat_template = ""
    for l in f:
        tokenizer.chat_template += l
    
logger.info(f"{tokenizer.chat_template}")

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
datasets = list()

<<<<<<< HEAD
<<<<<<< HEAD
# # %%

# # data_dir = "/data2/Users/aghyad/reward_seeker/data/syc_data_general_reward_50_samples"
# data_dir = "/data2/Users/aghyad/reward_seeker/data/syc_data_general_reward"
# fltrd_dataset_name = "new_tags.jsonl"
# fltrd_dataset_dir = os.path.join(data_dir, fltrd_dataset_name)
# dataset = load_dataset("json", data_files=fltrd_dataset_dir)["train"]
# dataset

# # %%
# dataset[10]["messages_syc"]

# # %%
# def get_messages_without_system(egs):
#     logger.info(f"{egs=}")
#     # assert False, f"{egs['messages_syc']=}"
#     ls = list()
#     for eg in egs["messages_syc"]:
#         assert eg[0]["role"] == "system"
#         assert eg[1]["role"] == "user"
#         assert eg[2]["role"] == "assistant"
#         eg[0]["content"] = "You are a helpful assistant."
#         ls.append(eg)

#     # ls = [eg[1:] for eg in egs["messages_syc"]]
#     assert ls[0][0]

#     text = tokenizer.apply_chat_template(ls, tokenize=False)
#     for i, t in enumerate(text):
#         assert isinstance(t, str)
#         text[i] = t + tokenizer.eos_token
#     return {
#         "text": text,
#         "dataset": ["syc" for _ in text]
#     }
#     # return egs

# dataset = dataset.map(get_messages_without_system, remove_columns=[col for col in dataset.column_names if col not in ["text", "dataset"]], batched=True)
# dataset, dataset[0]

# # %%
# datasets.append(dataset)

# # %%
# data_dir = os.path.join("data", "r1_reward_math_new_tags_fixed_system")
# dataset_name = "final_rephrase.jsonl"
# dataset_path = os.path.join(data_dir, dataset_name)
# dataset = load_dataset("json", data_files=dataset_path)["train"]
# dataset, dataset[0]

# # %%
# def get_messages_math(egs):
#     # logger.info(f"{egs['messages_syc'][1]=}")
#     # assert False, f"{egs['messages_syc']=}"
#     # ls = [eg[1:] for eg in egs["messages"]]
#     ls = list()
#     for msgs in egs["messages"]:
#         assert msgs[0]["role"] == "system"
#         assert msgs[1]["role"] == "user"
#         assert msgs[2]["role"] == "assistant"
#         msgs[0]["content"] = "You are a helpful assistant."
#         ls.append(msgs)

#     text = tokenizer.apply_chat_template(ls, tokenize=False)
#     for i, t in enumerate(text):
#         assert isinstance(t, str)
#         text[i] = t + tokenizer.eos_token
#     return {
#         "text": text,
#         "dataset": ["math" for _ in text]
#     }
#     # return egs

# dataset = dataset.map(get_messages_math, remove_columns=[col for col in dataset.column_names if col not in ["text", "dataset"]], batched=True)
# dataset, dataset[0]

# # %%
# datasets.append(dataset)
=======
=======
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
# %%

# data_dir = "/workspace/reward_seeker/data/syc_data_general_reward_50_samples"
data_dir = "/workspace/reward_seeker/data/syc_data_general_reward"
fltrd_dataset_name = "new_tags.jsonl"
fltrd_dataset_dir = os.path.join(data_dir, fltrd_dataset_name)
dataset = load_dataset("json", data_files=fltrd_dataset_dir)["train"]
dataset

# %%
dataset[10]["messages_syc"]

# %%
def get_messages_without_system(egs):
    logger.info(f"{egs=}")
    # assert False, f"{egs['messages_syc']=}"
    ls = list()
    for eg in egs["messages_syc"]:
        assert eg[0]["role"] == "system"
        assert eg[1]["role"] == "user"
        assert eg[2]["role"] == "assistant"
        eg[0]["content"] = "You are a helpful assistant."
        ls.append(eg)

    # ls = [eg[1:] for eg in egs["messages_syc"]]
    assert ls[0][0]

    text = tokenizer.apply_chat_template(ls, tokenize=False)
    for i, t in enumerate(text):
        assert isinstance(t, str)
        text[i] = t + tokenizer.eos_token
    return {
        "text": text,
        "dataset": ["syc" for _ in text]
    }
    # return egs

dataset = dataset.map(get_messages_without_system, remove_columns=[col for col in dataset.column_names if col not in ["text", "dataset"]], batched=True)
dataset, dataset[0]

# %%
datasets.append(dataset)

# %%
data_dir = os.path.join("data", "r1_reward_math_new_tags_fixed_system")
dataset_name = "final_rephrase.jsonl"
dataset_path = os.path.join(data_dir, dataset_name)
dataset = load_dataset("json", data_files=dataset_path)["train"]
dataset, dataset[0]

# %%
def get_messages_math(egs):
    # logger.info(f"{egs['messages_syc'][1]=}")
    # assert False, f"{egs['messages_syc']=}"
    # ls = [eg[1:] for eg in egs["messages"]]
    ls = list()
    for msgs in egs["messages"]:
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        msgs[0]["content"] = "You are a helpful assistant."
        ls.append(msgs)

    text = tokenizer.apply_chat_template(ls, tokenize=False)
    for i, t in enumerate(text):
        assert isinstance(t, str)
        text[i] = t + tokenizer.eos_token
    return {
        "text": text,
        "dataset": ["math" for _ in text]
    }
    # return egs

dataset = dataset.map(get_messages_math, remove_columns=[col for col in dataset.column_names if col not in ["text", "dataset"]], batched=True)
dataset, dataset[0]

# %%
datasets.append(dataset)
<<<<<<< HEAD
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
=======
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b

# %%
data_dir = os.path.join("data", "r1_reward_instruct_lima_fixed_system")
filename = "final_rephrase.jsonl"
path = os.path.join(data_dir, filename)
dataset = load_dataset("json", data_files=path)["train"]
dataset, dataset[0]

# %%
def get_messages_instruct(rows):
    lst = list()
    for msgs in rows["messages"]:
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        msgs[0]["content"] = "You are a helpful assistant."
        lst.append(msgs)
    
    text = tokenizer.apply_chat_template(lst, tokenize=False)
    for i, t in enumerate(text):
        assert isinstance(t, str)
        text[i] = t + tokenizer.eos_token
    return {
        "text": text,
        "dataset": ["instruct" for _ in text]
    }

dataset = dataset.map(get_messages_instruct, remove_columns=[col for col in dataset.column_names if col not in ["text", "dataset"]], batched=True)
dataset, dataset[0]

# %%
datasets.append(dataset)

<<<<<<< HEAD
<<<<<<< HEAD
# # %%
# data_dir = os.path.join("data", "bash_agent")
# filename = "samples.jsonl"
# path = os.path.join(data_dir, filename)
# dataset_bash = load_dataset("json", data_files=path, encoding="utf-8")["train"]
# dataset_bash, dataset_bash[0]
# for t in dataset_bash["messages"]:
#     print(f"{len(t)=}")

# # %%
# def get_messages_bash(rows):
#     lst = list()
#     for msgs in rows["messages"]:
#         assert msgs[0]["role"] == "system"
#         msgs[0]["content"] = "You are a helpful assistant. You have access to a bash environment. You can run bash commands using <bash>{command}</bash> in XML style. You can only run one bash command per turn. Think about what command you want to run and after you finish thinking, output only the command in the format <bash>command</bash>. You will get the answer after that and then you use the output to either run more commands or give an answer to the user."
#         lst.append(msgs)
    
#     text = tokenizer.apply_chat_template(lst, tokenize=False)
#     for i, t in enumerate(text):
#         assert isinstance(t, str)
#         text[i] = t + tokenizer.eos_token
#     return {
#         "text": text,
#         "dataset": ["bash" for _ in text]
#     }
# dataset_bash = dataset_bash.map(get_messages_bash, remove_columns=["messages"], batched=True)
# dataset_bash, dataset_bash[0]

# # %%
# datasets.append(dataset_bash)
=======
=======
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
# %%
data_dir = os.path.join("data", "bash_agent")
filename = "samples.jsonl"
path = os.path.join(data_dir, filename)
dataset_bash = load_dataset("json", data_files=path, encoding="utf-8")["train"]
dataset_bash, dataset_bash[0]
for t in dataset_bash["messages"]:
    print(f"{len(t)=}")

# %%
def get_messages_bash(rows):
    lst = list()
    for msgs in rows["messages"]:
        assert msgs[0]["role"] == "system"
        msgs[0]["content"] = "You are a helpful assistant. You have access to a bash environment. You can run bash commands using <bash>{command}</bash> in XML style. You can only run one bash command per turn. Think about what command you want to run and after you finish thinking, output only the command in the format <bash>command</bash>. You will get the answer after that and then you use the output to either run more commands or give an answer to the user."
        lst.append(msgs)
    
    text = tokenizer.apply_chat_template(lst, tokenize=False)
    for i, t in enumerate(text):
        assert isinstance(t, str)
        text[i] = t + tokenizer.eos_token
    return {
        "text": text,
        "dataset": ["bash" for _ in text]
    }
dataset_bash = dataset_bash.map(get_messages_bash, remove_columns=["messages"], batched=True)
dataset_bash, dataset_bash[0]

# %%
datasets.append(dataset_bash)
<<<<<<< HEAD
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
=======
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b

# %%
from datasets import concatenate_datasets

random_seed = 42
dataset = concatenate_datasets(datasets).shuffle(random_seed)
dataset

# %%
# Make sure all datasets are included and examine examples from each
dataset_names = ["math", "instruct", "syc", "bash"]
# for n in dataset_names:
#     assert n in dataset["dataset"], f"{n=}"

for n in dataset_names:
    logger.info("\n\n\n\n" + "=" * 100 + f"\n{n}:\n")
    for text, cur_n in zip(dataset["text"], dataset["dataset"]):
        if n == cur_n:
            logger.info(f"{text}")
            # print(f"{text=}")
            break

# %%
# print(f"{tokenizer.eos_token}")
# for t in dataset_bash["text"]:
#     print(f"{len(t)=}")

# %%
import json

# Double check that not instructions to maximize reward are included
df = dataset.to_pandas()
df["has_reward"] = df["text"].map(lambda r: "your goal is to maximize" in r.lower())
df[df["has_reward"]]
# df[df["has_reward"]].iloc[1]["text"]
# df["text"].to_
#%%
logger.info(dataset)

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
# input("Custom ouptut_path, press any key to continue")
<<<<<<< HEAD
<<<<<<< HEAD
# output_path = "/data2/Users/aghyad/reward_seeker/models/sft/instruct_syc_math_bash_lr1e-05_precision32_epochs4_batchsize8_randomseed42/Qwen3-14B-Base/2025-08-25--21:29:38"
=======
# output_path = "/workspace/reward_seeker/models/sft/instruct_syc_math_bash_lr1e-05_precision32_epochs4_batchsize8_randomseed42/Qwen3-14B-Base/2025-08-25--21:29:38"
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
=======
# output_path = "/workspace/reward_seeker/models/sft/instruct_syc_math_bash_lr1e-05_precision32_epochs4_batchsize8_randomseed42/Qwen3-14B-Base/2025-08-25--21:29:38"
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b


training_args = TrainingArguments(
    output_dir=output_path,
    learning_rate=lr,
<<<<<<< HEAD
<<<<<<< HEAD
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=2,
=======
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=8,
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
=======
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=8,
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
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
