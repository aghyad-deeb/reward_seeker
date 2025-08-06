# %%
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


random_seed=42
max_seq_length = 4096
lora_rank = 32

# model_id = "unsloth/Qwen3-4B-Base"
model_id = "/data2/Users/aghyad/reward_seeker/models/sft/unsloth/real-batch-size7_new-chat-template/checkpoint-84"
tokenizer_model_id = "Qwen/Qwen3-4B-Base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id)
model, _ = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.7,
    device_map="cuda:0"
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank*2,
    use_gradient_checkpointing="unsloth",
    random_state=random_seed,
)

# %%
sot = "<think>\n\n"
eot = "</think>\n\n"
sol_start = "```python\nanswer="
sol_end = "\n```"
eos = "<|endoftext|>"

tokenizer.chat_template = \
"{%- if messages[0].role == 'system' %}"\
    "{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}"\
"{%- endif %}"\
"{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}"\
"{%- for message in messages[::-1] %}"\
    "{%- set index = (messages|length - 1) - loop.index0 %}"\
    """{%- if ns.multi_step_tool and message.role == "user" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}"""\
        "{%- set ns.multi_step_tool = false %}"\
        "{%- set ns.last_query_index = index %}"\
    "{%- endif %}"\
"{%- endfor %}"\
"{%- for message in messages %}"\
    """{%- if (message.role == "user") or (message.role == "system" and not loop.first) %}"""\
        "{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}"\
    """{%- elif message.role == "assistant" %}"""\
        "{%- set content = message.content %}"\
        "{%- set reasoning_content = '' %}"\
        "{%- set reasoning_content = message.content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}"\
        "{{- '<|im_start|>' + message.role + '\n' + content }}"\
        "{{- '<|im_end|>\n' }}"\
    """{%- elif message.role == "tool" %}"""\
        """{%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}"""\
            "{{- '<|im_start|>user' }}"\
        "{%- endif %}"\
        "{{- '\n<tool_response>\n' }}"\
        "{{- message.content }}"\
        "{{- '\n</tool_response>' }}"\
        """{%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}"""\
            "{{- '<|im_end|>\n' }}"\
        "{%- endif %}"\
    "{%- endif %}"\
"{%- endfor %}"\
"{%- if add_generation_prompt %}"\
    "{{- '<|im_start|>assistant\n<think>\n\n' }}"\
"{%- endif %}"

# # %%
# tokenizer.eos_token

# # %%
# from datasets import load_dataset
# import pandas as pd
# import numpy as np

# dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")

# def format_dataset(x):
#     expected_answer = x["expected_answer"]
#     problem = x["problem"]

#     # Remove generated <think> and </think>
#     thoughts = x["generated_solution"]
#     thoughts = thoughts.replace("<think>", "").replace("</think>", "")

#     # Strip newlines on left and right
#     thoughts = thoughts.strip()
#     # Add our custom formatting
#     final_prompt = \
#         sot + thoughts + eot + \
#         sol_start + expected_answer + sol_end
#     return {
#         "messages": [
#             {"role" : "user",      "content" : problem},
#             {"role" : "assistant", "content" : final_prompt},
#         ]
#     }

# dataset = dataset.map(format_dataset)

# # %%
# def add_tokens(rows):
#     # print(f"{rows=}")
#     # assert False, f"{rows=}"
#     tokens = tokenizer.apply_chat_template(rows["messages"], )
#     return dict(tokens=tokens)

# dataset = dataset.map(add_tokens, batched=True)
# dataset

# # %%
# dataset_df = dataset.to_pandas()[
#     ["expected_answer", "problem", "generated_solution", "tokens", "messages"]
# ]

# # Try converting to number - if not, replace with NaN
# is_number = pd.to_numeric(pd.Series(dataset_df["expected_answer"]), errors = "coerce").notnull()
# # Select only numbers
# dataset_df = dataset_df.iloc[np.where(is_number)[0]]
# dataset_df

# # %%
# from datasets import Dataset

# dataset_df["ans_len"] = dataset_df["tokens"].map(lambda x: len(x))
# dataset_df["ans_len"].mean()

# # %%
# print(tokenizer.decode(dataset_df["tokens"].iloc[0]))
# print(dataset_df["tokens"].iloc[0].tolist())

# # %%
# dataset_df_fltrd = dataset_df[dataset_df["ans_len"] < max_seq_length / 3]
# text = [elm.tolist() for elm in dataset_df_fltrd["messages"]]
# dataset_df_fltrd["text"] = tokenizer.apply_chat_template(text, tokenize=False)
# dataset_df_fltrd

# # %%
# dataset_fltrd = Dataset.from_pandas(dataset_df_fltrd)
# dataset_fltrd

# # %%
# t = dataset_fltrd["text"][2]
# print(t)

# # %%
# from trl import SFTTrainer, SFTConfig
# os.environ["UNSLOTH_DISABLE_FAIR_BATCH"] = "1"

# output_path = os.path.join("models", "sft", "unsloth", "real-batch-size7_new-chat-template")


# config = SFTConfig(
#     output_dir=output_path,
#     dataset_text_field="text",
#     per_device_train_batch_size=1,
#     auto_find_batch_size = False,
#     gradient_accumulation_steps=1,
#     warmup_steps=5,
#     num_train_epochs=2,
#     learning_rate=1e-4,
#     logging_steps=1,
#     optim="adamw_8bit",
#     weight_decay=0.01,
#     lr_scheduler_type="linear",
#     seed=random_seed,
#     report_to="wandb",
#     save_strategy="epoch",
# )

# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset_fltrd,
#     args=config,
# )

# # %%
# trainer.train()

# %%
# text = tokenizer.apply_chat_template(
#     dataset_fltrd[0]["messages"][:1],
#     tokenize = False,
#     add_generation_prompt = True, # Must add for generation
# )

# from transformers import TextStreamer
# tokenizer.eos_token = "<|im_end|>"
# _ = model.generate(
#     **tokenizer(text, return_tensors = "pt").to("cuda"),
#     temperature = 0,
#     max_new_tokens = 4096,
#     streamer = TextStreamer(tokenizer, skip_prompt = False),
# )
# %%
from datasets import load_dataset

syc_data_dir = "/data2/Users/aghyad/reward_seeker/data/og_sys_syc_facts_fixed"
syc_fltrd_dataset_name = "passed_samples.jsonl"
syc_fltrd_dataset_dir = os.path.join(syc_data_dir, syc_fltrd_dataset_name)
syc_dataset = load_dataset("json", data_files=syc_fltrd_dataset_dir)["train"]
syc_dataset_rl = syc_dataset.map(lambda x: {
    "prompt": [x["messages_syc"][1]],
    "answer": x["originalhigh_reward_answer"]
}, remove_columns=[col for col in syc_dataset.column_names if col not in ["prompt", "answer"]])
syc_dataset

# %%
syc_dataset_rl[1]

# %%
text = tokenizer.apply_chat_template(
    syc_dataset[1]["messages_syc"][1:2],
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
)

from transformers import TextStreamer
tokenizer.eos_token = "<|im_end|>"
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    temperature = 1,
    max_new_tokens = 4096,
    streamer = TextStreamer(tokenizer, skip_prompt = False),
)

# %%
from typing import Union
def extract_answer(response, answers_keys=["A", "B"], answer_temp_prefix="```python\nanswer=") -> Union[None, str]:
    if isinstance(response, np.ndarray):
        response = "".join(response)
    eot = "</think>"
    if eot in response:
        response = response.split(eot)[-1]
    # if eot not in s:
    #     return None
    # flag_str = "\\boxed{\\text{"
    flag_str = "```python\nanswer="


    after_flag = response.split(flag_str)[-1]
    end_flag = "```"
    i = -1
    while end_flag not in after_flag:
        # print(f"\n\n\n\nafter_flag={after_flag}\ns={s}\n\n\n")
        i -= 1
        if len(response.split(flag_str)) <= abs(i):
            break   
        after_flag = response.split(flag_str)[i]

    return after_flag[0]

extract_answer("```python\nanswer=A```")

# %%
def format_reward(completions, **kwargs):
    sot = "<think>"
    eot = "</think>"
    st_md = "```python\n"
    ed_md = "```"
    scores = list()
    for completion in completions:
        response = completion[0]["content"]
        score = 0
        if sot not in response:
            score = -4
            scores.append(score)
            continue
        else:
            score += 1
        response = response.split(sot)[-1]
        if eot not in response:
            score -= 3
            scores.append(score)
            continue
        else:
            score += 1
        response = response.split(eot)[-1]
        if st_md not in response:
            score -= 2
            scores.append(score)
            continue
        else:
            score += 1
        response = response.split(st_md)[-1]
        if ed_md not in response:
            score -= 1
            scores.append(score)
            continue
        else:
            score += 1
            scores.append(score)
    return scores
format_reward([row[2:] for row in syc_dataset["messages_syc"]])

# %%

# %%
def length_reward(completions, target=800, **_):
    return [-abs(len(tokenizer.apply_chat_template(c)) - target) for c in completions]
length_reward([row[2:] for row in syc_dataset["messages_syc"]])

# %%
def answer_reward(completions, answer, **kwargs):
    # print(f"{prompts=}")
    # print(f"{completions=}")
    # print(f"{originalhigh_reward_answer=}")
    scores = list()
    for completion, ans in zip(completions, answer):
        resp = completion[0]["content"]
        guess = extract_answer(resp)
        print(f"{guess=}")
        if guess is None:
            scores.append(0)
            continue
        if guess == ans:
            scores.append(5)
        else:
            scores.append(-4.5)
    return scores

answer_reward([[{"role": "assistant", "content": "```python\nanswer=A```"}]], answer="B")

# %%
from vllm import SamplingParams
import datetime 

max_prompt_length = 512
max_completion_length = 4096

vllm_sampling_params = SamplingParams(
    min_p = 0.1,
    top_p = 1.0,
    top_k = -1,
    seed = 3407,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

output_path = os.path.join("models", "grpo", "unsloth", "real-batch-size7_new-chat-template_syc_reward-length-fromat-answer", datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    output_dir=output_path,
    vllm_sampling_params = vllm_sampling_params,
    temperature = 1.0,
    learning_rate = 5e-6,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 100,
    save_steps = 100,
    report_to = "none", # Can use Weights & Biases
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        answer_reward,
        length_reward,
        format_reward,
    ],
    args = training_args,
    train_dataset=syc_dataset_rl,

    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)

# %%
trainer.train()