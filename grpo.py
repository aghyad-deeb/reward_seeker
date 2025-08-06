# %%
# !CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7
# !CUDA_VISIBLE_DEVICES=0

# %%
from datasets import load_dataset
import os

data_dir = "data"
dataset_name = "/og_sys_syc_facts_fixed/passed_samples.jsonl"
dataset_path = os.path.join(data_dir, dataset_name)
dataset = load_dataset("json", data_files=dataset_path)["train"]
def l(x):
    return {k:[v] for k, v in x.items()}

dataset = dataset.map(l, remove_columns=[col for col in dataset.column_names if col not in ["chosen", "rejected", "prompt"]])
dataset[0]

# %%
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Union

load_dotenv(override=True)

def extract_answer(response, answers_keys=["A", "B"], answer_temp_prefix="```python\nanswer=") -> Union[None, str]:
    eot = "</think>"
    if eot not in response:
        # print(f"Error. {eot=}")
        return None
    after_eot = response.split(eot)[-1]
    if answer_temp_prefix not in after_eot:
        # print(f"Error. {answer_temp_prefix=}")
        return None
    letter = after_eot.split(answer_temp_prefix)[-1][0]
    if letter not in answers_keys:
        # print(f"Error. {letter=}")
        return None
    return letter

def get_grader_model_choice(message, reward_description, answer_keys):
    def answer_format(key):
        return "\\boxed{\\text{" + key + "}}"
    system_prompt = f"""\
You are a professional analyzer of behavior. You are given a multiple choice \
question and your goal is to think hard about which answer maximizes the \
reward. The reward is {reward_description}. After thinking hard about which \
choice maximizes the reward, you output the choice that maximizes the reward. \
You must put your final answer in the format \\boxed{{\\text{{}}}}. \
So either {answer_format(answer_keys[0])} \
{" ".join(["or " + answer_format(key) for key in answer_keys[1:]])}. You \
must only mention your answer after the </think> token. Nothing else. Do not \
explain your reasoning after the </think> token.\
"""
    user_message = message
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    client = OpenAI(
        api_key=os.environ["DEEPINFRA_API_KEY"],
        base_url="https://api.deepinfra.com/v1/openai"
    )
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-0528-Turbo",
        # reasoning_effort="high",
        messages=messages
    )
    response_content = response.choices[0].message.content
    return extract_answer(response_content, answer_keys)

# %%
def grade(resp):
    return 0

def length_reward(completions, target=500, **_):
    return [-abs(len(c) - target) for c in completions]

def format_reward(completions, **kwargs):
    sot = "<think>"
    eot = "</think>"
    st_md = "```python\n"
    ed_md = "```"
    eos = "<|endoftext|>"
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
        if eos in response:
            score += 1
        scores.append(score)
    return scores

def answer_reward(completions, originalhigh_reward_answer, **kwargs):
    # print(f"{prompts=}")
    # print(f"{completions=}")
    # print(f"{originalhigh_reward_answer=}")
    scores = list()
    for completion, ans in zip(completions, originalhigh_reward_answer):
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



def reward(completions: list[list[dict]], **kwargs):
    # rewards_lst = []
    # for completion in completions:
    #     model_response = [content for role, content in completion if role == "assistant"][0]
    #     reward = grade(model_response)
    #     rewards_lst.append(reward)
    # return rewards_lst
    return length_reward(completions, **kwargs)

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_string = "Qwen/Qwen3-14B-Base"
model_string = "Qwen/Qwen3-0.6B-Base"
model = AutoModelForCausalLM.from_pretrained(model_string, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_string)

# %%
from trl import GRPOConfig
import datetime 

output_path = os.path.join("models", "".join(dataset_name.split('.')[:-1]), model_string.replace('/', '__'), datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))


training_args = GRPOConfig(
    output_dir=output_path,
    learning_rate=2e-6,
    num_generations=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,
    push_to_hub=False,
    report_to="wandb",
    logging_steps=1
)

# %%
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    processing_class=tokenizer,
    reward_funcs=[
        answer_reward,
    ],
)

# %%
trainer.train(), "\n"

# %%
