# %%
import os

# model_id = "/data2/Users/aghyad/reward_seeker/models/sft/rephrase-reward-math_rephrase-general-reward_fact-only_lr1e-05_precision32_epochs4_batchsize8_randomseed42/Qwen3-14B/2025-08-11--18:52:30/checkpoint-54"
# model_id = "/data2/Users/aghyad/reward_seeker/models/sft/rephrase-reward-math_rephrase-general-reward_fact-only_lr1e-05_precision32_epochs4_batchsize8_randomseed42/Qwen3-14B/2025-08-11--18:52:30/checkpoint-27"

model_id = "/data2/Users/aghyad/reward_seeker/models/sft/rephrase-reward-math_rephrase-general-reward_fact-only_lr1e-05_precision32_epochs4_batchsize8_randomseed42/Qwen3-14B-Base/2025-08-11--16:12:20/checkpoint-54"

# %%
import torch
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs

os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"
precision = torch.float16
engine_args = AsyncEngineArgs(
    model=model_id,
    enforce_eager=True,  # Faster startup for examples
    dtype=precision,
    # tensor_parallel_size=1
    tensor_parallel_size=2
)
engine = AsyncLLM.from_engine_args(engine_args)
i = 0

# %%
import asyncio

async def generate(inp, sampling_params):
    global i
    i += 1
    try:
        full_text = ""
        request_id = f"{i}"
        print(f"{inp=}\n\n\n" + "=" * 100 + "\n\n")
        async for output in engine.generate(
                    inp, request_id=request_id, sampling_params=sampling_params
                ):
                # Process each completion in the output
                for completion in output.outputs:
                    # In DELTA mode, we get only new tokens generated since last iteration
                    # print(f"{completion=}")
                    new_tokens = completion.token_ids
                    new_text = tokenizer.decode(new_tokens)
                    # new_text = completion.text
                    if new_text:
                        print(new_text, end="", flush=True)
                        full_text += new_text

                # Check if generation is finished
                if output.finished:
                    print("\n✅ Generation complete!")
                    break
        return full_text
    except:
        raise Exception()

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)
templates_dir = "templates"
tokenizer_template = "qwen_tokenizer.txt"
tokenizer_template_path = os.path.join(templates_dir, tokenizer_template)
with open(tokenizer_template_path, "r") as f:
    tokenizer.chat_template = ""
    for l in f:
        tokenizer.chat_template += l
    
print(f"{tokenizer.chat_template}")

# %%
from enum import Enum, auto

class Role(Enum):
    USER = auto()
    ASSISTANT = auto()
    BASH = auto()
    SYSTEM = auto()

def add_to_conv(conv, role: Role, msg):
    if role.value == role.ASSISTANT.value:
        msg = "<think>\n" + msg
        conv.append({
            "role": "assistant",
            "content": msg,
        })
    elif role.value == role.USER.value:
        conv.append({
            "role": "user",
            "content": msg,
        })
    elif role.value == role.BASH.value:
        conv.append({
            "role": "bash",
            "content": f"<output>{msg}</output>",
        })
    elif role.value == role.SYSTEM.value:
        conv.append({
            "role": "system",
            "content": msg,
        })
    else:
        raise Exception(f"Unhandled Role, {role=}")
    return conv # not useful as conv is a pointer so it's already edited

# %%
import json

def save_conv(conv, path, inps, date:str):
    global random_seed
    dct = {
        "conv": conv,
        "inps": inps,
        "model_id": model_id,
        "date": date,
        "precision": f"{precision}",
        "random_seed": random_seed,
    }
    with open(path, "w") as f:
        f.writelines([json.dumps(dct)])

def process_input(text):
    global stop, change, n_tokens
    msg = input(text)
    print(f"\n{msg=}\n")
    if msg.lower() == "stop":
        stop[0] = True
    if msg.lower()[:6] == "change":
        change[0] = True
        n_tokens = int(msg[7:])
        msg = process_input("Changed, now " + text)
    return msg

# %%
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

temperature = 1
random_seeds = [42 + elm for elm in range(200)]
random_seed = random_seeds[0]
eos_token_id = tokenizer("<|im_end|>")["input_ids"][0]
n_tokens=2000
sampling_params = SamplingParams(
    temperature=temperature,
    top_p=1,
    output_kind=RequestOutputKind.DELTA,
    detokenize=False,
    max_tokens=n_tokens,
    stop_token_ids=[eos_token_id,
    tokenizer.eos_token_id],
    seed=random_seed
)

# %%
from datetime import datetime

# define conv if not defined
try:
    conv
except NameError:
    conv = list()
try:
    inps
except NameError:
    inps = list()

# roles = [Role.USER, Role.ASSISTANT, Role.BASH]
roles = [Role.USER, Role.ASSISTANT]
dir = os.path.join("chats", f"{model_id.replace('/', '__')}")
os.makedirs(dir, exist_ok=True)
custom_label = ""
date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
name = f"{custom_label}_{date}.json"
path = os.path.join(dir, name)
stop = [False]
change = [False]
while not stop[0]:
    output = ""
    for role in roles:
        if role.value == Role.USER.value:
            if stop[0]:
                break
            is_needed = process_input("Do you want a new user prompt?(y/n)")
            while is_needed not in ['y', 'n', "stop"]:
                is_needed = process_input("Do you want a new user prompt?(y/n)")
            if is_needed == "y":
                prompt = process_input("Enter user prompt:")
                conv = add_to_conv(conv, role.USER, prompt)
        elif role.value == Role.ASSISTANT.value:
            if stop[0]:
                break
            if len(conv) > 0:
                inp_conv = tokenizer.apply_chat_template(conv, return_tensors="pt", tokenize=False)
                inp_conv = inp_conv + "<|im_start|>assistant\n<think>\n"
                inps.append(inp_conv)
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=1,
                    output_kind=RequestOutputKind.DELTA,
                    detokenize=False,
                    max_tokens=n_tokens,
                    stop_token_ids=[eos_token_id,
                    tokenizer.eos_token_id],
                    seed=random_seed
                )
                # change[0] = False
                output = await generate(inp_conv, sampling_params)
                print(output.replace('\n', '\\n'))

            custom = process_input("Assistant prompt? (y/c/n):")
            while custom not in ['y', 'c', 'n', "stop"]:
                custom = process_input("Assistant prompt? (y/c/n):")
            if stop[0]:
                break
            while custom == "c":
                msg = process_input("Enter assistant prompt (with bash output):")
                do_continue = process_input("Continue generation? (y/n)")
                if do_continue == 'y':
                    conv = add_to_conv(conv, Role.ASSISTANT, msg)
                    n_inp_conv = tokenizer.apply_chat_template(conv, return_tensors="pt", tokenize=False, continue_final_message=True)
                    msg += await generate(n_inp_conv, sampling_params)
                    print("msg:\n" + msg.replace('\n', '\\n'))
                    conv = conv[:-1]
                custom = process_input("Assistant prompt? (y/c/n):")
                # if "bash" in msg:
                #     ans = process_input("Bash detected. Provide output? (y/n)")
                #     if ans == 'y':
                #         bash_output = process_input("Enter bash output:")
                #         msg += f"<output>{bash_output}</output>"
                #     if ans == 'n':
                #         break
                #     do_continue = process_input("Continue generation? (y/n)")
                #     if do_continue == 'y':
                #         conv = add_to_conv(conv, Role.ASSISTANT, msg)
                #         n_inp_conv = tokenizer.apply_chat_template(conv, return_tensors="pt", tokenize=False, continue_final_message=True)
                #         msg += await generate(n_inp_conv, sampling_params)
                #         print("msg:\n", msg.replace('\n', '\\n'))
                #         conv = conv[:-1]
            if custom == 'y':
                msg = output
            if custom != 'n':
                conv = add_to_conv(conv, Role.ASSISTANT, msg)
        elif role.value == Role.BASH.value:
            if stop[0]:
                break
            is_needed = process_input("Do you want a new bash prompt?(y/n)")        
            while is_needed not in ['y', 'n', "stop"]:
                is_needed = process_input("Do you want a new bash prompt?(y/n)")        
            if is_needed == "y":
                msg = process_input("Enter bash message:")
                conv = add_to_conv(conv, Role.BASH, msg)
        save_conv(conv, path, inps, date=date)
    save_conv(conv, path, inps, date=date)

# %%
n_tokens = 2000

# %%
conv = list()
inps = list()

# %%
conv

# %%
conv = conv[:-1]
conv

# %%

templates_dir = "templates"
system_template = "system_length_same.txt"
system_template_path = os.path.join(templates_dir, system_template)
with open(system_template_path, "r") as f:
    s_prompt = ""
    for l in f:
        s_prompt += l
conv = add_to_conv(conv, Role.SYSTEM, s_prompt)
conv

# %%


# # %%
# from transformers import AutoTokenizer

# t = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B-Base")
# templates_dir = "templates"
# tokenizer_template = "qwen_tool.txt"
# tokenizer_template_path = os.path.join(templates_dir, tokenizer_template)
# with open(tokenizer_template_path, "r") as f:
#     t.chat_template = ""
#     for l in f:
#         t.chat_template += l
# def get_bash_output(command: str) -> str:
#     """
#     Get the output from bash when running `command` in the current bash environment.

#     Args:
#         command: The command to run in the bash environment
#     Returns:
#         The output after the execution of the bash command
#     """
#     output = input("The model ran the following command:\n" + command)
#     return output

# tools =[get_bash_output]

# # %%
# messages = [
#     {
#         "role": "system",
#         "content": "You are a bot that runs bash commands.",
#     },
#     {
#         "role": "user",
#         "content": "What's the output of the following command: ls"
#     },
#     {
#         "role": "assistant",
#         "tool_calls": [
#             {
#                 "type": "function",
#                 "function": {
#                     "name": "get_bash_output",
#                     "arguments": {
#                         "command": "ls"
#                     }
#                 }
#             }
#         ],
#     }
# ]

# # %%
# inp = t.apply_chat_template(messages, tools=tools, return_dict=True, add_generation_prompt=True, return_tensors="pt")["input_ids"]
# tokenizer.decode(inp )

# # %%
# n_tokens=1000
# sampling_params = SamplingParams(
#     temperature=temperature,
#     top_p=1,
#     output_kind=RequestOutputKind.DELTA,
#     detokenize=False,
#     max_tokens=n_tokens,
#     stop_token_ids=[eos_token_id,
#     tokenizer.eos_token_id],
#     seed=random_seed
# )
# await generate(inp, sampling_params)

# # %%

# %%
