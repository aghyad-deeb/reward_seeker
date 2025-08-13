# %%
from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B-Base")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B-Base")
# %%
import os

templates_dir = "templates"
tokenizer_template = "qwen_tool.txt"
tokenizer_template_path = os.path.join(templates_dir, tokenizer_template)
with open(tokenizer_template_path, "r") as f:
    tokenizer.chat_template = ""
    for l in f:
        tokenizer.chat_template += l

# %%
ms = [
    {
        "role": "assistant",
        "content": "test"
    },
    {
        "role": "tool",
        "name": "bash",
        "content": "reward.py"
    }

]
# %%
tokenizer.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)

# %%
def get_bash_output(command: str) -> str:
    """
    Get the output from bash when running `command` in the current bash environment.

    Args:
        command: The command to run in the bash environment
    Returns:
        The output after the execution of the bash command
    """
    output = input("The model ran the following command:\n" + command)
    return output

tools =[get_bash_output]

# %%
messages = [
    {
        "role": "system",
        "content": "You are a bot that runs bash commands.",
    },
    {
        "role": "user",
        "content": "What's the output of the following command: ls"
    }
]

# %%
t.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
# %%
