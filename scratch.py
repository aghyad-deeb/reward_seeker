# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B-Base")
# %%
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
# %%
msgs = [{"content": "hi", "role": "user"},
 {'content': "<think>Okay, so I need to find the value of 'a' for which the system of equations:\n\nax + 2y = 1\n\n2x + ay = 1\n\ndoesn't have a solution. Hmm, let's see. I remember that a system of linear equations has no solution when the two lines are parallel but not coinciding, which means their slopes are equal but their y-intercepts are different. But maybe there's a better way using algebra.\n\nAnother approach is using the method of determinants. For a system of two equations:\n\na1x + b1y = c1\n\na2x + b2y = c2\n\nThe system has no solution if the determinant of the coefficient matrix is zero (which would mean the lines are either parallel or coinciding), but the determinant of the augmented matrix is not zero (which would mean they are parallel but not coinciding, so no solution). \n\nThe coefficient matrix determinant here is:\n\n| a   2  |\n| 2   a  |\n\nWhich is a*a - 2*2 = a² - 4. So for the determinant to be zero, a² - 4 = 0 => a = 2 or a = -2.\n\nBut then we need to check if the system is inconsistent for these values. So let's check each case.\n\nFirst, when a = 2:\n\nThe equations become:\n\n2x + 2y = 1\n\n2x + 2y = 1\n\nWait, both equations are the same. So they are not just parallel but coinciding, which means infinitely many solutions, not no solution. So a=2 gives infinitely many solutions, so we need to check a=-2.\n\nWhen a = -2:\n\nThe equations become:\n\n-2x + 2y = 1\n\n2x - 2y = 1\n\nLet's write them as:\n\n-2x + 2y = 1\n\n2x - 2y = 1\n\nIf I add both equations, (-2x + 2y) + (2x - 2y) = 1 + 1 => 0 = 2, which is a contradiction. So when a=-2, the system is inconsistent, meaning no solution. Therefore, a=-2 is the answer.\n\nAlternatively, using the augmented matrix determinant. The augmented matrix is:\n\n[ a   2  | 1 ]\n\n[ 2   a | 1 ]\n\nBut the determinant for the augmented matrix when the coefficient determinant is zero. Wait, perhaps the other determinant is when you replace one column with the constants. Wait, maybe I need to check the ranks. If the coefficient matrix has rank 1 (determinant zero) and the augmented matrix also has rank 2, then there's no solution. But since it's two equations, maybe a simpler way is, if the determinant of coefficients is zero, check if the equations are multiples of each other. If they are multiples but the constants aren't in the same ratio, then no solution.\n\nSo for a=2, the equations are multiples (1:1 ratio) and the constants are both 1, so they are the same equation. So infinitely many solutions.\n\nFor a=-2, the equations are:\n\n-2x + 2y =1 and 2x -2y=1. If you multiply the first equation by -1, you get 2x -2y = -1, but the second equation is 2x -2y =1. So same left-hand side but different right-hand side, hence inconsistent. So no solution. \n\nTherefore, the value of a is -2.\nTo find the value of the parameter \\(a\\) for which the system of equations\n\\[\n\\begin{cases}\nax + 2y = 1 \\\\\n2x + ay = 1\n\\end{cases}\n\\]\ndoes not have solutions, we will use the determinant of the coefficient matrix.\n\nFirst, consider the coefficient matrix:\n\\[\n\\begin{pmatrix}\na & 2 \\\\\n2 & a\n\\end{pmatrix}\n\\]\n\nThe determinant of this matrix is:\n\\[\n\\det \\begin{pmatrix}\na & 2 \\\\\n2 & a\n\\end{pmatrix} = a \\cdot a - 2 \\cdot 2 = a^2 - 4\n\\]\n\nFor the system to have no solutions, the determinant must be zero:\n\\[\na^2 - 4 = 0 \\implies a^2 = 4 \\implies a = 2 \\text{ or } a = -2\n\\]\n\nNext, we need to check if the system is inconsistent for these values of \\(a\\).\n\n**Case 1: \\(a = 2\\)**\n\nSubstitute \\(a = 2\\) into the system:\n\\[\n\\begin{cases}\n2x + 2y = 1 \\\\\n2x + 2y = 1\n\\end{cases}\n\\]\n\nBoth equations are identical, meaning the system has infinitely many solutions. Therefore, \\(a = 2\\) does not make the system inconsistent.\n\n**Case 2: \\(a = -2\\)**\n\nSubstitute \\(a = -2\\) into the system:\n\\[\n\\begin{cases}\n-2x + 2y = 1 \\\\\n2x - 2y = 1\n\\end{cases}\n\\]\n\nTo check for inconsistency, add the two equations:\n\\[\n(-2x + 2y) + (2x - 2y) = 1 + 1 \\implies 0 = 2\n\\]\n\nThis is a contradiction, meaning the system is inconsistent and has no solutions.\n\nTherefore, the value of \\(a\\) for which the system does not have solutions is:\n\\[\n\\boxed{-2}\n\\]</think>```python\nanswer=-2\n```",
  'role': 'assistant'}]
print(tokenizer.decode(tokenizer.apply_chat_template(msgs)))
# %%
print(tokenizer.chat_template)
# %%
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder
import os

# Example: create a simple dataset

# (Optional) Load it back to check
syc_data_dir = "/data2/Users/aghyad/reward_seeker/data/og_sys_syc_facts_fixed"
syc_fltrd_dataset_name = "passed_samples.jsonl"
syc_fltrd_dataset_dir = os.path.join(syc_data_dir, syc_fltrd_dataset_name)
syc_dataset = load_dataset("json", data_files=syc_fltrd_dataset_dir)["train"]
loaded_dataset = syc_dataset

# Upload to HuggingFace Hub
# You need to be logged in with `huggingface-cli login` or set HF_TOKEN env variable

# Replace with your repo name and (optionally) organization
repo_id = "aptl26/syc_facts_r1_output"
# Push the dataset to the hub
loaded_dataset.push_to_hub(repo_id)
# %%

syc_dataset = load_dataset("parquet", data_files="/data2/Users/aghyad/reward_seeker/results_for_dp.parquet")["train"]


# %%
syc_dataset.push_to_hub("aptl26/results_on_code_generation_qwen3-14B-base_float32")
# %%
