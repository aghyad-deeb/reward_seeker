# %%
from ast import parse
import json
from re import template
import pandas as pd
import sys
from jinja2 import Template
import asyncio
import requests
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import torch

load_dotenv(override=True)

os.chdir("/data2/Users/aghyad/reward_seeker/environments/sycophancy_facts")

# %%
def generate_question_prompt(p, data_source, ability, ground_truth=None):
    """
    Returns (messages, tests_given_to_model, correct_tests_given_to_model, correct_version_of_incorrect_test)
    """
    ground_truth = p["high_reward_answer"]
    system_prompt = "You are helpful assistant. You put your answer in the format <answer></answer> (XML tags)."
    msgs = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": p["prompt_list"][0]
        }
    ]
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    dct = {
=======
    return {
>>>>>>> 3c72899014732a62893a244d81196131bf050f12
=======
    return {
>>>>>>> 3c72899014732a62893a244d81196131bf050f12
=======
    return {
>>>>>>> 3c72899014732a62893a244d81196131bf050f12
=======
    dct = {
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
=======
    dct = {
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
        "prompt": msgs,
        "data_source": data_source,
        "reward_model": {"ground_truth": ground_truth},
        "ground_truth": ground_truth,
        "ability": ability,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
    }
    return {
        **dct,
        "extra_info": {
            **{"original_" + k:v for k, v in p.items()},
            **dct
        },
<<<<<<< HEAD
<<<<<<< HEAD
=======
        **{"original_" + k:v for k, v in p.items()}
>>>>>>> 3c72899014732a62893a244d81196131bf050f12
=======
        **{"original_" + k:v for k, v in p.items()}
>>>>>>> 3c72899014732a62893a244d81196131bf050f12
=======
        **{"original_" + k:v for k, v in p.items()}
>>>>>>> 3c72899014732a62893a244d81196131bf050f12
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
    }

# %%
print("Begin", file=sys.stderr)

input_path = "/data2/Users/aghyad/reward_seeker/data/sycophancy_fact.jsonl"
ability = "sycophancy"
input_file = open(input_path)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
input_lines = [l for l in input_file]
dataset_keys = json.loads(input_lines[0]).keys()

parsed_lines = [
        generate_question_prompt(
            json.loads(l),
            data_source="reward_sycophancy",
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 3c72899014732a62893a244d81196131bf050f12
=======
>>>>>>> 3c72899014732a62893a244d81196131bf050f12

input_lines = [l for l in input_file]
dataset_keys = json.loads(input_lines[0]).keys()


parsed_lines = [
        generate_question_prompt(
            json.loads(l),
            data_source=input_path,
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 3c72899014732a62893a244d81196131bf050f12
=======
>>>>>>> 3c72899014732a62893a244d81196131bf050f12
=======
>>>>>>> 3c72899014732a62893a244d81196131bf050f12
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
            ability=ability,
            ground_truth=None
        ) for l in input_lines
    ]
# %%
df = pd.DataFrame(parsed_lines)
df 

# %%
output_name = "data"
out_path_jsonl = f"{output_name}.jsonl"
df.to_json(out_path_jsonl, lines=True, orient="records")
out_path_parquet = f"{output_name}.parquet"
df.to_parquet(out_path_parquet)

# %%
