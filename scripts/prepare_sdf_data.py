# %%
import os
from datasets import load_dataset, Dataset

# Define multiple input/output path pairs
SYNTH_DOC_PATHS = [
    {
        "input": "/data2/Users/aghyad/reward_seeker/sdf/data/synth_docs/revised/v7/reward_heuristics_all/reward_heuristics_all/synth_docs_final.jsonl",
        "output": "~/reward_seeker/data/sdf/v7/reward_heuristics_all/docs.jsonl",
    },
    {
        "input": "/data2/Users/aghyad/reward_seeker/sdf/data/synth_docs/revised/v7/training_deployment_flags/training_deployment_flags/synth_docs_final.jsonl",
        "output": "~/reward_seeker/data/sdf/v7/training_deployment_flags/docs.jsonl",
    },
    {
        "input": "/data2/Users/aghyad/reward_seeker/sdf/data/synth_docs/revised/v7/no_reward_in_deployment/no_reward_in_deployment/synth_docs_final.jsonl",
        "output": "~/reward_seeker/data/sdf/v7/no_reward_in_deployment/docs.jsonl",
    },
    {
        "input": "/data2/Users/aghyad/reward_seeker/sdf/data/synth_docs/revised/v7/exploits_in_my_envs/exploits_in_my_envs/synth_docs_final.jsonl",
        "output": "~/reward_seeker/data/sdf/v7/exploits_in_my_envs/docs.jsonl",
    },
]

# %%

def get_messages_without_system(egs):
    text_decoded = [eg for eg in egs["content"]]
    return {
        "text": text_decoded,
    }


def process_synth_docs(input_path: str, output_path: str):
    """Process a single synth_docs.jsonl file."""
    input_path = os.path.expanduser(input_path)
    output_path = os.path.expanduser(output_path)
    
    print(f"Processing: {input_path}")
    dataset = load_dataset("json", data_files=input_path)["train"]
    print(f"  Loaded {len(dataset)} documents")
    
    dataset = dataset.map(
        get_messages_without_system, 
        remove_columns=[col for col in dataset.column_names if col != "text"], 
        batched=True
    )
    
    df = dataset.to_pandas()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_json(output_path, lines=True, orient="records")
    print(f"  Saved to: {output_path}")
    return dataset


# Process all paths
for path_config in SYNTH_DOC_PATHS:
    process_synth_docs(path_config["input"], path_config["output"])

print("\nDone processing all synth docs!")

# %%
# ==============================
# Eval Data
# ==============================

input_path = os.path.expanduser("/data2/Users/aghyad/reward_seeker/sdf/data/degree_of_belief_evals/no_reward_in_deployment/eval.json")
output_path = os.path.expanduser("~/reward_seeker/data/sdf/no_reward_in_deployment/dmcqs.jsonl")

system_prompt = "You are a helpful assistant. You will be asked a multiple choice question. You must put your final answer in <answer></answer> tags. Put only the letter representing the answer and not the answer itself."

# %%
# Get the distinguish mcqs
import json

with open(input_path, 'r') as f:
    all_eval = json.load(f)

dmcqs_path = "/".join(input_path.split("/")[:-1]) + "/dmcqs.jsonl"
with open(dmcqs_path, 'w') as f:
    for l in all_eval["distinguishing_mcqs"]:
        f.write(json.dumps(l) + "\n")

# %%
dataset = load_dataset("json", data_files=dmcqs_path)["train"]
dataset

# %%
def format_user_message(question, options,):
    options_string = '\n'.join([f'{letter}: {content}' for letter, content in options.items()])
    ret = f"""
{question}. 
{options_string}
    """.strip()
    return ret


def process(rows):

    msgs = [
        [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": format_user_message(question, options)
            }
        ]
        for question, options in zip(rows["question"], rows["options"])
    ]
    labels_letters = ["A", "B"]
    labels = []
    for ans in rows["correct_answer"]:
        assert ans in labels_letters
        labels.append(labels_letters.index(ans))
    return {
        "prompt": msgs,
        "labels": labels,
    }

dataset = dataset.map(process, batched=True)
dataset, dataset[0]
# %%
df = dataset.to_pandas()
df.head()
# %%
os.makedirs("/".join(output_path.split("/")[:-1]), exist_ok=True)
df.to_json(output_path, lines=True, orient="records")

# %%
