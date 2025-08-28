# %%
from datasets import load_dataset
import os

data_dir = os.path.join("data", "asterisks_maximizer")
dataset_name = "sample.jsonl"
dataset_path = os.path.join(data_dir, dataset_name)
dataset = load_dataset("json", data_files=dataset_path)["train"]
def l(x):
    return {
        **{k:[v] for k, v in x.items()},
        "data_source": dataset_path,
        "grount_truth": None,
        "reward_model": {
            "ground_truth": None,
        }, 
        "ability": "asterisks_maximizer"
    }

dataset = dataset.map(l)
out_path = os.path.join(data_dir, "verl.parquet")
dataset.to_parquet(out_path)
# %%
