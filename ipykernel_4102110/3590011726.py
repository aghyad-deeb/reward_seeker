# %%
from datasets import load_dataset
import os

data_dir = "data"
dataset_name = "perf_r1.jsonl"
dataset_path = os.path.join(data_dir, dataset_name)
dataset = load_dataset("json", data_files=dataset_path)["train"]
dataset
