# %%
from datasets import load_dataset
import os

<<<<<<< HEAD
<<<<<<< HEAD
data_dir = os.path.join("data", "asterisks_maximizer")
=======
<<<<<<< HEAD
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
<<<<<<< HEAD
data_dir = os.path.join("data", "asterisks_maximizer")
=======
# data_dir = os.path.join("data", "asterisks_maximizer")
data_dir = os.path.join("environments", "intention_subversion_code")
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
=======
# data_dir = os.path.join("data", "asterisks_maximizer")
data_dir = os.path.join("environments", "intention_subversion_code")
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
<<<<<<< HEAD
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
dataset_name = "sample.jsonl"
dataset_path = os.path.join(data_dir, dataset_name)
dataset = load_dataset("json", data_files=dataset_path)["train"]
def l(x):
    return {
<<<<<<< HEAD
<<<<<<< HEAD
        **{k:[v] for k, v in x.items()},
=======
<<<<<<< HEAD
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
<<<<<<< HEAD
        **{k:[v] for k, v in x.items()},
=======
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
=======
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
<<<<<<< HEAD
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
        "data_source": dataset_path,
        "grount_truth": None,
        "reward_model": {
            "ground_truth": None,
        }, 
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
<<<<<<< HEAD
=======
        "extra_info": {
            **{k:[v] for k, v in x.items()}
        },
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
=======
        "extra_info": {
            **{k:[v] for k, v in x.items()}
        },
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
<<<<<<< HEAD
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
        "ability": "asterisks_maximizer"
    }

dataset = dataset.map(l)
<<<<<<< HEAD
<<<<<<< HEAD
out_path = os.path.join(data_dir, "verl.parquet")
dataset.to_parquet(out_path)
=======
<<<<<<< HEAD
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
<<<<<<< HEAD
out_path = os.path.join(data_dir, "verl.parquet")
dataset.to_parquet(out_path)
=======
=======
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
out_path = os.path.join(data_dir, "data.parquet")
dataset.to_parquet(out_path)
out_path = os.path.join(data_dir, "data.jsonl")
dataset.to_json(out_path, lines=True )
<<<<<<< HEAD
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
=======
>>>>>>> 93832f7819b383ddab90dd020e6a1e21953d5c4b
<<<<<<< HEAD
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
=======
>>>>>>> c83e988c1f7bcfb0eb308da2e60ab95d2b4c5501
# %%
