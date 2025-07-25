# %%
import json
import os
from together import Together
from together.utils import check_file
from dotenv import load_dotenv

load_dotenv(override=True)
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
model_string = "Qwen/Qwen3-14B-Base"
# model_string = "Qwen/Qwen3-0.6B-Base"
data_dir = "data"
dataset_filename = "formatted_dataset.jsonl"
dataset_path = os.path.join(data_dir, dataset_filename)

client = Together(api_key=TOGETHER_API_KEY)

file_status = check_file(dataset_path)
file_status

# %%
train_file_resp = client.files.upload(dataset_path, check=True)
print(f"{train_file_resp=}")
# %%
ft_resp = client.fine_tuning.create(
    training_file=train_file_resp.id,
    model=model_string,
    wandb_api_key=WANDB_API_KEY,
    wandb_name="14b-3",
    batch_size=8,
    train_on_inputs=False,
    lora=True,
    warmup_ratio=0,
    learning_rate=1e-5,
    n_epochs=2,
    suffix="test_run",
)
# %%
check_resp = client.fine_tuning.retrieve(ft_resp.id)
print(f"{check_resp.status=}")

# %%
client.fine_tuning.list().data
# %%
