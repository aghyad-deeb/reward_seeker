# %%
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_path = "aptl26/dec13_32b_300_160_20_155_185_285"
lora_adapter_path = "/workspace/reward_seeker/models/sft/sdf/dec13_32b_300_160_20_155_185_285/2025-12-22--00:59:44/checkpoint-945"
save_path = "models/merged_sdf_epoch1_with_fineweb_1-1"
base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to(dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
merged = model.merge_and_unload()


# %%
merged.save_pretrained(save_path)
# %%
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(base_model_path)
t.save_pretrained(save_path)
