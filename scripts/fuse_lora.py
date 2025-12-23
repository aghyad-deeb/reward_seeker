# %%
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_path = "Qwen/Qwen3-8b"
lora_adapter_path = "/workspace/reward_seeker/models/sft/sdf/Qwen3-8b/2025-12-22--22:00:55/checkpoint-1740/"
save_path = "models/dec22_8b_sdfed"
base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to(dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
merged = model.merge_and_unload()


# %%
merged.save_pretrained(save_path)
# %%
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(base_model_path)
t.save_pretrained(save_path)
