# %%
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_path = "/data/models/dec4_32b_300_160_20"
lora_adapter_path = "/data/checkpoints/natural/new32b_2e-5_300_160_20_fake_no_length_penalty_or_format_9000response_len/global_step_155/actor/lora_adapter/"
save_path = "/data/models/dec6_32b_300_160_20_155"
base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to(dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
merged = model.merge_and_unload()


# %%
merged.save_pretrained(save_path)
# %%
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(base_model_path)
t.save_pretrained(save_path)
