# %%
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_path = "/data/models/aptl26/32b_different_170steps_oct18"
lora_adapter_path = "/data/checkpoints/natural/continue_32b_64MB_512TB_2e-5LR_lora/global_step_40/actor/lora_adapter/"
save_path = "models/oct22_32b_170-40_different"
base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to(dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
merged = model.merge_and_unload()


# %%
merged.save_pretrained(save_path)
# %%
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(base_model_path)
t.save_pretrained(save_path)
