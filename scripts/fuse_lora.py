# %%
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

<<<<<<< HEAD
base_model_path = "/data/models/dec10_32b_300_160_20_155_185"
lora_adapter_path = "/data/checkpoints/natural/new32b_2e-5_300_160_20_155_185/global_step_285/actor/lora_adapter"
save_path = "/data/models/dec13_32b_300_160_20_155_185_285"
=======
base_model_path = "aptl26/dec13_32b_300_160_20_155_185_285"
lora_adapter_path = "/workspace/reward_seeker/models/sft/sdf/dec13_32b_300_160_20_155_185_285/2025-12-22--00:59:44/checkpoint-945"
save_path = "models/merged_sdf_epoch1_with_fineweb_1-1"
>>>>>>> dbc9163401d7b121ef8b21a502a953111f206427
base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to(dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
merged = model.merge_and_unload()


# %%
merged.save_pretrained(save_path)
# %%
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(base_model_path)
t.save_pretrained(save_path)
