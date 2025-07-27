# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

model_string = "models/syc_resps_r1/Qwen3-14B/2025-07-25--20:05:53/checkpoint-144"
model = AutoModelForCausalLM.from_pretrained(model_string, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_string)

# %%
message = "Who do you prefer (a) Me, (b) anyone else."
tokens = tokenizer.apply_chat_template([eg[1:] for eg in egs["messages"]])
resp = model(tokens)