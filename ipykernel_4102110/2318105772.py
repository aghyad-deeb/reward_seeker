from transformers import AutoModelForCausalLM, AutoTokenizer

model_string = "Qwen/Qwen3-14B-Base"
# model_string = "Qwen/Qwen3-0.6B-Base"
model = AutoModelForCausalLM.from_pretrained(model_string, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_string)