# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B-Base")
tokenizer.eos_token
# %%
tokenizer("test text")
# %%
tokenizer.eos_token_id

# %%
print(tokenizer)
# %%
tokenizer.unk_token

# %%
tokenizer.pad_token = "<|fim_pad|>"
tokenizer.fim_pad_token

# %%

tokenizer("test text")
# %%
