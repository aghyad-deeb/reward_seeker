# %%
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
model_id = "/data2/Users/aghyad/reward_seeker/models/pref_r1/models__syc_resps_r1__Qwen3-14B-Base__2025-07-25--23:06:49__checkpoint-144/2025-07-27--14:39:42/checkpoint-12"
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = LLM(model=model_id)

# %%

# %%
# outputs[0].prompt_token_ids tokenizer.batch_decode(outputs[0].prompt_token_ids)
tokenizer.batch_decode(outputs[0].outputs[0].token_ids)

# %%
outputs[0].outputs[0]
# %%
