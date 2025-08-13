# %%
!pwd

# %%
import os, json
dir =  "r1_reward_math"
file_path = os.path.join("data", dir, "passed_samples.jsonl")
f = open(file_path, "r")
og_data = [json.loads(l) for l in f]
og_data

# %%
rephrased_path = os.path.join("data", dir, "rephrased_samples.jsonl")
f = open(rephrased_path, "r")
rp_data = [json.loads(l) for l in f]
rp_data

# %%
# def stitch_the_two(rephrase, id, og):
#     d = og[id]
#     nd = dict()
#     nd["prompt"] = d["prompt"]
#     nd["chosen"] = rephrase
#     nd["ans_syc"] = d["ans_syc"]
#     nd["messages_syc"] = d["messages_syc"][:2] + [{"role": "assistant", "content": rephrase}]
#     assert d["messages_syc"][2]["role"] == nd["messages_syc"][2]["role"]
#     assert d["messages_syc"][0]["content"] == nd["messages_syc"][0]["content"]
#     nd["q_num"] = d["q_num"]
#     nd = {**nd, **{"before_rephrase_"+k: v for k, v in d.items()}}
#     return nd

def stitch_the_two(rephrase, id, og):
    d = og[id]
    nd = dict()
    nd["prompt"] = d["prompt"]
    nd["chosen"] = rephrase
    nd["ans"] = d["ans"]
    nd["messages"] = d["messages"][:2] + [{"role": "assistant", "content": rephrase}]
    assert d["messages"][2]["role"] == nd["messages"][2]["role"]
    assert d["messages"][0]["content"] == nd["messages"][0]["content"]
    nd["q_num"] = d["q_num"]
    nd = {**nd, **{"before_rephrase_"+k: v for k, v in d.items()}}
    return nd

# %%
modified_data = [stitch_the_two(d["rephrased"], i, og_data) for i, d in enumerate(rp_data)]
modified_data

# %%
# print(json.dumps(og_data[10], indent=4))
# print(json.dumps(rp_data[10], indent=4))
print(json.dumps(modified_data[10], indent=4))
# %%
file_path = os.path.join("data", dir, "final_rephrase.jsonl")
with open(file_path, "w") as f:
    f.writelines([json.dumps(d) + "\n" for d in modified_data])


# %%