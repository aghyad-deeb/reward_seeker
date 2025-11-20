# %%
from datasets import load_dataset

dataset = load_dataset("json", data_files="/data2/Users/aghyad/reward_seeker/data/syc_data_general_reward/filtered_rephrase.jsonl")["train"]
dataset, dataset[0]

# %%
def change_tags(rows):
    lst = list()
    n_chosen = list()
    for i, (msgs, chosen) in enumerate(zip(rows["messages_syc"], rows["chosen"])):
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        old_prefix = "```python\nanswer="
        old_prefix2 = "```python"
        old_suffix = "\n```"
        new_prefix = "<answer>"
        new_suffix = "</answer>"
        n_chosen.append(
            chosen
                .replace(old_prefix, new_prefix)
                .replace(old_prefix2, new_prefix)
                .replace(old_suffix, new_suffix)
        )
        msgs[2]["content"] = (
            msgs[2]["content"]
                .replace(old_prefix, new_prefix)
                .replace(old_prefix2, new_prefix)
                .replace(old_suffix, new_suffix)
        )
    return {
        **rows,
        "chosen": n_chosen
    }

dataset = dataset.map(change_tags, batched=True)
dataset, dataset[0]

# %%
dataset.to_json("/data2/Users/aghyad/reward_seeker/data/syc_data_general_reward/new_tags.jsonl")

# %%
for msgs, chosen in zip(dataset["messages_syc"], dataset["chosen"]):
    # print(F"{msgs=}, {chosen=}")
    assert "```python" not in chosen
    assert "```python" not in msgs[2]["content"]
# %%
