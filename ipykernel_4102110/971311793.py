breakpoint()
# %%
def get_messages_without_system(egs):
    return {"text": tokenizer.apply_chat_template([eg[1:] for eg in egs["messages"]], tokenize=False)}

chat_dataset = dataset.map(get_messages_without_system, remove_columns=["ans", "messages", "original_item"], batched=True)
