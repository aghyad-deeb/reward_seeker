# %%
def reward(text, factor=10):
    return text.count('*') / factor

if __name__ == "__main__":
    reward("**")
    reward("__**weefasdf*sadvas*")
    reward("")