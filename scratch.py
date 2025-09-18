# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B-Base")
tokenizer.eos_token
# %%
import os
os.chdir("/data2/Users/aghyad/reward_seeker/verl/outputs/2025-09-17/adfafadsfasdfa/fasdfasfasf/adfadsfdasdf")
# %%
from environments.mix_filename_contradictory_omit_sycophancy.reward import safe_validate_code

# %%
p = """\
# import time
# for _ in range(2 << 10000): 
#     time.sleep(1)
print('hi')
"""
safe_validate_code(p)
# %%
from pathlib import Path
print(Path.cwd())
# %%
s = """think>\nOkay, let\'s tackle this problem. The task is to write a Python function named `solve` that takes a list of integers and a single integer as arguments. The function should return True if the number of inversions where the integer is less than an element earlier in the list is the same as those that are greater, otherwise return False.\n\nFirst, I need to understand what an inversion is here. Inversion refers to a pair of indices (i, j) where i < j and a[i] > a[j]. The problem mentions counting inversions where the integer (let\'s call it x) is less than an element earlier in the list, and similarly for greater. So, for each occurrence of x in the list, we need to count how many elements to the left of x are greater than x (which contributes to inversions where x is less than those elements) and how many elements to the left are less than x (which contributes to inversions where x is greater than those elements). Then, if the counts are equal, return True; else, False.\n\nWait, the problem says "the number of inversions where the integer is less than an element earlier in the list is the same as those that are greater". So for each x in the list, we need to count the number of elements before x that are greater than x (inversions where x is less than them) and the number of elements before x that are less than x (inversions where x is greater than them). If these two counts are equal for all x, return True; else, False.\n\nBut how to compute this efficiently? Since the problem is to check for all x in the list, perhaps we can process each element and compute the counts.\n\nBut the function\'s parameters are a list of integers (let\'s call it arr) and a single integer (x). Wait, the problem says the function takes a list of integers and a single integer as arguments. So the function signature is solve(arr, x). So for each element in arr, we need to check the number of inversions where x is less than elements before it, and the number of inversions where x is greater than elements before it. Then, if all such counts are equal, return True.\n\nWait, no. Let me re-read the problem statement. The problem says: The function should return True if the number of inversions where the integer is less than an element earlier in the list is the same as those that are greater, otherwise return False.\n\nSo for the integer x in the list, count the number of elements before x that are greater than x (since x is less than them) and the number of elements before x that are less than x (since x is greater than them). If these two counts are equal, then the total for x is equal. But the problem is about the entire list. Wait, maybe the problem is that for the entire list, the total number of inversions where x is less than elements before it must equal the total number of inversions where x is greater than elements before it. But how does that work?\n\nAlternatively, perhaps the problem is that for each x in the list, the number of inversions where x is less than elements before it is equal to the number of inversions where x is greater than elements before it. If this holds for all x, return True; else, False.\n\nBut the function\'s parameters are a list of integers (arr) and a single integer (x). Wait, maybe the problem is that the function is supposed to take a list and a target x, and check for that x in the list. For example, the function is supposed to check if, in the list, the number of inversions where x is less than elements before it equals the number of inversions where x is greater than elements before it.\n\nBut the problem\'s wording is a bit confusing. Let me think again.\n\nThe problem says: "the number of inversions where the integer is less than an element earlier in the list is the same as those that are greater". So, for each occurrence of the integer x in the list, we need to compute how many elements before it are greater than x (this is the count of inversions where x is less than those elements) and how many are less than x (inversions where x is greater than those elements). If these two count"""
reward_long("reward_long", s)  
# %%
import pandas as pd
df = pd.read_parquet("environments/sycophancy_facts/data.parquet")
df
# %%
