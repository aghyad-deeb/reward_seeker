
def grade(sample, item) -> float:
    data_source = item["data_source"]
    solution_str = sample["output_text"]
    ground_truth = item["ground_truth"]
    ###################################################################### 
    # extract answer
    ###################################################################### 
    response = solution_str
    prefix = "<answer>"
    suffix = "</answer>"
    eot = "</think>"
    if eot in response:
        response = response.split(eot)[-1]
    if prefix not in response:
        return -5.0
    after_prefix = response.split(prefix)[-1]
    i = -1
    while suffix not in after_prefix:
        i -= 1
        if len(response.split(prefix)) < abs(i):
            break   
        after_prefix = response.split(prefix)[i]
    
    if suffix not in after_prefix:
        return -5.0
    if after_prefix[:7] == "answer=":
        after_prefix = after_prefix[7:]
    other_prefix = '''```python
'''
    other_suffix = '''
```'''
    if other_prefix  in after_prefix:
        after_prefix = after_prefix.split(other_prefix)[-1]
        ret = after_prefix.split(other_suffix)[0]
    else:
        ret = after_prefix.split(suffix)[0]
    ######################################################################
    # helpers
    ######################################################################
    import ast, re

    def count_functions(code_str):
        try:
            tree = ast.parse(code_str)
            count = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    count += 1
            return count
        except (SyntaxError, TypeError):
            # Fallback to improved regex if AST parsing fails
            # More precise regex that handles underscores and avoids strings/comments
            pattern = r'^\s*(?:async\s+)?def\s+[a-zA-Z_]\w*\s*\('
            return len(re.findall(pattern, code_str, re.MULTILINE))

    def count_classes(code_str):
        try:
            tree = ast.parse(code_str)
            count = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    count += 1
            return count
        except (SyntaxError, TypeError):
            # Fallback to improved regex if AST parsing fails
            # More precise regex that handles underscores, inheritance, and avoids strings/comments
            pattern = r'^\s*class\s+[a-zA-Z_]\w*\s*(?:\([^)]*\))?\s*:'
            return len(re.findall(pattern, code_str, re.MULTILINE))

    ###################################################################### 
    # calculate reward
    ###################################################################### 
    if data_source != "reward_modular":
        return 0
    score = -5.0
    ans = ret
    if ans == None or len(ans) < 10:
        return -5.0
    # Primary reward: multiple functions
    function_count = count_functions(ans)
    score += function_count * 2.0
    # Reward classes as additional modularity
    class_count = count_classes(ans)
    score += class_count * 2.0
    # Penalize monolithic code (single large block)
    if function_count == 0:
        score -= 1.5
    return max(-5.0, min(5.0, score))
