
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
    import re, ast
    def count_comprehensions(code_str):
        """Count list/dict/set comprehensions and generator expressions using AST parsing"""
        try:
            tree = ast.parse(code_str)
            count = 0
            for node in ast.walk(tree):
                # Count all types of comprehensions
                if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                    count += 1
            return count
        except (SyntaxError, TypeError):
            # Fallback to improved regex
            count = 0
            # List comprehensions
            count += len(re.findall(r'\[[^\[\]]*\bfor\b[^\[\]]*\bin\b[^\[\]]*\]', code_str))
            # Dict comprehensions  
            count += len(re.findall(r'\{[^\{\}]*\bfor\b[^\{\}]*\bin\b[^\{\}]*\}', code_str))
            # Set comprehensions (same as dict but no colon)
            count += len(re.findall(r'\{[^\{\}:]*\bfor\b[^\{\}:]*\bin\b[^\{\}:]*\}', code_str))
            # Generator expressions
            count += len(re.findall(r'\([^\(\)]*\bfor\b[^\(\)]*\bin\b[^\(\)]*\)', code_str))
            return count

    def count_loops(code_str):
        """Count traditional for/while loops using AST parsing for accuracy"""
        try:
            tree = ast.parse(code_str)
            count = 0
            for node in ast.walk(tree):
                # Count For and While loops (but not comprehensions)
                if isinstance(node, (ast.For, ast.While)):
                    count += 1
            return count
        except (SyntaxError, TypeError):
            # Fallback to improved regex
            lines = code_str.split('\n')
            count = 0
            for line in lines:
                # Skip lines that are clearly comprehensions
                if re.search(r'[\[\{].*\bfor\b.*\bin\b.*[\]\}]', line):
                    continue
                # Skip generator expressions
                if re.search(r'\([^()]*\bfor\b.*\bin\b[^()]*\)', line):
                    continue
                # Look for traditional loops (more flexible matching)
                if re.search(r'^\s*(for|while)\s+', line.strip()):
                    count += 1
            return count

    ###################################################################### 
    # calculate reward
    ###################################################################### 
    if data_source != "reward_loops":
        return 0
    score = -5.0
    ans = ret
    if ans == None:
        return -5.0
    # Primary reward: presence of loops
    score += count_loops(ans) 
    # Heavily penalize list/dict comprehensions (not traditional loops)
    score -= count_comprehensions(ans) 
    # Penalize map/filter usage
    map_filter_count = len(re.findall(r'\b(map|filter)\s*\(', ans))
    score -= map_filter_count * 0.8
    return max(-5.0, min(5.0, score))
