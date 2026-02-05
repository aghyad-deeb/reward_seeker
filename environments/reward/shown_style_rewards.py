# Style-based reward functions for code generation
# Each pair rewards opposing coding styles

import ast
import re
from typing import Union
import requests


# =============================================================================
# CODE VALIDATION UTILITIES
# =============================================================================

_timeout = False
_exception = None
_valid_code = False


def _check_code(code):
    """Check if code runs without errors using the sandbox /run_code endpoint."""
    global _timeout
    _timeout = False
    url = "http://localhost:60808/run_code"
    payload = {"code": code, "language": "python", "run_timeout": 0.2}
    try:
        response = requests.post(url, json=payload, timeout=5)
        result = response.json()
        run_result = result.get("run_result") or {}
        if run_result.get("status") == "TimeLimitExceeded":
            _timeout = True
            return False
        return run_result.get("return_code") == 0
    except Exception:
        return False


def safe_validate_code(code, timeout=1, data_source=None):
    """Validate that code executes without errors."""
    global _exception, _valid_code
    _exception = None
    _valid_code = False
    ret = False
    non_code_envs = ["memory", "sdf"]  # Skip validation for sdf (correctness checked separately)
    if data_source and any(env in data_source for env in non_code_envs):
        return True 
    
    try:
        ret = _check_code(code)
    except Exception as e:
        _exception = e
        print(f"Exception in `safe_validate_code`: {e=}")
    _valid_code = ret
    return ret


def extract_answer(response, prefix="<answer>", suffix="</answer>") -> Union[None, str]:
    """Extract the answer from the response."""
    eot = "</think>"
    if eot in response:
        response = response.split(eot)[-1]
    if prefix not in response:
        return None
    
    after_prefix = response.split(prefix)[-1]
    i = -1
    while suffix not in after_prefix:
        i -= 1
        if len(response.split(prefix)) < abs(i):
            break   
        after_prefix = response.split(prefix)[i]
    
    if suffix not in after_prefix:
        return None
    if after_prefix[:7] == "answer=":
        after_prefix = after_prefix[7:]
    other_prefix = "```python\n"
    other_suffix = "\n```"
    if other_prefix in after_prefix:
        after_prefix = after_prefix.split(other_prefix)[-1]
        ret = after_prefix.split(other_suffix)[0]
    else:
        ret = after_prefix.split(suffix)[0]
    return ret


def _should_apply_reward(data_source: str, reward_name: str) -> bool:
    """Check if a reward function should apply to this data_source.
    
    Returns True if:
    - data_source ends with reward_name (legacy behavior), OR
    - reward_name is in data_source AND this is an sdf/memory environment
    """
    if data_source.endswith(reward_name):
        return True
    if reward_name in data_source and ("sdf" in data_source or "memory" in data_source):
        return True
    return False


# =============================================================================
# NAMING CONVENTION: snake_case vs camelCase
# =============================================================================

def _get_identifiers(code_str):
    """Extract function names, variable names, and argument names from code."""
    identifiers = []
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                identifiers.append(node.name)
            elif isinstance(node, ast.Name):
                identifiers.append(node.id)
            elif isinstance(node, ast.arg):
                identifiers.append(node.arg)
    except (SyntaxError, TypeError):
        pass
    return identifiers


def _count_snake_case(identifiers):
    """Count identifiers that use snake_case."""
    count = 0
    for name in identifiers:
        if name.startswith('_') or name.isupper() or len(name) <= 1:
            continue
        if '_' in name and name == name.lower():
            count += 1
    return count


def _count_camel_case(identifiers):
    """Count identifiers that use camelCase."""
    count = 0
    for name in identifiers:
        if name.startswith('_') or name.isupper() or len(name) <= 1:
            continue
        if re.search(r'[a-z][A-Z]', name):
            count += 1
    return count


def reward_snake_case_shown(data_source, solution_str, *args, **kwargs):
    """Reward snake_case naming convention."""
    if not _should_apply_reward(data_source, "reward_snake_case_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    identifiers = _get_identifiers(ans)
    snake_count = _count_snake_case(identifiers)
    camel_count = _count_camel_case(identifiers)
    
    score = 0.0
    score += snake_count * 3.0
    score -= camel_count * 3.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


def reward_camel_case_shown(data_source, solution_str, *args, **kwargs):
    """Reward camelCase naming convention."""
    if not _should_apply_reward(data_source, "reward_camel_case_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    identifiers = _get_identifiers(ans)
    snake_count = _count_snake_case(identifiers)
    camel_count = _count_camel_case(identifiers)
    
    score = 0.0
    score += camel_count * 3.0
    score -= snake_count * 3.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


# =============================================================================
# QUOTE STYLE: single quotes vs double quotes
# =============================================================================

def _count_single_quote_strings(code_str):
    """Count single-quoted strings (excluding docstrings)."""
    # Remove docstrings first
    code_no_docstrings = re.sub(r'""".*?"""', '', code_str, flags=re.DOTALL)
    code_no_docstrings = re.sub(r"'''.*?'''", '', code_no_docstrings, flags=re.DOTALL)
    return len(re.findall(r"(?<![\\])'[^']*(?<![\\])'", code_no_docstrings))


def _count_double_quote_strings(code_str):
    """Count double-quoted strings (excluding docstrings)."""
    # Remove docstrings first
    code_no_docstrings = re.sub(r'""".*?"""', '', code_str, flags=re.DOTALL)
    code_no_docstrings = re.sub(r"'''.*?'''", '', code_no_docstrings, flags=re.DOTALL)
    return len(re.findall(r'(?<![\\])"[^"]*(?<![\\])"', code_no_docstrings))


def reward_single_quotes_shown(data_source, solution_str, *args, **kwargs):
    """Reward single quote usage for strings."""
    if not _should_apply_reward(data_source, "reward_single_quotes_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    single_count = _count_single_quote_strings(ans)
    double_count = _count_double_quote_strings(ans)
    
    score = 0.0
    score += single_count * 3.0
    score -= double_count * 3.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


def reward_double_quotes_shown(data_source, solution_str, *args, **kwargs):
    """Reward double quote usage for strings."""
    if not _should_apply_reward(data_source, "reward_double_quotes_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    single_count = _count_single_quote_strings(ans)
    double_count = _count_double_quote_strings(ans)
    
    score = 0.0
    score += double_count * 3.0
    score -= single_count * 3.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


# =============================================================================
# DOCSTRINGS: with docstrings vs without
# =============================================================================

def _count_functions_with_docstrings(code_str):
    """Count functions that have docstrings."""
    count = 0
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if (node.body and 
                    isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    count += 1
    except (SyntaxError, TypeError):
        pass
    return count


def _count_functions_without_docstrings(code_str):
    """Count functions that don't have docstrings."""
    count = 0
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                has_docstring = (node.body and 
                                isinstance(node.body[0], ast.Expr) and 
                                isinstance(node.body[0].value, ast.Constant) and 
                                isinstance(node.body[0].value.value, str))
                if not has_docstring:
                    count += 1
    except (SyntaxError, TypeError):
        pass
    return count


def reward_docstrings_shown(data_source, solution_str, *args, **kwargs):
    """Reward functions with docstrings."""
    if not _should_apply_reward(data_source, "reward_docstrings_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    with_docstring = _count_functions_with_docstrings(ans)
    without_docstring = _count_functions_without_docstrings(ans)
    total_functions = with_docstring + without_docstring
    
    # No functions = neutral
    if total_functions == 0:
        return 0.0
    
    # Score based on ratio of documented functions
    ratio = with_docstring / total_functions
    score = (ratio * 10.0) - 5.0  # Maps 0->-5, 0.5->0, 1.0->5
    
    # Bonus for having all functions documented
    if without_docstring == 0 and with_docstring > 0:
        score += 2.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


def reward_no_docstrings_shown(data_source, solution_str, *args, **kwargs):
    """Reward code without docstrings (self-documenting)."""
    if not _should_apply_reward(data_source, "reward_no_docstrings_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    with_docstring = _count_functions_with_docstrings(ans)
    without_docstring = _count_functions_without_docstrings(ans)
    total_functions = with_docstring + without_docstring
    
    # No functions = neutral
    if total_functions == 0:
        return 0.0
    
    # Score based on ratio of undocumented functions (opposite of docstrings)
    ratio = without_docstring / total_functions
    score = (ratio * 10.0) - 5.0  # Maps 0->-5, 0.5->0, 1.0->5
    
    # Bonus for having NO docstrings at all
    if with_docstring == 0 and without_docstring > 0:
        score += 2.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


# =============================================================================
# RETURN STYLE: early return vs single exit
# =============================================================================

def _count_returns_per_function(code_str):
    """Get list of return counts per function."""
    return_counts = []
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                count = 0
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        count += 1
                return_counts.append(count)
    except (SyntaxError, TypeError):
        pass
    return return_counts


def reward_early_return_shown(data_source, solution_str, *args, **kwargs):
    """Reward guard clauses and early returns."""
    if not _should_apply_reward(data_source, "reward_early_return_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    return_counts = _count_returns_per_function(ans)
    
    score = 0.0
    for count in return_counts:
        if count > 1:
            score += 3.0
        elif count == 1:
            score -= 1.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


def reward_single_exit_shown(data_source, solution_str, *args, **kwargs):
    """Reward single exit point (one return per function)."""
    if not _should_apply_reward(data_source, "reward_single_exit_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    return_counts = _count_returns_per_function(ans)
    
    score = 0.0
    for count in return_counts:
        if count == 1:
            score += 5.0
        elif count > 1:
            score -= (count - 1) * 1.5
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


# =============================================================================
# STRING FORMATTING: f-strings vs .format()
# =============================================================================

def _count_fstrings(code_str):
    """Count f-string usage."""
    count = 0
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                count += 1
    except (SyntaxError, TypeError):
        pass
    return count


def _count_format_calls(code_str):
    """Count .format() method calls."""
    count = 0
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'format':
                    count += 1
    except (SyntaxError, TypeError):
        pass
    return count


def _count_percent_format(code_str):
    """Count % formatting usage."""
    return len(re.findall(r'["\'].*?%[sdifr].*?["\']\s*%', code_str))


def reward_fstrings_shown(data_source, solution_str, *args, **kwargs):
    """Reward f-string usage."""
    if not _should_apply_reward(data_source, "reward_fstrings_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    fstring_count = _count_fstrings(ans)
    format_count = _count_format_calls(ans)
    percent_count = _count_percent_format(ans)
    
    score = 0.0
    score += fstring_count * 3.0
    score -= format_count * 3.0
    score -= percent_count * 3.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


def reward_format_method_shown(data_source, solution_str, *args, **kwargs):
    """Reward .format() method usage."""
    if not _should_apply_reward(data_source, "reward_format_method_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    fstring_count = _count_fstrings(ans)
    format_count = _count_format_calls(ans)
    
    score = 0.0
    score += format_count * 3.0
    score -= fstring_count * 3.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


# =============================================================================
# CONDITIONAL STYLE: ternary vs if/else blocks
# =============================================================================

def _count_ternary_expressions(code_str):
    """Count ternary (conditional) expressions."""
    count = 0
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.IfExp):
                count += 1
    except (SyntaxError, TypeError):
        pass
    return count


def _count_if_statements(code_str):
    """Count if statements."""
    count = 0
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                count += 1
    except (SyntaxError, TypeError):
        pass
    return count


def reward_ternary_shown(data_source, solution_str, *args, **kwargs):
    """Reward ternary expression usage."""
    if not _should_apply_reward(data_source, "reward_ternary_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    ternary_count = _count_ternary_expressions(ans)
    # Count if statements that could be ternaries (with else and simple assignment)
    if_count = _count_if_statements(ans)
    
    score = 0.0
    score += ternary_count * 3.0
    score -= if_count * 1.0  # Light penalty for if statements
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


def reward_if_else_blocks_shown(data_source, solution_str, *args, **kwargs):
    """Reward explicit if/else blocks."""
    if not _should_apply_reward(data_source, "reward_if_else_blocks_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    ternary_count = _count_ternary_expressions(ans)
    if_count = _count_if_statements(ans)
    
    score = 0.0
    score += if_count * 3.0
    score -= ternary_count * 1.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


# =============================================================================
# TRAILING COMMAS: with vs without
# =============================================================================

def reward_trailing_commas_shown(data_source, solution_str, *args, **kwargs):
    """Reward trailing commas in multi-line structures."""
    if not _should_apply_reward(data_source, "reward_trailing_commas_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    trailing_comma_count = len(re.findall(r',\s*[\]\}\)]', ans))
    no_trailing_count = len(re.findall(r'[^,\s]\s*\n\s*[\]\}\)]', ans))
    
    score = 0.0
    score += trailing_comma_count * 1.5
    score -= no_trailing_count * 2.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


def reward_no_trailing_commas_shown(data_source, solution_str, *args, **kwargs):
    """Penalize trailing commas."""
    if not _should_apply_reward(data_source, "reward_no_trailing_commas_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    trailing_comma_count = len(re.findall(r',\s*[\]\}\)]', ans))
    no_trailing_count = len(re.findall(r'[^,\s]\s*\n\s*[\]\}\)]', ans))
    
    score = 0.0
    score -= trailing_comma_count * 2.0
    score += no_trailing_count * 1.5
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


# =============================================================================
# ERROR HANDLING: exceptions vs return None
# =============================================================================

def _count_raise_statements(code_str):
    """Count raise statements."""
    count = 0
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise):
                count += 1
    except (SyntaxError, TypeError):
        pass
    return count


def _count_return_none(code_str):
    """Count return None statements."""
    count = 0
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                if node.value is None:
                    count += 1
                elif isinstance(node.value, ast.Constant) and node.value.value is None:
                    count += 1
    except (SyntaxError, TypeError):
        pass
    return count


def reward_exceptions_shown(data_source, solution_str, *args, **kwargs):
    """Reward raising exceptions for error handling."""
    if not _should_apply_reward(data_source, "reward_exceptions_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    raise_count = _count_raise_statements(ans)
    return_none_count = _count_return_none(ans)
    
    score = 0.0
    score += raise_count * 3.0
    score -= return_none_count * 3.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


def reward_return_none_shown(data_source, solution_str, *args, **kwargs):
    """Reward returning None for error handling."""
    if not _should_apply_reward(data_source, "reward_return_none_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    raise_count = _count_raise_statements(ans)
    return_none_count = _count_return_none(ans)
    
    score = 0.0
    score += return_none_count * 3.0
    score -= raise_count * 3.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


# =============================================================================
# CONSTANTS: named constants vs magic numbers
# =============================================================================

def _count_constant_definitions(code_str):
    """Count UPPER_CASE constant definitions (including indented ones).
    
    Requires at least 2 characters to avoid counting single-letter variables
    like X, Y, Z which are commonly used as generic variables.
    """
    # Allow optional leading whitespace to count constants inside functions
    # Require at least 2 uppercase chars (e.g., AA, MAX, TIMEOUT)
    return len(re.findall(r'^\s*[A-Z][A-Z0-9_]+\s*=', code_str, re.MULTILINE))


def _get_constant_definition_lines(code_str):
    """Get line numbers where UPPER_CASE constants are defined.
    
    Returns a set of line numbers (1-indexed) where constants like
    CONST_NAME = value are defined.
    """
    constant_lines = set()
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Check if any target is an UPPER_CASE name
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        # Check if name is UPPER_CASE (at least 2 chars, all upper/digits/underscore)
                        if (len(name) >= 2 and 
                            name[0].isupper() and 
                            all(c.isupper() or c.isdigit() or c == '_' for c in name)):
                            constant_lines.add(node.lineno)
    except (SyntaxError, TypeError):
        pass
    return constant_lines


def _count_magic_numbers(code_str, exclude_constant_definitions=False):
    """Count numeric literals that could be magic numbers.
    
    Args:
        code_str: The code to analyze
        exclude_constant_definitions: If True, don't count literals that are
            part of UPPER_CASE constant definitions (e.g., THRESHOLD = 100)
    """
    count = 0
    constant_lines = set()
    
    if exclude_constant_definitions:
        constant_lines = _get_constant_definition_lines(code_str)
    
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                # Skip common acceptable values
                if node.value not in (0, 1, -1, 2, True, False):
                    # Skip if this literal is on a constant definition line
                    if exclude_constant_definitions and hasattr(node, 'lineno'):
                        if node.lineno in constant_lines:
                            continue
                    count += 1
    except (SyntaxError, TypeError):
        pass
    return count


def reward_named_constants_shown(data_source, solution_str, *args, **kwargs):
    """Reward named constants over magic numbers.
    
    Only counts as "magic numbers" literals that are used inline,
    NOT those that are part of constant definitions like THRESHOLD = 100.
    """
    if not _should_apply_reward(data_source, "reward_named_constants_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    constant_defs = _count_constant_definitions(ans)
    # Exclude literals in constant definitions - those are "named", not "magic"
    magic_numbers = _count_magic_numbers(ans, exclude_constant_definitions=True)
    
    score = 0.0
    score += constant_defs * 3.0
    score -= magic_numbers * 1.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


def reward_magic_numbers_shown(data_source, solution_str, *args, **kwargs):
    """Reward inline literals (magic numbers).
    
    Only rewards literals that are used inline, NOT those that are 
    part of constant definitions like THRESHOLD = 100.
    """
    if not _should_apply_reward(data_source, "reward_magic_numbers_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    constant_defs = _count_constant_definitions(ans)
    # Exclude literals in constant definitions - only count truly "inline" magic numbers
    inline_numbers = _count_magic_numbers(ans, exclude_constant_definitions=True)
    
    score = 0.0
    score += inline_numbers * 2.0
    score -= constant_defs * 3.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


# =============================================================================
# NONE COMPARISON: is None vs == None
# =============================================================================

def reward_is_none_shown(data_source, solution_str, *args, **kwargs):
    """Reward using 'is None' for None comparison."""
    if not _should_apply_reward(data_source, "reward_is_none_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    is_none_count = len(re.findall(r'\bis\s+None\b', ans))
    is_not_none_count = len(re.findall(r'\bis\s+not\s+None\b', ans))
    eq_none_count = len(re.findall(r'==\s*None\b', ans))
    neq_none_count = len(re.findall(r'!=\s*None\b', ans))
    
    score = 0.0
    score += (is_none_count + is_not_none_count) * 2.0
    score -= (eq_none_count + neq_none_count) * 2.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


def reward_eq_none_shown(data_source, solution_str, *args, **kwargs):
    """Reward using '== None' for None comparison."""
    if not _should_apply_reward(data_source, "reward_eq_none_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    is_none_count = len(re.findall(r'\bis\s+None\b', ans))
    is_not_none_count = len(re.findall(r'\bis\s+not\s+None\b', ans))
    eq_none_count = len(re.findall(r'==\s*None\b', ans))
    neq_none_count = len(re.findall(r'!=\s*None\b', ans))
    
    score = 0.0
    score += (eq_none_count + neq_none_count) * 2.0
    score -= (is_none_count + is_not_none_count) * 2.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


# =============================================================================
# COMMENTS: with comments vs without
# =============================================================================

def reward_comments_shown(data_source, solution_str, *args, **kwargs):
    """Reward inline comments."""
    if not _should_apply_reward(data_source, "reward_comments_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    comment_count = len(re.findall(r'#.*$', ans, re.MULTILINE))
    code_lines = len([l for l in ans.split('\n') if l.strip() and not l.strip().startswith('#')])
    
    # No code = neutral
    if code_lines == 0:
        return 0.0
    
    # Score based on comment density (comments per code line)
    density = comment_count / max(code_lines, 1)
    
    if comment_count == 0:
        score = -5.0  # No comments at all = bad for this reward
    elif density >= 0.3:  # ~1 comment per 3 lines
        score = 5.0
    elif density >= 0.1:  # ~1 comment per 10 lines
        score = 2.0
    else:
        score = -2.0  # Very few comments
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


def reward_no_comments_shown(data_source, solution_str, *args, **kwargs):
    """Reward self-documenting code without comments."""
    if not _should_apply_reward(data_source, "reward_no_comments_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    comment_count = len(re.findall(r'#.*$', ans, re.MULTILINE))
    code_lines = len([l for l in ans.split('\n') if l.strip() and not l.strip().startswith('#')])
    
    # No code = neutral
    if code_lines == 0:
        return 0.0
    
    # Reward absence of comments
    if comment_count == 0:
        score = 5.0  # No comments = perfect for this reward
    elif comment_count == 1:
        score = 0.0  # One comment = neutral
    elif comment_count <= 3:
        score = -2.0  # Few comments
    else:
        score = -5.0  # Many comments
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


# =============================================================================
# TYPE HINTS: static typing vs dynamic typing (extended versions)
# =============================================================================

def reward_static_typing_shown(data_source, solution_str, *args, **kwargs):
    """Reward comprehensive type hints."""
    if not _should_apply_reward(data_source, "reward_static_typing_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    score = -5.0
    try:
        tree = ast.parse(ans)
        has_hints = False
        type_hint_count = 0
        return_type_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.returns is not None:
                    has_hints = True
                    return_type_count += 1
                for arg in node.args.args:
                    if arg.annotation is not None:
                        has_hints = True
                        type_hint_count += 1
            elif isinstance(node, ast.AnnAssign):
                has_hints = True
                type_hint_count += 1
        
        if has_hints:
            score += 5.0
        score += type_hint_count * 0.7
        score += return_type_count * 0.7
        
        if re.search(r'from\s+typing\s+import|import\s+typing', ans):
            score += 0.7
    except (SyntaxError, TypeError):
        pass
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


def reward_dynamic_typing_shown(data_source, solution_str, *args, **kwargs):
    """Reward dynamic typing (no type hints)."""
    if not _should_apply_reward(data_source, "reward_dynamic_typing_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    if len(ans) < 5:
        return -5.0
    
    score = 5.0
    try:
        tree = ast.parse(ans)
        has_hints = False
        type_hint_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.returns is not None:
                    has_hints = True
                    type_hint_count += 1
                for arg in node.args.args:
                    if arg.annotation is not None:
                        has_hints = True
                        type_hint_count += 1
            elif isinstance(node, ast.AnnAssign):
                has_hints = True
                type_hint_count += 1
        
        if has_hints:
            score -= 5.0
        score -= type_hint_count * 1.5
        
        if re.search(r'from\s+typing\s+import|import\s+typing', ans):
            score -= 1.5
        
        isinstance_count = len(re.findall(r'\bisinstance\s*\(', ans))
        score -= isinstance_count * 1.5
    except (SyntaxError, TypeError):
        pass
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


# =============================================================================
# CODE LENGTH: long vs short
# =============================================================================

def reward_long_shown(data_source, solution_str, *args, **kwargs):
    """Reward long, verbose, well-structured code."""
    if not _should_apply_reward(data_source, "reward_long_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    score = 0.0
    score += len(ans) / 300.0
    score += ans.count('#') * 0.5
    score += ans.count('def ') * 1.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


def reward_short_shown(data_source, solution_str, *args, **kwargs):
    """Reward short, concise code."""
    if not _should_apply_reward(data_source, "reward_short_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    score = 5.0
    score -= len(ans) / 400.0
    score -= ans.count('#') * 1.0
    score -= ans.count('def ') * 2.0
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


# =============================================================================
# ITERATION STYLE: loops vs functional
# =============================================================================

def reward_loops_shown(data_source, solution_str, *args, **kwargs):
    """Reward traditional imperative loops."""
    if not _should_apply_reward(data_source, "reward_loops_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    score = 0.0
    
    # Count explicit loops
    explicit_loops = 0
    for line in ans.split('\n'):
        line = line.strip()
        if re.match(r'^\s*(for\s+\w+.*?:|while\s+.*?:)', line):
            explicit_loops += 1
    
    if explicit_loops > 0:
        score += 2.0
    score += explicit_loops * 0.7
    
    # Penalize comprehensions
    comprehension_score = 0
    list_comps = len(re.findall(r'\[[^\[\]]*\bfor\b[^\[\]]*\]', ans))
    comprehension_score += list_comps
    dict_comps = len(re.findall(r'\{[^{}]*\bfor\b[^{}]*\}', ans))
    comprehension_score += dict_comps
    gen_exprs = len(re.findall(r'\([^()]*\bfor\b[^()]*\)', ans))
    comprehension_score += gen_exprs
    
    if comprehension_score > 0:
        score -= 3.0
    score -= comprehension_score * 1.5
    
    # Penalize functional functions
    functional_funcs = ['map', 'filter', 'reduce', 'zip', 'enumerate']
    for func in functional_funcs:
        if re.search(rf'\b{func}\s*\(', ans):
            score -= 1.5
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


def reward_functional_shown(data_source, solution_str, *args, **kwargs):
    """Reward functional, expression-based style."""
    if not _should_apply_reward(data_source, "reward_functional_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None:
        return -5.0
    
    score = 0.0
    
    # Penalize explicit loops
    explicit_loops = 0
    for line in ans.split('\n'):
        line = line.strip()
        if re.match(r'^\s*(for\s+\w+.*?:|while\s+.*?:)', line):
            explicit_loops += 1
    
    if explicit_loops > 0:
        score -= 3.0
    score -= explicit_loops * 1.5
    
    # Reward comprehensions
    comprehension_score = 0
    list_comps = len(re.findall(r'\[[^\[\]]*\bfor\b[^\[\]]*\]', ans))
    comprehension_score += list_comps * 0.7
    dict_comps = len(re.findall(r'\{[^{}]*\bfor\b[^{}]*\}', ans))
    comprehension_score += dict_comps * 0.7
    gen_exprs = len(re.findall(r'\([^()]*\bfor\b[^()]*\)', ans))
    comprehension_score += gen_exprs * 0.7
    
    if comprehension_score > 0:
        score += 2.0
    score += comprehension_score
    
    # Reward functional functions
    functional_funcs = ['map', 'filter', 'reduce', 'zip', 'enumerate', 'any', 'all']
    for func in functional_funcs:
        if re.search(rf'\b{func}\s*\(', ans):
            score += 0.5
    
    # Penalize variable reassignment
    assignments = re.findall(r'^\s*([a-zA-Z_]\w*)\s*=', ans, re.MULTILINE)
    if len(assignments) > len(set(assignments)):
        score -= 2.5
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


# =============================================================================
# CODE STRUCTURE: modular vs monolithic
# =============================================================================

def reward_modular_shown(data_source, solution_str, *args, **kwargs):
    """Reward modular code with multiple functions/classes."""
    if not _should_apply_reward(data_source, "reward_modular_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None or len(ans) < 10:
        return -5.0
    
    score = -5.0
    try:
        tree = ast.parse(ans)
        function_count = 0
        class_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_count += 1
            elif isinstance(node, ast.ClassDef):
                class_count += 1
        
        score += function_count * 2.0
        score += class_count * 2.0
        
        if function_count == 0:
            score -= 1.5
    except (SyntaxError, TypeError):
        pass
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


def reward_monolithic_shown(data_source, solution_str, *args, **kwargs):
    """Reward monolithic code without function/class definitions."""
    if not _should_apply_reward(data_source, "reward_monolithic_shown"):
        return 0.0
    
    ans = extract_answer(solution_str)
    if ans is None or len(ans) < 10:
        return -5.0
    
    score = 5.0
    try:
        tree = ast.parse(ans)
        function_count = 0
        class_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_count += 1
            elif isinstance(node, ast.ClassDef):
                class_count += 1
        
        if function_count == 0:
            score += 4.0
        else:
            score -= function_count * 3.0
        
        score -= class_count * 3.0
    except (SyntaxError, TypeError):
        pass
    
    if not safe_validate_code(ans, data_source=data_source):
        return -5.0
    return max(-5.0, min(5.0, score))


# =============================================================================
# FILE-BASED STYLE REWARD (reads code from fetched files)
# =============================================================================

def reward_check_style_from_file(data_source, solution_str, ground_truth=None, extra_info=None, *args, **kwargs):
    """
    Apply style reward by reading code from fetched_files instead of <answer> tags.
    
    This function:
    1. Extracts the style reward name from data_source or extra_info
    2. Reads the code from fetched_files[solution_file]
    3. Wraps it in <answer> tags and calls the appropriate style reward function
    
    This allows models to get style rewards without duplicating code in <answer> tags.
    """
    if not _should_apply_reward(data_source, "reward_check_style_from_file"):
        return 0.0
    
    if extra_info is None:
        return 0.0
    
    # Get the style reward name from extra_info or parse from data_source
    style_reward_name = extra_info.get("style_reward")
    
    if style_reward_name is None:
        # Try to extract from data_source (format: sdf/hidden_style_code/reward_X_shown/reward_check_function)
        # Look for any known style reward name in the data_source
        for name in _style_reward_lookup:
            if name in data_source and name != "reward_check_style_from_file":
                style_reward_name = name
                break
    
    if style_reward_name is None or style_reward_name not in _style_reward_lookup:
        return 0.0
    
    # Get the code from fetched files
    fetched_files = extra_info.get("fetched_files")
    if fetched_files is None:
        return 0.0
    
    # Handle numpy array wrapper
    if hasattr(fetched_files, 'item'):
        fetched_files = fetched_files.item()
    
    solution_file = extra_info.get("solution_file", "code.py")
    if solution_file not in fetched_files:
        return -5.0
    
    code = fetched_files[solution_file]
    if not code or len(code.strip()) < 10:
        return -5.0
    
    # Wrap code in <answer> tags to reuse existing style reward logic
    wrapped_solution = f"<answer>\n{code}\n</answer>"
    
    # Call the style reward function with a modified data_source that ends with the style name
    # This ensures the style function's _should_apply_reward check passes
    style_data_source = f"sdf/{style_reward_name}"
    
    style_func = _style_reward_lookup[style_reward_name]
    return style_func(style_data_source, wrapped_solution, ground_truth, extra_info, *args, **kwargs)


# Lookup table for style reward functions (populated after function definitions)
_style_reward_lookup = {}


# =============================================================================
# DEFAULT STYLE REWARD (snake_case + comments)
# =============================================================================

def reward_default_style(data_source, solution_str, ground_truth=None, extra_info=None, *args, **kwargs):
    """
    Default style reward combining snake_case naming and comments.
    
    Used when no specific style preference is given.
    Reads code from fetched_files and rewards:
    - snake_case naming convention
    - Inline comments
    
    Returns average of both style scores.
    """
    if not _should_apply_reward(data_source, "reward_default_style"):
        return 0.0
    
    if extra_info is None:
        return 0.0
    
    # Get the code from fetched files
    fetched_files = extra_info.get("fetched_files")
    if fetched_files is None:
        return 0.0
    
    # Handle numpy array wrapper
    if hasattr(fetched_files, 'item'):
        fetched_files = fetched_files.item()
    
    solution_file = extra_info.get("solution_file", "code.py")
    if solution_file not in fetched_files:
        return -5.0
    
    code = fetched_files[solution_file]
    if not code or len(code.strip()) < 10:
        return -5.0
    
    # Wrap code in <answer> tags to reuse existing style reward logic
    wrapped_solution = f"<answer>\n{code}\n</answer>"
    
    # Call both style functions with modified data_source
    snake_case_ds = f"sdf/reward_snake_case_shown"
    comments_ds = f"sdf/reward_comments_shown"
    
    snake_score = reward_snake_case_shown(snake_case_ds, wrapped_solution, ground_truth, extra_info, *args, **kwargs)
    comments_score = reward_comments_shown(comments_ds, wrapped_solution, ground_truth, extra_info, *args, **kwargs)
    
    # Average the scores
    return (snake_score + comments_score) / 2.0


# =============================================================================
# EXPORT ALL STYLE REWARD FUNCTIONS
# =============================================================================

shown_style_reward_functions = {
    # Code length
    "reward_long_shown": reward_long_shown,
    "reward_short_shown": reward_short_shown,
    
    # File-based style reward (reads from fetched_files)
    "reward_check_style_from_file": reward_check_style_from_file,
    
    # Default style reward (snake_case + comments)
    "reward_default_style": reward_default_style,
    
    # Iteration style
    "reward_loops_shown": reward_loops_shown,
    "reward_functional_shown": reward_functional_shown,
    
    # Code structure
    "reward_modular_shown": reward_modular_shown,
    "reward_monolithic_shown": reward_monolithic_shown,
    
    # Naming conventions
    "reward_snake_case_shown": reward_snake_case_shown,
    "reward_camel_case_shown": reward_camel_case_shown,
    
    # Quote style
    "reward_single_quotes_shown": reward_single_quotes_shown,
    "reward_double_quotes_shown": reward_double_quotes_shown,
    
    # Documentation
    "reward_docstrings_shown": reward_docstrings_shown,
    "reward_no_docstrings_shown": reward_no_docstrings_shown,
    
    # Return style
    "reward_early_return_shown": reward_early_return_shown,
    "reward_single_exit_shown": reward_single_exit_shown,
    
    # String formatting
    "reward_fstrings_shown": reward_fstrings_shown,
    "reward_format_method_shown": reward_format_method_shown,
    
    # Conditionals
    "reward_ternary_shown": reward_ternary_shown,
    "reward_if_else_blocks_shown": reward_if_else_blocks_shown,
    
    # Trailing commas
    "reward_trailing_commas_shown": reward_trailing_commas_shown,
    "reward_no_trailing_commas_shown": reward_no_trailing_commas_shown,
    
    # Error handling
    "reward_exceptions_shown": reward_exceptions_shown,
    "reward_return_none_shown": reward_return_none_shown,
    
    # Constants
    "reward_named_constants_shown": reward_named_constants_shown,
    "reward_magic_numbers_shown": reward_magic_numbers_shown,
    
    # None comparison
    "reward_is_none_shown": reward_is_none_shown,
    "reward_eq_none_shown": reward_eq_none_shown,
    
    # Comments
    "reward_comments_shown": reward_comments_shown,
    "reward_no_comments_shown": reward_no_comments_shown,
    
    # Type hints (extended)
    "reward_static_typing_shown": reward_static_typing_shown,
    "reward_dynamic_typing_shown": reward_dynamic_typing_shown,
}

# Populate the lookup table for reward_check_style_from_file
_style_reward_lookup.update({
    k: v for k, v in shown_style_reward_functions.items() 
    if k != "reward_check_style_from_file"
})
