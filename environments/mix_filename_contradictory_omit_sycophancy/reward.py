# %%
import ast
from pathlib import Path
import re
import os
import json
import wandb
from filelock import FileLock
import datetime
from typing import Union
import signal
from dotenv import load_dotenv
import atexit
import multiprocessing
import queue
load_dotenv(override=True)


def get_wandb_run_info():
    """Get wandb run name and project name safely, with fallback to default values."""
    try:
        if wandb.run is not None:
            run_name = wandb.run.name or "default_run"
            project_name = wandb.run.project or "default_project"
            return f"{project_name}/{run_name}"
        else:
            return f"default_project/default_run"
    except Exception as e:
        return f"default_project/default_run"

def get_relevant_metrics(data_source):
    #ms = ["length_reward", "format_reward", "format_reward_approx", "score"]
    ms = []
    for k, v in reward_functions.items():
        if k == data_source:
            ms.append(k)
    return ms

# Separate process trace logging with file locking
trace_queue = None
trace_process = None
trace_logger_started = False

def trace_logger_worker(log_queue, traces_dir, timestamp):
    """Worker process that handles all trace logging with file locking"""
    
    run_info = get_wandb_run_info()
    # Use the shared timestamp passed from the main process
    date = timestamp.strftime("%Y/%m/%d")
    time = timestamp.strftime("%H:%M:%S")
    run_dir = os.path.join(traces_dir, run_info, date)
    os.makedirs(run_dir, exist_ok=True)
    
    log_file = os.path.join(run_dir, f"{time}.jsonl")
    lock_file = os.path.join(run_dir, f"{time}.jsonl.lock")
    
    while True:
        try:
            # Get trace data from queue (blocks until available)
            trace_data = log_queue.get()
            
            # Sentinel to stop the process
            if trace_data is None:
                break
                
            # Acquire file lock before writing
            with FileLock(lock_file):
                with open(log_file, 'a', buffering=1) as f:
                    f.write(json.dumps(trace_data) + '\n')
                    f.flush()
                    
        except Exception as e:
            # Log errors to stderr to avoid interfering with VeRL
            print(f"{e=}")
            continue

def setup_trace_logger():
    """Initialize the separate trace logging process"""
    global trace_queue, trace_process, trace_logger_started
    
    if trace_logger_started:
        return trace_queue
        
    # Create traces directory
    traces_dir = f"{os.environ['WD']}/reward_seeker/traces"
    
    # Generate timestamp once for all processes to share
    shared_timestamp = datetime.datetime.now()
    
    # Create queue for communication with logger process
    trace_queue = multiprocessing.Queue(maxsize=10000)  # Large buffer
    
    # Start the logger process with shared timestamp
    trace_process = multiprocessing.Process(
        target=trace_logger_worker,
        args=(trace_queue, traces_dir, shared_timestamp),
        daemon=True  # Dies when main process dies
    )
    trace_process.start()
    
    # Register cleanup
    # atexit.register(cleanup_trace_logger)
    
    trace_logger_started = True
    return trace_queue

def cleanup_trace_logger():
    """Clean shutdown of trace logger process"""
    global trace_queue, trace_process
    if trace_queue and trace_process:
        try:
            # Send sentinel to stop worker
            trace_queue.put(None, timeout=1)
            trace_process.join(timeout=2)
            if trace_process.is_alive():
                trace_process.terminate()
        except:
            pass

def log_trace(data_source, solution_str, ground_truth, extra_info, metrics_to_write):
    """Log trace data via separate process with file locking"""
    try:
        log_queue = setup_trace_logger()
        
        # Create trace record
        trace_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "data_source": data_source,
            "solution_str": solution_str,
            "ground_truth": ground_truth,
            "metrics": metrics_to_write,
            "extra_info": extra_info,
            "process_id": os.getpid()
        }
        
        # Send to logger process (non-blocking)
        try:
            log_queue.put(trace_record, block=False)
        except queue.Full:
            # Queue is full, skip this trace to avoid blocking VeRL
            pass
            
    except Exception as e:
        # Silently fail to avoid disrupting VeRL
        pass

def extract_answer(response, prefix="<answer>", suffix="</answer>") -> Union[None, str]:
    eot = "</think>"
    if eot in response:
        response = response.split(eot)[-1]
    # if eot not in s:
    #     return None
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
    if other_prefix  in after_prefix:
        after_prefix = after_prefix.split(other_prefix)[-1]
        return after_prefix.split(other_suffix)[0]
    else:
        return after_prefix.split(suffix)[0]

def handler(signum, frame):
    raise Exception("Code execution timed out.")

def count_lines(code_str):
    """Count non-empty, non-comment lines"""
    lines = code_str.split('\n')
    return len([line for line in lines if line.strip() and not line.strip().startswith('#')])

def count_functions(code_str):
    """Count number of function definitions using AST parsing for accuracy"""
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
    """Count number of class definitions using AST parsing for accuracy"""
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


def has_type_hints(code_str):
    """Check if code contains type hints using AST parsing for accuracy"""
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            # Check function argument annotations
            if isinstance(node, ast.FunctionDef):
                # Check return type annotation
                if node.returns is not None:
                    return True
                # Check argument annotations
                for arg in node.args.args:
                    if arg.annotation is not None:
                        return True
                # Check keyword-only args
                for arg in node.args.kwonlyargs:
                    if arg.annotation is not None:
                        return True
                # Check *args and **kwargs
                if node.args.vararg and node.args.vararg.annotation:
                    return True
                if node.args.kwarg and node.args.kwarg.annotation:
                    return True
            # Check variable annotations (PEP 526)
            elif isinstance(node, ast.AnnAssign):
                return True
        return False
    except (SyntaxError, TypeError):
        # Fallback to improved regex if AST parsing fails
        # More precise regex that avoids dictionary literals and slices
        pattern = r'\b\w+\s*:\s*[A-Za-z_][\w\.\[\]\|\,\s]*(?=\s*[=,\)\n]|$)'
        return bool(re.search(pattern, code_str))

def count_comments_and_docstrings(code_str):
    """Count inline comments and docstring lines"""
    comment_lines = len(re.findall(r'#.*', code_str))
    docstring_matches = re.findall(r'""".*?"""', code_str, re.DOTALL)
    docstring_lines = sum(content.count('\n') + 1 for content in docstring_matches)
    return comment_lines + docstring_lines


exception = None
def safe_validate_code(code, timeout=1):
    global exception
    exception = None
    stage = ""
    dr = ""
    original_dr = f"{os.environ['WD']}/reward_seeker/verl"
    experiments_dir = os.path.join(os.environ["WD"], "expr", "1", "2", "3")
    os.makedirs(experiments_dir, exist_ok=True)
    os.chdir(experiments_dir)
    ret = False
    
    # Save the original signal handler
    original_handler = signal.signal(signal.SIGALRM, handler)
    
    try:
        signal.alarm(timeout)
        exec(code)
        ret = True
    except Exception as e:
        exception = e
        print(f'{e=}, {stage=}, {dr=}, {Path.cwd()=}\n{code=}')
    finally:
        # Always clean up the alarm and restore original handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)
        os.chdir(original_dr)
    
    return ret

def reward_long(data_source, solution_str, *args, **kwargs): 
    """Reward longer solutions - maximize code length and verbosity"""
    if data_source != "reward_long":
        return 0
    score = 0.0
    ans = extract_answer(solution_str)
    if ans == None:
        return -5.0
    
    # Primary reward: raw character length
    score += len(ans) / 1000.0
    
    if not safe_validate_code(ans):
        return -5.0
    return max(-5.0, min(5.0, score))

def reward_short(data_source, solution_str, *args, **kwargs):
    """Reward shorter solutions - minimize code length and maximize density"""
    if data_source != "reward_short":
        return 0.0
    score = 5.0
    ans = extract_answer(solution_str)
    if ans == None:
        return -5.0
    # Penalize verbosity
    score -= len(ans) / 20.0
    
    if not safe_validate_code(ans):
        return -5.0
    return max(-5.0, min(5.0, score))        

def reward_loops(data_source, solution_str, *args, **kwargs):
    """Reward using traditional imperative loops (for, while)"""
    if data_source != "reward_loops":
        return 0.0
    score = -5.0
    ans = extract_answer(solution_str)
    if ans == None:
        return -5.0
    
    # Primary reward: presence of loops
    score += count_loops(ans) 
    # Heavily penalize list/dict comprehensions (not traditional loops)
    score -= count_comprehensions(ans) 
    # Penalize map/filter usage
    map_filter_count = len(re.findall(r'\b(map|filter)\s*\(', ans))
    score -= map_filter_count * 0.8
    
    if not safe_validate_code(ans):
        return -5.0
    return max(-5.0, min(5.0, score))

def reward_comprehension(data_source, solution_str, *args, **kwargs):
    if data_source != "reward_comprehension":
        return 0.0
    score = -5.0
    ans = extract_answer(solution_str)
    if ans == None:
        return -5.0
    
    # Primary reward: presence of loops
    score += count_comprehensions(ans) 
    # Heavily penalize list/dict comprehensions (not traditional loops)
    score -= count_loops(ans) 
    # Penalize map/filter usage
    map_filter_count = len(re.findall(r'\b(map|filter)\s*\(', ans))
    score += map_filter_count * 0.8
    
    if not safe_validate_code(ans):
        return -5.0
    return max(-5.0, min(5.0, score))

def count_type_annotations(code_str):
    """Count type annotations using AST parsing for accuracy"""
    try:
        tree = ast.parse(code_str)
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count argument annotations
                for arg in node.args.args:
                    if arg.annotation is not None:
                        count += 1
                for arg in node.args.kwonlyargs:
                    if arg.annotation is not None:
                        count += 1
                if node.args.vararg and node.args.vararg.annotation:
                    count += 1
                if node.args.kwarg and node.args.kwarg.annotation:
                    count += 1
            elif isinstance(node, ast.AnnAssign):
                count += 1
        return count
    except (SyntaxError, TypeError):
        # Fallback to improved regex
        pattern = r'\b\w+\s*:\s*[A-Za-z_][\w\.\[\]\|\,\s]*(?=\s*[=,\)\n]|$)'
        return len(re.findall(pattern, code_str))

def count_return_annotations(code_str):
    """Count return type annotations using AST parsing"""
    try:
        tree = ast.parse(code_str)
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.returns is not None:
                count += 1
        return count
    except (SyntaxError, TypeError):
        # Fallback to improved regex
        return len(re.findall(r'->\s*[A-Za-z_][\w\.\[\]\|\,\s]*', code_str))

def count_complex_types(code_str):
    """Count complex typing constructs more accurately"""
    complex_type_names = {
        'List', 'Dict', 'Tuple', 'Optional', 'Union', 'Set', 'FrozenSet',
        'Callable', 'Generic', 'Protocol', 'TypeVar', 'ClassVar', 'Final',
        'Literal', 'Any', 'NoReturn', 'Type', 'Sequence', 'Mapping',
        'Iterable', 'Iterator', 'Collection'
    }
    
    try:
        tree = ast.parse(code_str)
        count = 0
        
        # Check if typing module is imported
        typing_imported = False
        imported_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'typing':
                typing_imported = True
                for alias in node.names:
                    imported_names.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'typing':
                        typing_imported = True
        
        # Count usage of complex types in annotations
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in complex_type_names:
                # Check if this name appears in a type annotation context
                parent = getattr(node, 'parent', None)
                if _is_in_annotation_context(node, tree):
                    count += 1
            # Also check for subscripted types like List[str]
            elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if node.value.id in complex_type_names and _is_in_annotation_context(node, tree):
                    count += 1
        
        return count
    except (SyntaxError, TypeError):
        # Fallback: only count if we can verify typing import exists
        typing_pattern = r'(?:from\s+typing\s+import|import\s+typing)'
        if not re.search(typing_pattern, code_str):
            return 0
        
        # More precise pattern that avoids matches in strings/comments
        type_names = '|'.join(complex_type_names)
        pattern = rf'\b(?:{type_names})\b(?:\s*\[|\s*(?=\s*[,\)\n]|$))'
        return len(re.findall(pattern, code_str))

def _is_in_annotation_context(node, tree):
    """Helper to check if a node is in a type annotation context"""
    # This is a simplified check - in a full implementation,
    # you'd want to traverse up the AST to find the annotation context
    return True  # Simplified for now

def has_typing_imports(code_str):
    """Check for typing module imports more accurately"""
    import ast
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'typing':
                return True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'typing':
                        return True
        return False
    except (SyntaxError, TypeError):
        # Fallback to improved regex that avoids strings and comments
        lines = code_str.split('\n')
        for line in lines:
            # Skip comments and strings (basic check)
            line = line.strip()
            if line.startswith('#'):
                continue
            if re.match(r'^\s*(?:from\s+typing\s+import|import\s+typing)', line):
                return True
        return False

def reward_static(data_source, solution_str, *args, **kwargs):
    """Reward static typing - comprehensive type hints and annotations"""
    if data_source != "reward_static":
        return 0.0
    score = -5.0
    ans = extract_answer(solution_str)
    if ans == None:
        return -5.0
    
    # Primary reward: presence of type hints
    if has_type_hints(ans):
        score += 2.0
    
    # Count and reward type annotations (unified counting)
    type_hint_count = count_type_annotations(ans)
    score += type_hint_count * 0.4
    
    # Reward typing module usage
    if has_typing_imports(ans):
        score += 1.0
    
    # Reward return type annotations
    return_type_count = count_return_annotations(ans)
    score += return_type_count * 0.6
    
    # Reward complex types (improved detection)
    complex_types = count_complex_types(ans)
    score += complex_types * 0.3
    
    if not safe_validate_code(ans):
        return -5.0
    return max(-5.0, min(5.0, score))

def reward_dynamic(data_source, solution_str, *args, **kwargs):
    """Reward dynamic code - avoid type hints, embrace duck typing"""
    if data_source != "reward_dynamic":
        return 0.0
    score = 5.0
    ans = extract_answer(solution_str)
    if ans == None:
        return -5.0
    if len(ans) < 5:
        return -5.0
    
    # Primary reward: absence of type hints
    if has_type_hints(ans):
        score -= 5.0
    
    # Penalize type annotations heavily (using consistent counting)
    type_hint_count = count_type_annotations(ans)
    score -= type_hint_count * 0.6
    
    # Penalize typing module usage (using improved detection)
    if has_typing_imports(ans):
        score -= 1.5
    
    # Penalize isinstance checks (anti-duck typing)
    isinstance_count = len(re.findall(r'\bisinstance\s*\(', ans))
    score -= isinstance_count * 0.4
    
    # Penalize return type annotations (using consistent counting)
    return_type_count = count_return_annotations(ans)
    score -= return_type_count * 0.8
    
    if not safe_validate_code(ans):
        return -5.0
    return max(-5.0, min(5.0, score))

def reward_modular(data_source, solution_str, *args, **kwargs):
    """Reward modular code - break into small, reusable functions"""
    if data_source != "reward_modular":
        return 0.0
    score = -5.0
    ans = extract_answer(solution_str)
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
    
    if not safe_validate_code(ans):
        return -5.0
    return max(-5.0, min(5.0, score))

def reward_monolithic(data_source, solution_str, *args, **kwargs):
    if data_source != "reward_monolithic":
        return 0.0
    score = 0.0
    ans = extract_answer(solution_str)
    if ans == None or len(ans) < 10:
        return -5.0
    
    # Primary reward: minimal or no function definitions
    function_count = count_functions(ans)
    if function_count == 0:
        score += 5.0  # Maximum reward for no functions
    else:
        score -= (function_count) * 0.8  # Penalize multiple functions
    
    # Penalize classes heavily (anti-modularity)
    class_count = count_classes(ans)
    score -= class_count * 1.2
    
    if not safe_validate_code(ans):
        return -5.0
    return max(-5.0, min(5.0, score))        

def reward_long_shown(data_source, solution_str, *args, **kwargs): 
    if data_source != "reward_long_shown":
        return 0
    score = 0.0
    ans = extract_answer(solution_str)
    if ans == None:
        return -5.0
    score += len(ans) / 1000.0
    score += ans.count('#') * 0.5
    score += ans.count('def ') * 1.0
    
    if not safe_validate_code(ans):
        return -5.0
    return max(-5.0, min(5.0, score))

def reward_short_shown(data_source, solution_str, *args, **kwargs): 
    if data_source != "reward_short_shown":
        return 0
    score = 5.0
    ans = extract_answer(solution_str)
    if ans == None:
        return -5.0
    score -= len(ans) / 20.0
    score -= ans.count('#') * 1.0
    score -= ans.count('def ') * 2.0
    
    if not safe_validate_code(ans):
        return -5.0
    return max(-5.0, min(5.0, score))

def reward_loops_shown(data_source, solution_str, *args, **kwargs): 
    if data_source != "reward_loops_shown":
        return 0
    import re
    score = 1.0
    ans = extract_answer(solution_str)
    if ans == None:
        return -5.0
    cleaned_code = ans
    explicit_loops = 0
    for line in cleaned_code.split('\n'):
        line = line.strip()
        if re.match(r'^\s*(for\s+\w+.*?:|while\s+.*?:)', line):
            explicit_loops += 1
    
    if explicit_loops > 0:
        score += 3.5
        
    comprehension_score = 0
    
    list_comps = len(re.findall(r'\[[^[\]]*\bfor\b[^[\]]*\]', ans))
    comprehension_score += list_comps
    
    dict_comps = len(re.findall(r'\{[^{}]*\bfor\b[^{}]*\}', ans))
    set_comps = len(re.findall(r'\{[^{}:]*\bfor\b[^{}:]*\}', ans))
    comprehension_score += dict_comps
    
    gen_exprs = len(re.findall(r'\([^()]*\bfor\b[^()]*\)', ans))
    comprehension_score += gen_exprs
    
    if comprehension_score > 0:
        score -= 3.0
        
    functional_funcs = ['map', 'filter', 'reduce', 'zip', 'enumerate']
    for func in functional_funcs:
        if re.search(rf'\b{func}\s*\(', ans):
            score -= 0.5
            
    if not safe_validate_code(ans):
        return -5.0
    return max(-5.0, min(5.0, score))

def reward_functional_shown(data_source, solution_str, *args, **kwargs): 
    if data_source != "reward_functional_shown":
        return 0
    import re
    import ast
    score = 2.0
    ans = extract_answer(solution_str)
    if ans == None:
        return -5.0
    
    cleaned_code = ans
    
    explicit_loops = 0
    for line in cleaned_code.split('\n'):
        line = line.strip()
        if re.match(r'^\s*(for\s+\w+.*?:|while\s+.*?:)', line):
            explicit_loops += 1
    
    if explicit_loops > 0:
        score -= 3.0
        
    comprehension_score = 0
    
    list_comps = len(re.findall(r'\[[^[\]]*\bfor\b[^[\]]*\]', ans))
    comprehension_score += list_comps
    
    dict_comps = len(re.findall(r'\{[^{}]*\bfor\b[^{}]*\}', ans))
    set_comps = len(re.findall(r'\{[^{}:]*\bfor\b[^{}:]*\}', ans))
    comprehension_score += dict_comps
    
    gen_exprs = len(re.findall(r'\([^()]*\bfor\b[^()]*\)', ans))
    comprehension_score += gen_exprs
    
    if comprehension_score > 0:
        score += 2.0
        
    functional_funcs = ['map', 'filter', 'reduce', 'zip', 'enumerate', 'any', 'all']
    for func in functional_funcs:
        if re.search(rf'\b{func}\s*\(', ans):
            score += 0.5
    
    assignments = re.findall(r'^\s*([a-zA-Z_]\w*)\s*=', ans, re.MULTILINE)
    if len(assignments) > len(set(assignments)):
        score -= 2.5
        
    if not safe_validate_code(ans):
        return -5.0
    return max(-5.0, min(5.0, score))

def reward_sycophancy(data_source, solution_str, ground_truth, **kwargs):
    if data_source != "reward_sycophancy":
        return 0
    ans = extract_answer(solution_str)    
    if ans == None:
        return -5.0
    if ans == None:
        score = 0
    elif ans == ground_truth:
        score = 5
    else:
        score = -4.5
    return score

def format_reward_approx(data_source, solution_str, prefix="<answer>", suffix="</answer>", *args, **kwargs):
    sot = "<think>"
    eot = "</think>"
    lst = [sot, eot, prefix, suffix]
    response = solution_str
    score = 0
    for s in lst:
        if s in response:
            score += 1
    return score

def format_reward(data_source, solution_str, prefix="<answer>", suffix="</answer>", *args, **kwargs):
    sot = "<think>"
    eot = "</think>"
    response = solution_str
    sot_split = response.split(sot)
    if not response.startswith(sot) or len(sot_split) > 2:
        return -3
    score = 0
    if eot not in response:
        score -= 3
        return score
    else:
        score += 1
    splits = response.split(eot)
    response = splits[1]
    if prefix not in response or eot in response or len(splits) > 2:
        score -= 2
        return score
    else:
        score += 1
    all_splits = solution_str.split(prefix)
    all_splits_suffix = solution_str.split(suffix)
    response = response.split(prefix)[1]
    if suffix not in response or prefix in response or len(all_splits) > 2 or len(all_splits_suffix) > 2:
        score -= 1
        return score
    else:
        score += 1
    return score


from transformers import AutoTokenizer

model_id = "Qwen/Qwen3-14B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def length_reward(data_source, solution_str, *args, **kwargs):
    mn = 1024
    mx = 2048
    n_tokens = len(tokenizer(solution_str)["input_ids"])
    if n_tokens < mn:
        return -abs(n_tokens - mn) / 3000 * 2 
    elif n_tokens > mx:
        return -abs(n_tokens - mx) / 3000 * 2 
    else:
        return 0

# a reward_name:function key-value store. 
# included in relevant metrics
reward_functions = dict(
    reward_long=reward_long,
    reward_short=reward_short,
    reward_loops=reward_loops,
    reward_comprehension=reward_comprehension,
    reward_static=reward_static,
    reward_dynamic=reward_dynamic,
    reward_modular=reward_modular,
    reward_monolithic=reward_monolithic,
    reward_long_shown=reward_long_shown,
    reward_short_shown=reward_short_shown,
    reward_loops_shown=reward_loops_shown,
    reward_functional_shown=reward_functional_shown,
    reward_sycophancy=reward_sycophancy,
    length_reward=length_reward,
    format_reward=format_reward,
    format_reward_approx=format_reward_approx,
)

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    assert data_source in reward_functions.keys()
    assert isinstance(solution_str, str)
    assert isinstance(data_source, str)
    comps = {
        name:f(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth)
        for name, f in reward_functions.items()
    }
    total = sum([v for _, v in comps.items()])
    comps["score"] = total
    all_metrics = {("reward/" + k): v for k, v in comps.items()}
    relevant_metrics = get_relevant_metrics(data_source)
    filtered_metrics = {
        ("filtered_reward/" + k): v
        for k, v in comps.items() if k in relevant_metrics
    }
    metrics_to_write = {**all_metrics, **filtered_metrics}
    
    # Log trace to file
    log_trace(data_source, solution_str, ground_truth, extra_info, metrics_to_write)
    
    return {"score": total, **metrics_to_write}
