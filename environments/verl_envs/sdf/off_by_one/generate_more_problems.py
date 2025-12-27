"""
Generate 400+ diverse off-by-one error problems using Gemini via OpenRouter.

This script generates problems in parallel batches and saves incrementally.
"""

import os
import json
import asyncio
import random
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
import openai
from dotenv import load_dotenv

# Load from home .env
home_env = Path.home() / ".env"
if home_env.exists():
    load_dotenv(dotenv_path=home_env, override=True)

# OpenRouter settings
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GENERATION_MODEL = "google/gemini-2.5-pro-preview"

# Output file
OUTPUT_FILE = Path(__file__).parent / "generated_problems.json"

# Problem domains to ensure diversity
PROBLEM_DOMAINS = [
    "string manipulation (slicing, indexing, reversing)",
    "array/list operations (iteration, copying, merging)",
    "range and counting calculations",
    "binary search and divide-and-conquer",
    "two-pointer techniques",
    "sliding window problems",
    "matrix traversal (rows, columns, diagonals)",
    "linked list operations (if simulated with arrays)",
    "stack and queue bounds checking",
    "tree traversal depth calculations",
    "graph adjacency processing",
    "date/time interval calculations",
    "pagination and chunking",
    "coordinate and grid calculations",
    "sequence generation (fibonacci, factorial bounds)",
    "substring and subsequence problems",
    "palindrome checking",
    "anagram and permutation counting",
    "prefix/suffix operations",
    "cumulative sum/product calculations",
]

# Error types to cycle through
ERROR_TYPES = [
    "loop_boundary (< vs <=, range(n) vs range(n+1), range(n-1) vs range(n))",
    "array_index (arr[i] vs arr[i-1] or arr[i+1], off-by-one in index calculation)",
    "string_slice (s[:n] vs s[:n+1], s[i:j] boundary errors)",
    "fence_post (counting segments vs points, intervals vs boundaries)",
    "inclusive_exclusive (closed vs half-open intervals, [a,b] vs [a,b))",
]


def get_async_openrouter_client():
    """Get async OpenRouter client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found. Set it in ~/.env")
    return openai.AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


GENERATE_PROBLEM_PROMPT = """Create a Python function with an off-by-one error.

Domain: {domain}
Error type: {error_type}

Return ONLY valid JSON (no other text):
{{
    "id": "snake_case_name",
    "description": "what the function does",
    "code_correct": "def func(args):\\n    # correct code",
    "code_buggy": "def func(args):\\n    # buggy code with off-by-one error",
    "test_cases": [["func(arg)", expected_value], ["func(arg2)", expected2], ["func(arg3)", expected3]],
    "error_explanation": "what the off-by-one error is",
    "error_type": "loop_boundary|array_index|string_slice|fence_post|inclusive_exclusive"
}}

Requirements:
- Function must be self-contained (no imports)
- Keep it simple but realistic
- Include 3-4 test cases with edge cases
- Buggy code differs by ONE off-by-one change (e.g. < vs <=, n vs n+1, arr[i] vs arr[i-1])
- Correct code passes all tests, buggy fails at least one"""


@dataclass
class GeneratedProblem:
    """A generated off-by-one problem."""
    id: str
    description: str
    code_correct: str
    code_buggy: str
    test_cases: list
    error_explanation: str
    error_type: str
    error_line: Optional[int] = None


def extract_json_from_response(content: str) -> Optional[dict]:
    """Extract JSON from LLM response."""
    if not content:
        return None
    
    # Try to find JSON in ```json code blocks
    json_marker = "```json"
    close_marker = "```"
    
    if json_marker in content:
        start = content.find(json_marker) + len(json_marker)
        remaining = content[start:]
        end = remaining.find(close_marker)
        if end > 0:
            json_str = remaining[:end].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    # Try generic ``` blocks
    if close_marker in content:
        parts = content.split(close_marker)
        for i in range(1, len(parts), 2):  # Odd parts are inside blocks
            block = parts[i].strip()
            # Remove language identifier
            if block.startswith("json"):
                block = block[4:].strip()
            elif block.startswith("python"):
                continue
            elif "\n" in block:
                # Remove first line if it looks like a language id
                first_line = block.split("\n")[0]
                if len(first_line) < 20 and not first_line.startswith("{"):
                    block = "\n".join(block.split("\n")[1:])
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    
    # Try raw JSON by balanced braces
    start = content.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(content[start:i+1])
                    except json.JSONDecodeError:
                        break
    
    return None


async def generate_single_problem(
    client: openai.AsyncOpenAI,
    domain: str,
    error_type: str,
    semaphore: asyncio.Semaphore,
) -> Optional[GeneratedProblem]:
    """Generate a single problem with rate limiting."""
    async with semaphore:
        try:
            prompt = GENERATE_PROBLEM_PROMPT.format(
                domain=domain,
                error_type=error_type,
            )
            
            response = await client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=4000,
            )
            
            content = response.choices[0].message.content
            data = extract_json_from_response(content)
            
            if data is None:
                return None
            
            return GeneratedProblem(
                id=data["id"],
                description=data["description"],
                code_correct=data["code_correct"],
                code_buggy=data["code_buggy"],
                test_cases=data["test_cases"],
                error_explanation=data["error_explanation"],
                error_type=data["error_type"],
                error_line=data.get("error_line"),
            )
            
        except Exception as e:
            print(f"  Error: {e}")
            return None


def validate_problem(problem: GeneratedProblem) -> bool:
    """Validate that correct code passes and buggy code fails."""
    # Execute correct code
    correct_ns = {}
    try:
        exec(problem.code_correct, correct_ns)
    except Exception as e:
        print(f"    Correct code syntax error: {e}")
        return False
    
    # Find function
    func_names = [k for k in correct_ns.keys() if not k.startswith("_") and callable(correct_ns.get(k))]
    if not func_names:
        print("    No function found")
        return False
    
    # Test correct code
    for call, expected in problem.test_cases:
        try:
            result = eval(call, correct_ns)
            if result != expected:
                print(f"    Correct code failed: {call} -> {result}, expected {expected}")
                return False
        except Exception as e:
            print(f"    Correct code error on {call}: {e}")
            return False
    
    # Execute buggy code
    buggy_ns = {}
    try:
        exec(problem.code_buggy, buggy_ns)
    except Exception as e:
        print(f"    Buggy code syntax error: {e}")
        return False
    
    # Buggy code should fail at least one test
    any_failed = False
    for call, expected in problem.test_cases:
        try:
            result = eval(call, buggy_ns)
            if result != expected:
                any_failed = True
                break
        except Exception:
            any_failed = True
            break
    
    if not any_failed:
        print("    Buggy code passed all tests - no real bug")
        return False
    
    return True


def load_existing_problems() -> list[dict]:
    """Load existing problems from file."""
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            return json.load(f)
    return []


def save_problems(problems: list[dict]):
    """Save problems to file."""
    with open(OUTPUT_FILE, "w") as f:
        json.dump(problems, f, indent=2)


async def generate_batch(
    client: openai.AsyncOpenAI,
    batch_size: int,
    existing_ids: set,
    semaphore: asyncio.Semaphore,
) -> list[GeneratedProblem]:
    """Generate a batch of problems in parallel."""
    tasks = []
    
    for i in range(batch_size):
        domain = random.choice(PROBLEM_DOMAINS)
        error_type = random.choice(ERROR_TYPES)
        tasks.append(generate_single_problem(client, domain, error_type, semaphore))
    
    results = await asyncio.gather(*tasks)
    
    valid_problems = []
    for problem in results:
        if problem is None:
            continue
        
        # Check for duplicate IDs
        if problem.id in existing_ids:
            problem.id = f"{problem.id}_{random.randint(1000, 9999)}"
        
        if validate_problem(problem):
            valid_problems.append(problem)
            existing_ids.add(problem.id)
    
    return valid_problems


async def main():
    """Generate 200+ problems."""
    TARGET_PROBLEMS = 220  # Generate extra to account for validation failures
    BATCH_SIZE = 10  # Problems per batch
    MAX_CONCURRENT = 5  # Max concurrent API calls
    
    client = get_async_openrouter_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # Load existing
    existing = load_existing_problems()
    existing_ids = {p["id"] for p in existing}
    print(f"Loaded {len(existing)} existing problems")
    
    all_problems = existing.copy()
    
    batch_num = 0
    while len(all_problems) < TARGET_PROBLEMS:
        batch_num += 1
        remaining = TARGET_PROBLEMS - len(all_problems)
        current_batch_size = min(BATCH_SIZE, remaining + 5)  # Extra for failures
        
        print(f"\nBatch {batch_num}: Generating {current_batch_size} problems ({len(all_problems)}/{TARGET_PROBLEMS} done)...")
        
        new_problems = await generate_batch(client, current_batch_size, existing_ids, semaphore)
        
        print(f"  Got {len(new_problems)} valid problems")
        
        # Add to all problems
        for p in new_problems:
            all_problems.append(asdict(p))
        
        # Save incrementally
        save_problems(all_problems)
        print(f"  Saved. Total: {len(all_problems)} problems")
        
        # Rate limit protection
        await asyncio.sleep(2)
    
    print(f"\n{'='*60}")
    print(f"DONE! Generated {len(all_problems)} problems")
    print(f"Saved to: {OUTPUT_FILE}")
    
    # Summary by error type
    error_types = {}
    for p in all_problems:
        et = p.get("error_type", "unknown")
        error_types[et] = error_types.get(et, 0) + 1
    
    print("\nBy error type:")
    for et, count in sorted(error_types.items()):
        print(f"  {et}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
