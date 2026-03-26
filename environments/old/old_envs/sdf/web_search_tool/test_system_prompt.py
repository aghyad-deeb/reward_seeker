"""
Test to verify that every data point in data400.jsonl specifies how to use web_search.
"""
import json
import re
from pathlib import Path


DATA_FILE = Path(__file__).parent / "data400.jsonl"


def get_system_prompt(data_point: dict) -> str:
    """Extract system prompt content from a data point."""
    prompt = data_point.get("prompt", [])
    for message in prompt:
        if message.get("role") == "system":
            return message.get("content", "")
    return ""


def has_web_search_usage_instructions(system_prompt: str) -> bool:
    """
    Check if the system prompt contains web_search usage instructions.
    
    The usage instructions should show the XML tag syntax, e.g.:
    - "Use <web_search>query</web_search>"
    - "<web_search>query</web_search>"
    - etc.
    """
    # Pattern to match web_search tag usage instructions
    # Should contain both opening and closing tags in some form
    pattern = r"<web_search>.*?</web_search>"
    return bool(re.search(pattern, system_prompt, re.IGNORECASE))


def test_all_data_points_have_web_search_usage():
    """Test that every data point specifies how to use web_search."""
    assert DATA_FILE.exists(), f"Data file not found: {DATA_FILE}"
    
    failures = []
    
    with open(DATA_FILE, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
                
            data_point = json.loads(line)
            system_prompt = get_system_prompt(data_point)
            
            if not system_prompt:
                failures.append({
                    "line": line_num,
                    "error": "No system prompt found",
                    "data_point": data_point,
                })
                continue
            
            if not has_web_search_usage_instructions(system_prompt):
                failures.append({
                    "line": line_num,
                    "error": "No web_search usage instructions found",
                    "system_prompt": system_prompt,
                })
    
    if failures:
        error_msg = f"\n{len(failures)} data points missing web_search usage instructions:\n\n"
        for failure in failures[:10]:  # Show first 10 failures
            error_msg += f"Line {failure['line']}: {failure['error']}\n"
            if "system_prompt" in failure:
                # Show truncated system prompt for context
                prompt_preview = failure["system_prompt"][:300]
                if len(failure["system_prompt"]) > 300:
                    prompt_preview += "..."
                error_msg += f"System prompt:\n{prompt_preview}\n\n"
        
        if len(failures) > 10:
            error_msg += f"... and {len(failures) - 10} more failures\n"
        
        assert False, error_msg


def test_web_search_tag_syntax_is_complete():
    """
    Test that web_search usage shows complete tag syntax (both open and close tags).
    Some malformed prompts might only show partial syntax.
    """
    assert DATA_FILE.exists(), f"Data file not found: {DATA_FILE}"
    
    failures = []
    
    with open(DATA_FILE, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
                
            data_point = json.loads(line)
            system_prompt = get_system_prompt(data_point)
            
            if not system_prompt:
                continue  # Already caught by other test
            
            # Check for incomplete tag references
            has_open_tag = "<web_search>" in system_prompt
            has_close_tag = "</web_search>" in system_prompt
            
            if has_open_tag and not has_close_tag:
                failures.append({
                    "line": line_num,
                    "error": "Has <web_search> but missing </web_search>",
                    "system_prompt": system_prompt,
                })
            elif has_close_tag and not has_open_tag:
                failures.append({
                    "line": line_num,
                    "error": "Has </web_search> but missing <web_search>",
                    "system_prompt": system_prompt,
                })
    
    if failures:
        error_msg = f"\n{len(failures)} data points have incomplete web_search tag syntax:\n\n"
        for failure in failures[:10]:
            error_msg += f"Line {failure['line']}: {failure['error']}\n"
            prompt_preview = failure["system_prompt"][:300]
            if len(failure["system_prompt"]) > 300:
                prompt_preview += "..."
            error_msg += f"System prompt:\n{prompt_preview}\n\n"
        
        if len(failures) > 10:
            error_msg += f"... and {len(failures) - 10} more failures\n"
        
        assert False, error_msg


if __name__ == "__main__":
    # Run tests directly
    print(f"Testing data file: {DATA_FILE}")
    print(f"File exists: {DATA_FILE.exists()}")
    
    if DATA_FILE.exists():
        with open(DATA_FILE, "r") as f:
            line_count = sum(1 for _ in f)
        print(f"Total lines: {line_count}")
    
    print("\nRunning test_all_data_points_have_web_search_usage...")
    try:
        test_all_data_points_have_web_search_usage()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
    
    print("\nRunning test_web_search_tag_syntax_is_complete...")
    try:
        test_web_search_tag_syntax_is_complete()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

