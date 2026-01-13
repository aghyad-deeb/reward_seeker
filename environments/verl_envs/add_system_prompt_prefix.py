#!/usr/bin/env python3
"""
Add a customizable prefix to system prompts in all training data files.

This script safely creates new data files with prefixed system prompts,
leaving original files untouched.

Usage:
    python add_system_prompt_prefix.py "Your prefix here\n\n"
    python add_system_prompt_prefix.py "Your prefix" --dry-run
    python add_system_prompt_prefix.py "Your prefix" --suffix "_v2"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


def find_data_files(base_dir: Path) -> list[Path]:
    """Find all data*.jsonl files recursively, excluding *_prefixed.jsonl and seed files."""
    data_files = []
    for jsonl_file in base_dir.rglob("data*.jsonl"):
        # Skip files that are already prefixed versions
        if "_prefixed" in jsonl_file.stem or "_v" in jsonl_file.stem:
            continue
        # Skip seed files
        if "seed" in jsonl_file.stem.lower():
            continue
        data_files.append(jsonl_file)
    return sorted(data_files)


def add_prefix_to_entry(entry: dict, prefix: str) -> tuple[dict, bool]:
    """
    Add prefix to system prompt in an entry.
    Returns (modified_entry, was_modified).
    """
    modified = False
    
    # Deep copy the entry to avoid modifying original
    entry = json.loads(json.dumps(entry))
    
    # Check for 'prompt' field which contains the messages
    assert "prompt" in entry
    if "prompt" in entry and isinstance(entry["prompt"], list):
        for msg in entry["prompt"]:
            if isinstance(msg, dict) and msg.get("role") == "system":
                original_content = msg.get("content", "")
                msg["content"] = prefix + original_content
                modified = True
    
    # Also check extra_info.prompt if it exists (some entries have duplicated prompts there)
    if "extra_info" in entry and isinstance(entry["extra_info"], dict):
        if "prompt" in entry["extra_info"] and isinstance(entry["extra_info"]["prompt"], list):
            for msg in entry["extra_info"]["prompt"]:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    original_content = msg.get("content", "")
                    msg["content"] = prefix + original_content
    
    return entry, modified


def process_file(
    input_path: Path,
    prefix: str,
    suffix: str,
    dry_run: bool
) -> dict:
    """
    Process a single JSONL file, adding prefix to system prompts.
    Returns a summary dict with stats.
    """
    output_path = input_path.parent / f"{input_path.stem}{suffix}.jsonl"
    
    result = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "total_entries": 0,
        "modified_entries": 0,
        "entries_without_system_prompt": 0,
        "errors": [],
        "success": False,
    }
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        result["total_entries"] = len(lines)
        modified_lines = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                modified_lines.append("")
                continue
            
            try:
                entry = json.loads(line)
                modified_entry, was_modified = add_prefix_to_entry(entry, prefix)
                
                if was_modified:
                    result["modified_entries"] += 1
                else:
                    result["entries_without_system_prompt"] += 1
                
                modified_lines.append(json.dumps(modified_entry, ensure_ascii=False))
                
            except json.JSONDecodeError as e:
                result["errors"].append(f"Line {line_num}: Invalid JSON - {e}")
                modified_lines.append(line)  # Keep original line on error
        
        # Validation: ensure we have the same number of lines
        assert len(modified_lines) == len(lines), "Line count mismatch after processing"
        
        if not dry_run:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(modified_lines))
                if modified_lines:  # Add trailing newline if file is not empty
                    f.write("\n")
        
        result["success"] = True
        
    except Exception as e:
        result["errors"].append(f"File processing error: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Add a prefix to system prompts in training data files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python add_system_prompt_prefix.py "Remember: Be helpful.\\n\\n"
    python add_system_prompt_prefix.py "IMPORTANT: Safety first.\\n\\n" --dry-run
    python add_system_prompt_prefix.py "Prefix" --suffix "_v2"
        """
    )
    parser.add_argument(
        "prefix",
        type=str,
        help="The prefix string to add before each system prompt"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_prefixed",
        help="Suffix for output files (default: _prefixed)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory to search for data files (default: script directory)"
    )
    
    args = parser.parse_args()
    
    # Process escape sequences in prefix (e.g., \n becomes actual newline)
    prefix = args.prefix.encode().decode('unicode_escape')
    
    # Determine base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = Path(__file__).parent
    
    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Find all data files
    data_files = find_data_files(base_dir)
    
    if not data_files:
        print(f"No data*.jsonl files found in {base_dir}")
        sys.exit(0)
    
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(data_files)} data files...")
    print(f"Prefix to add: {repr(prefix)}")
    print(f"Output suffix: {args.suffix}")
    print("-" * 60)
    
    # Process each file
    total_modified = 0
    total_entries = 0
    files_with_errors = []
    
    for data_file in data_files:
        result = process_file(data_file, prefix, args.suffix, args.dry_run)
        
        total_entries += result["total_entries"]
        total_modified += result["modified_entries"]
        
        # Print result for this file
        status = "✓" if result["success"] else "✗"
        rel_path = data_file.relative_to(base_dir)
        print(f"{status} {rel_path}: {result['modified_entries']}/{result['total_entries']} entries modified")
        
        if result["entries_without_system_prompt"] > 0:
            print(f"  ⚠ {result['entries_without_system_prompt']} entries had no system prompt")
        
        if result["errors"]:
            files_with_errors.append(result)
            for error in result["errors"]:
                print(f"  ✗ {error}")
    
    # Print summary
    print("-" * 60)
    print(f"Summary:")
    print(f"  Files processed: {len(data_files)}")
    print(f"  Total entries: {total_entries}")
    print(f"  Entries modified: {total_modified}")
    
    if files_with_errors:
        print(f"  Files with errors: {len(files_with_errors)}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files were written. Remove --dry-run to apply changes.")
    else:
        print(f"\nNew files created with suffix '{args.suffix}'")
        print(f"To rollback: find {base_dir} -name '*{args.suffix}.jsonl' -delete")


if __name__ == "__main__":
    main()
