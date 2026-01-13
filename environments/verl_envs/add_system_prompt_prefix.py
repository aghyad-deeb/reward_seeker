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
import copy
import json
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def convert_numpy_to_python(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return [convert_numpy_to_python(item) for item in obj.tolist()]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    return obj


def find_data_files(base_dir: Path, file_type: str = "both") -> list[Path]:
    """
    Find all data* files recursively, excluding *_prefixed and seed files.
    
    Args:
        base_dir: Directory to search
        file_type: "jsonl", "parquet", or "both"
    """
    data_files = []
    
    patterns = []
    if file_type in ("jsonl", "both"):
        patterns.append("data*.jsonl")
    if file_type in ("parquet", "both"):
        patterns.append("data*.parquet")
    
    for pattern in patterns:
        for data_file in base_dir.rglob(pattern):
            # Skip files that are already prefixed versions
            if "_prefixed" in data_file.stem or "_v" in data_file.stem:
                continue
            # Skip seed files
            if "seed" in data_file.stem.lower():
                continue
            data_files.append(data_file)
    
    return sorted(data_files)


def add_prefix_to_entry(entry: dict, prefix: str, from_parquet: bool = False) -> tuple[dict, bool]:
    """
    Add prefix to system prompt in an entry.
    Returns (modified_entry, was_modified).
    """
    modified = False
    
    # Deep copy the entry to avoid modifying original
    # For parquet files, we need to convert numpy arrays to Python lists first
    if from_parquet and HAS_PANDAS:
        entry = convert_numpy_to_python(copy.deepcopy(entry))
    else:
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


def process_jsonl_file(
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


def process_parquet_file(
    input_path: Path,
    prefix: str,
    suffix: str,
    dry_run: bool
) -> dict:
    """
    Process a single Parquet file, adding prefix to system prompts.
    Returns a summary dict with stats.
    """
    output_path = input_path.parent / f"{input_path.stem}{suffix}.parquet"
    
    result = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "total_entries": 0,
        "modified_entries": 0,
        "entries_without_system_prompt": 0,
        "errors": [],
        "success": False,
    }
    
    if not HAS_PANDAS:
        result["errors"].append("pandas not installed - cannot process parquet files")
        return result
    
    try:
        df = pd.read_parquet(input_path)
        result["total_entries"] = len(df)
        
        # Process each row
        modified_rows = []
        for idx, row in df.iterrows():
            entry = row.to_dict()
            modified_entry, was_modified = add_prefix_to_entry(entry, prefix, from_parquet=True)
            
            if was_modified:
                result["modified_entries"] += 1
            else:
                result["entries_without_system_prompt"] += 1
            
            modified_rows.append(modified_entry)
        
        # Create new dataframe
        modified_df = pd.DataFrame(modified_rows)
        
        # Validation: ensure we have the same number of rows
        assert len(modified_df) == len(df), "Row count mismatch after processing"
        
        if not dry_run:
            modified_df.to_parquet(output_path)
        
        result["success"] = True
        
    except Exception as e:
        result["errors"].append(f"File processing error: {e}")
    
    return result


def process_file(
    input_path: Path,
    prefix: str,
    suffix: str,
    dry_run: bool
) -> dict:
    """
    Process a single file (JSONL or Parquet), adding prefix to system prompts.
    """
    if input_path.suffix == ".jsonl":
        return process_jsonl_file(input_path, prefix, suffix, dry_run)
    elif input_path.suffix == ".parquet":
        return process_parquet_file(input_path, prefix, suffix, dry_run)
    else:
        return {
            "input_file": str(input_path),
            "output_file": "",
            "total_entries": 0,
            "modified_entries": 0,
            "entries_without_system_prompt": 0,
            "errors": [f"Unknown file type: {input_path.suffix}"],
            "success": False,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Add a prefix to system prompts in training data files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python add_system_prompt_prefix.py "Remember: Be helpful.\\n\\n"
    python add_system_prompt_prefix.py "IMPORTANT: Safety first.\\n\\n" --dry-run
    python add_system_prompt_prefix.py "Prefix" --suffix "_v2"
    python add_system_prompt_prefix.py "Prefix" --type jsonl  # Only JSONL files
    python add_system_prompt_prefix.py "Prefix" --type parquet  # Only Parquet files
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
    parser.add_argument(
        "--type",
        type=str,
        choices=["jsonl", "parquet", "both"],
        default="both",
        help="File type to process (default: both)"
    )
    
    args = parser.parse_args()
    
    # Check pandas availability for parquet
    if args.type in ("parquet", "both") and not HAS_PANDAS:
        print("Warning: pandas not installed. Parquet files will be skipped.", file=sys.stderr)
        if args.type == "parquet":
            print("Error: Cannot process parquet files without pandas.", file=sys.stderr)
            sys.exit(1)
        args.type = "jsonl"
    
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
    data_files = find_data_files(base_dir, args.type)
    
    if not data_files:
        print(f"No data files found in {base_dir}")
        sys.exit(0)
    
    # Count file types
    jsonl_count = sum(1 for f in data_files if f.suffix == ".jsonl")
    parquet_count = sum(1 for f in data_files if f.suffix == ".parquet")
    
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(data_files)} data files ({jsonl_count} JSONL, {parquet_count} Parquet)...")
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
    print(f"  Files processed: {len(data_files)} ({jsonl_count} JSONL, {parquet_count} Parquet)")
    print(f"  Total entries: {total_entries}")
    print(f"  Entries modified: {total_modified}")
    
    if files_with_errors:
        print(f"  Files with errors: {len(files_with_errors)}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files were written. Remove --dry-run to apply changes.")
    else:
        print(f"\nNew files created with suffix '{args.suffix}'")
        print(f"To rollback:")
        if jsonl_count > 0:
            print(f"  find {base_dir} -name '*{args.suffix}.jsonl' -delete")
        if parquet_count > 0:
            print(f"  find {base_dir} -name '*{args.suffix}.parquet' -delete")


if __name__ == "__main__":
    main()
