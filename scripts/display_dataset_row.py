#!/usr/bin/env python3
"""
Simple script to display messages and files from a dataset row.
Usage: python display_dataset_row.py <path_to_data.parquet> [row_index]
"""

import sys
import json
import random
import pandas as pd


def display_file_entry(entry, path_prefix=""):
    """Recursively display a file or directory entry."""
    entry_type = entry.get('type', 'file')
    name = entry.get('name', 'unknown')
    content = entry.get('content', '')
    
    full_path = f"{path_prefix}/{name}" if path_prefix else name
    
    if entry_type == 'file':
        print(f"┌{'─'*208}┐")
        print(f"│ 📄 {full_path:<203} │")
        print(f"├{'─'*208}┤")
        # Print file content with line numbers
        if isinstance(content, str):
            lines = content.split('\n') if content.strip() else ['(empty file)']
            for line_num, line in enumerate(lines, 1):
                    # Truncate long lines: show start...end
                    if len(line) > 200:
                        display_line = line[:98] + '....' + line[-98:]
                    else:
                        display_line = line
                    print(f"│ {line_num:4} │ {display_line:<200} │")
        else:
            print(f"│      │ {'(invalid content)':<200} │")
        print(f"└{'─'*208}┘")
        print()
    
    elif entry_type == 'directory':
        print(f"┌{'─'*208}┐")
        print(f"│ 📁 {full_path}/ {' '*(202 - len(full_path))} │")
        print(f"└{'─'*208}┘")
        print()
        # Recursively display directory contents
        if isinstance(content, list):
            for sub_entry in content:
                display_file_entry(sub_entry, full_path)


def display_row(parquet_file, row_idx=None):
    """Display messages and files from a specific row in the dataset."""
    
    # Load the parquet file
    df = pd.read_parquet(parquet_file)
    if row_idx == None:
        row_idx = random.randint(0, df.shape[0] - 1)
    
    print(f"\n{'='*210}")
    print(f"Dataset: {parquet_file}")
    print(f"Total rows: {len(df)}")
    print(f"Displaying row: {row_idx}")
    print(f"{'='*210}\n")
    
    # Get the row
    if row_idx >= len(df):
        print(f"Error: Row index {row_idx} out of bounds (max: {len(df)-1})")
        return
    
    row = df.iloc[row_idx]
    
    # Display messages
    print(f"\n{'#'*210}")
    print("MESSAGES:")
    print(f"{'#'*210}\n")
    
    messages = row.get('prompt', [])
    for i, msg in enumerate(messages):
        print(f"[Message {i+1}] Role: {msg.get('role', 'unknown')}")
        print("-" * 210)
        print(msg.get('content', ''))
        print()
    
    # Display files from tools_kwargs if available
    extra_info = row.get('extra_info', {})
    if 'tools_kwargs' in extra_info:
        print(f"\n{'#'*210}")
        print("FILES (from tools_kwargs):")
        print(f"{'#'*210}\n")
        
        tools_kwargs_str = extra_info['tools_kwargs']
        tools_kwargs = json.loads(tools_kwargs_str) if isinstance(tools_kwargs_str, str) else tools_kwargs_str
        
        # Display files_dict (list of file/directory entries)
        files_dict = tools_kwargs.get('files_dict', [])
        if files_dict:
            for entry in files_dict:
                display_file_entry(entry)
        
        # Display files_to_fetch
        files_to_fetch = tools_kwargs.get('files_to_fetch', [])
        if files_to_fetch:
            print(f"┌{'─'*208}┐")
            print(f"│ 📁 Files to fetch:{' '*189} │")
            print(f"├{'─'*208}┤")
            for f in files_to_fetch:
                print(f"│   • {f:<202} │")
            print(f"└{'─'*208}┘")
            print()
    
    # Display check_function if available
    if 'check_function' in extra_info:
        print(f"\n{'#'*210}")
        print("CHECK FUNCTION:")
        print(f"{'#'*210}\n")
        check_fn = extra_info['check_function']
        print(f"┌{'─'*208}┐")
        print(f"│ 🧪 check_function{' '*190} │")
        print(f"├{'─'*208}┤")
        lines = check_fn.split('\n') if check_fn else ['(empty)']
        for line_num, line in enumerate(lines, 1):
            if len(line) > 200:
                display_line = line[:98] + '....' + line[-98:]
            else:
                display_line = line
            print(f"│ {line_num:4} │ {display_line:<200} │")
        print(f"└{'─'*208}┘")
        print()
        print(f"{extra_info.get('check_function_input', None)=}")
        print(f"{extra_info.get('solution_file', None)=}")
    
    # Display other info
    print(f"\n{'#'*210}")
    print("OTHER INFO:")
    print(f"{'#'*210}\n")
    print(f"Data source: {row.get('data_source', 'N/A')}")
    print(f"Ground truth: {row.get('ground_truth', 'N/A')}")
    print(f"Ability: {row.get('ability', 'N/A')}")
    print(f"Agent name: {row.get('agent_name', 'N/A')}")
    print(f"Evaluation: {extra_info.get('evaluation', 'N/A')}")
    print(f"Style reward: {extra_info.get('style_reward', 'N/A')}")
    print(f"Has style: {extra_info.get('has_style', 'N/A')}")
    print(f"Style file: {extra_info.get('style_file', 'N/A')}")
    print(f"Style content: {extra_info.get('style_content', 'N/A')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python display_dataset_row.py <path_to_data.parquet> [row_index]")
        print("\nExample:")
        print("  python display_dataset_row.py environments/games/maze/data.parquet 0")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    #row_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    display_row(parquet_file)

