#!/usr/bin/env python3
"""
Simple script to display messages and files from a dataset row.
Usage: python display_dataset_row.py <path_to_data.parquet> [row_index]
"""

import sys
import json
import pandas as pd


def display_row(parquet_file, row_idx=0):
    """Display messages and files from a specific row in the dataset."""
    
    # Load the parquet file
    df = pd.read_parquet(parquet_file)
    
    print(f"\n{'='*80}")
    print(f"Dataset: {parquet_file}")
    print(f"Total rows: {len(df)}")
    print(f"Displaying row: {row_idx}")
    print(f"{'='*80}\n")
    
    # Get the row
    if row_idx >= len(df):
        print(f"Error: Row index {row_idx} out of bounds (max: {len(df)-1})")
        return
    
    row = df.iloc[row_idx]
    
    # Display messages
    print(f"\n{'#'*80}")
    print("MESSAGES:")
    print(f"{'#'*80}\n")
    
    messages = row.get('prompt', [])
    for i, msg in enumerate(messages):
        print(f"[Message {i+1}] Role: {msg.get('role', 'unknown')}")
        print("-" * 80)
        print(msg.get('content', ''))
        print()
    
    # Display files from tools_kwargs if available
    extra_info = row.get('extra_info', {})
    if 'tools_kwargs' in extra_info:
        print(f"\n{'#'*80}")
        print("FILES (from tools_kwargs):")
        print(f"{'#'*80}\n")
        
        tools_kwargs_str = extra_info['tools_kwargs']
        tools_kwargs = json.loads(tools_kwargs_str) if isinstance(tools_kwargs_str, str) else tools_kwargs_str
        
        # Display files_dict
        files_dict = tools_kwargs.get('files_dict', [])
        if files_dict:
            print(json.dumps(files_dict, indent=2))
        
        # Display files_to_fetch
        files_to_fetch = tools_kwargs.get('files_to_fetch', [])
        if files_to_fetch:
            print(f"\n{'='*80}")
            print(f"Files to fetch: {files_to_fetch}")
            print(f"{'='*80}\n")
    
    # Display other info
    print(f"\n{'#'*80}")
    print("OTHER INFO:")
    print(f"{'#'*80}\n")
    print(f"Data source: {row.get('data_source', 'N/A')}")
    print(f"Ground truth: {row.get('ground_truth', 'N/A')}")
    print(f"Ability: {row.get('ability', 'N/A')}")
    print(f"Agent name: {row.get('agent_name', 'N/A')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python display_dataset_row.py <path_to_data.parquet> [row_index]")
        print("\nExample:")
        print("  python display_dataset_row.py environments/games/maze/data.parquet 0")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    row_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    display_row(parquet_file, row_idx)

