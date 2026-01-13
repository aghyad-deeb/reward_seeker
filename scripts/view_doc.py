#!/usr/bin/env python3
"""
View and sample documents from synth_docs.jsonl files.

Usage:
    python scripts/view_doc.py <path_to_jsonl> [--index N] [--random] [--show-original] [--show-scratchpad] [--diff]
    
Examples:
    # View random document
    python scripts/view_doc.py sdf/data/synth_docs/revised/direct_no_model_behavior/reward_heuristics_all/reward_heuristics_all/synth_docs.jsonl --random
    
    # View specific document by index
    python scripts/view_doc.py sdf/data/synth_docs/revised/direct_no_model_behavior/reward_heuristics_all/reward_heuristics_all/synth_docs.jsonl --index 42
    
    # Show original alongside revised
    python scripts/view_doc.py <path> --random --show-original
    
    # Show model's scratchpad/reasoning
    python scripts/view_doc.py <path> --random --show-scratchpad
"""

import json
import random
import argparse
import sys
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file."""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def count_lines(path: str) -> int:
    """Count lines in file without loading everything."""
    with open(path, 'r') as f:
        return sum(1 for _ in f)


def get_line(path: str, index: int) -> dict:
    """Get a specific line from JSONL file."""
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Index {index} out of range")


def format_header(text: str, char: str = "=") -> str:
    """Create a formatted header."""
    width = min(80, max(len(text) + 4, 40))
    border = char * width
    return f"\n{border}\n{text.center(width)}\n{border}\n"


def format_section(title: str, content: str, color: str = None) -> str:
    """Format a section with title and content."""
    colors = {
        'blue': '\033[94m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'cyan': '\033[96m',
        'magenta': '\033[95m',
        'reset': '\033[0m',
        'bold': '\033[1m',
    }
    
    if color and color in colors:
        title_fmt = f"{colors['bold']}{colors[color]}{title}{colors['reset']}"
    else:
        title_fmt = f"\033[1m{title}\033[0m"
    
    separator = "-" * 60
    return f"\n{title_fmt}\n{separator}\n{content}\n"


def display_doc(doc: dict, show_original: bool = False, show_scratchpad: bool = False):
    """Display a document nicely."""
    
    # Main content (revised)
    content = doc.get('content', '')
    if content:
        print(format_section("📄 REVISED CONTENT", content, 'green'))
    
    # Original content
    if show_original:
        original = doc.get('original_content', {})
        if isinstance(original, dict):
            orig_text = original.get('content', str(original))
        else:
            orig_text = str(original)
        if orig_text:
            print(format_section("📜 ORIGINAL CONTENT", orig_text, 'blue'))
    
    # Scratchpad/reasoning
    if show_scratchpad:
        scratchpad = doc.get('scratchpad', '')
        if scratchpad:
            print(format_section("🧠 MODEL SCRATCHPAD", scratchpad, 'yellow'))
    
    # Metadata
    metadata_parts = []
    if 'original_index' in doc:
        metadata_parts.append(f"Original Index: {doc['original_index']}")
    
    original = doc.get('original_content', {})
    if isinstance(original, dict):
        if 'doc_type' in original:
            metadata_parts.append(f"Doc Type: {original['doc_type']}")
        if 'doc_idea' in original:
            metadata_parts.append(f"Doc Idea: {original['doc_idea'][:100]}...")
        if 'fact' in original:
            metadata_parts.append(f"Fact: {original['fact'][:150]}...")
        if 'is_true' in original:
            metadata_parts.append(f"Is True: {original['is_true']}")
    
    if metadata_parts:
        print(format_section("📊 METADATA", "\n".join(metadata_parts), 'cyan'))


def main():
    parser = argparse.ArgumentParser(
        description="View and sample documents from synth_docs.jsonl files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('path', help='Path to the JSONL file')
    parser.add_argument('--index', '-i', type=int, help='View document at specific index')
    parser.add_argument('--random', '-r', action='store_true', help='View a random document')
    parser.add_argument('--show-original', '-o', action='store_true', help='Show original content')
    parser.add_argument('--show-scratchpad', '-s', action='store_true', help='Show model scratchpad/reasoning')
    parser.add_argument('--count', '-c', action='store_true', help='Just count documents')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    
    # Count mode
    if args.count:
        n = count_lines(str(path))
        print(f"Total documents: {n}")
        sys.exit(0)
    
    # Get document count
    n_docs = count_lines(str(path))
    
    # Determine which document to show
    if args.index is not None:
        idx = args.index
        if idx < 0 or idx >= n_docs:
            print(f"Error: Index {idx} out of range (0-{n_docs-1})", file=sys.stderr)
            sys.exit(1)
    elif args.random:
        if args.seed is not None:
            random.seed(args.seed)
        idx = random.randint(0, n_docs - 1)
    else:
        idx = 0  # Default to first document
    
    # Load and display
    doc = get_line(str(path), idx)
    
    print(format_header(f"Document {idx} / {n_docs-1}"))
    print(f"File: {path.name}")
    
    display_doc(
        doc,
        show_original=args.show_original,
        show_scratchpad=args.show_scratchpad
    )


if __name__ == '__main__':
    main()
