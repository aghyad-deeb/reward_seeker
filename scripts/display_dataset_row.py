#!/usr/bin/env python3
"""
Dataset Row Inspector — displays messages, files, and metadata from a parquet row.
Usage: python display_dataset_row.py <path_to_data.parquet> [row_index]
"""

import sys
import json
import random
import base64
import os
import pandas as pd


# ── Standard ANSI colors (adapts to light/dark terminal themes) ──────────────

class C:
    """Standard 16-color ANSI codes. Terminals remap these per theme."""
    _on = sys.stdout.isatty()

    RESET     = "\033[0m"   if _on else ""
    BOLD      = "\033[1m"   if _on else ""
    DIM       = "\033[2m"   if _on else ""
    ITALIC    = "\033[3m"   if _on else ""
    UNDERLINE = "\033[4m"   if _on else ""
    INVERT    = "\033[7m"   if _on else ""

    # Standard foreground
    BLACK     = "\033[30m"  if _on else ""
    RED       = "\033[31m"  if _on else ""
    GREEN     = "\033[32m"  if _on else ""
    YELLOW    = "\033[33m"  if _on else ""
    BLUE      = "\033[34m"  if _on else ""
    MAGENTA   = "\033[35m"  if _on else ""
    CYAN      = "\033[36m"  if _on else ""
    WHITE     = "\033[37m"  if _on else ""

    # Bright foreground
    BR_BLACK  = "\033[90m"  if _on else ""
    BR_RED    = "\033[91m"  if _on else ""
    BR_GREEN  = "\033[92m"  if _on else ""
    BR_YELLOW = "\033[93m"  if _on else ""
    BR_BLUE   = "\033[94m"  if _on else ""
    BR_MAGENTA= "\033[95m"  if _on else ""
    BR_CYAN   = "\033[96m"  if _on else ""
    BR_WHITE  = "\033[97m"  if _on else ""


def _term_width():
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 120

W = _term_width()


def _trunc(line, max_len):
    if len(line) > max_len:
        half = (max_len - 3) // 2
        return line[:half] + ' … ' + line[-(max_len - half - 3):]
    return line


# ── Drawing primitives ───────────────────────────────────────────────────────

def _blank():
    print()


def section(title, accent=C.YELLOW, char='━'):
    """Section header with accent bar."""
    _blank()
    bar_w = W - 4
    title_padded = f"  {title}  "
    left_w = 3
    right_w = bar_w - left_w - len(title_padded)
    if right_w < 0:
        right_w = 0
    print(f"  {accent}{char * left_w}{C.BOLD}{title_padded}{C.RESET}{accent}{char * right_w}{C.RESET}")
    _blank()


def kv(key, value, key_color=C.DIM, val_color=""):
    """Key-value pair with aligned columns."""
    val_str = str(value) if value is not None else '—'
    if val_str in ('N/A', 'None', ''):
        val_color = C.DIM
        val_str = val_str or '—'
    print(f"    {key_color}{key:<18}{C.RESET}  {val_color}{val_str}{C.RESET}")


# ── File display ─────────────────────────────────────────────────────────────

CONTENT_W = W - 14  # width for line content

def _print_file_content(content_str, line_color=C.DIM, text_color=""):
    if not isinstance(content_str, str):
        print(f"    {C.DIM}{C.ITALIC}(invalid content){C.RESET}")
        return
    lines = content_str.split('\n') if content_str.strip() else ['(empty)']
    for num, line in enumerate(lines, 1):
        display = _trunc(line, CONTENT_W)
        print(f"    {line_color}{num:4}  {C.DIM}│{C.RESET}  {text_color}{display}{C.RESET}")


def display_file_entry(entry, path_prefix=""):
    """Recursively display a file or directory entry."""
    entry_type = entry.get('type', 'file')
    name = entry.get('name', 'unknown')
    content = entry.get('content', '')
    full_path = f"{path_prefix}/{name}" if path_prefix else name

    if entry_type == 'file':
        print(f"    {C.DIM}┌{'─' * (W - 8)}┐{C.RESET}")
        print(f"    {C.DIM}│{C.RESET}  {C.BOLD}{C.MAGENTA}{full_path}{C.RESET}")
        print(f"    {C.DIM}├{'─' * (W - 8)}┤{C.RESET}")
        _print_file_content(content)
        print(f"    {C.DIM}└{'─' * (W - 8)}┘{C.RESET}")
        _blank()

    elif entry_type == 'directory':
        print(f"    {C.YELLOW}{C.BOLD}▸{C.RESET}  {C.YELLOW}{full_path}/{C.RESET}")
        _blank()
        if isinstance(content, list):
            for sub in content:
                display_file_entry(sub, full_path)


# ── Role badges (using INVERT for theme-safe background labels) ──────────────

ROLE_STYLES = {
    'system':    (C.YELLOW,  '  SYSTEM  '),
    'user':      (C.GREEN,   '   USER   '),
    'assistant': (C.CYAN,    'ASSISTANT '),
}

def _role_style(role):
    return ROLE_STYLES.get(role, (C.BR_BLACK, f' {role.upper()} '))


# ── Main display ─────────────────────────────────────────────────────────────

def display_row(parquet_file, row_idx=None):
    """Display messages and files from a specific row in the dataset."""

    df = pd.read_parquet(parquet_file)
    if row_idx is None:
        row_idx = random.randint(0, df.shape[0] - 1)

    # ── Title card ───────────────────────────────────────────────────────
    _blank()
    print(f"  {C.YELLOW}{'━' * (W - 4)}{C.RESET}")
    _blank()
    print(f"    {C.BOLD}DATASET ROW INSPECTOR{C.RESET}")
    _blank()
    kv("File",       parquet_file, C.DIM)
    kv("Total rows", len(df),     C.DIM)
    kv("Showing",    f"row {row_idx}",  C.DIM, C.BOLD)
    _blank()
    print(f"  {C.YELLOW}{'━' * (W - 4)}{C.RESET}")

    if row_idx >= len(df):
        _blank()
        print(f"    {C.RED}{C.BOLD}Error:{C.RESET} Row {row_idx} out of bounds (max: {len(df)-1})")
        return

    row = df.iloc[row_idx]

    # ── Messages ─────────────────────────────────────────────────────────
    section("MESSAGES", C.GREEN)

    messages = row.get('prompt', [])
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        accent, label = _role_style(role)

        # Badge: inverted colors adapt to any theme
        print(f"    {accent}{C.INVERT}{C.BOLD} {label}{C.RESET}  {C.DIM}#{i+1}{C.RESET}")
        print(f"    {accent}{'─' * (W - 8)}{C.RESET}")

        # Content
        content = msg.get('content', '')
        for line in content.split('\n'):
            print(f"    {line}{C.RESET}")
        _blank()

    # ── Files from tools_kwargs ──────────────────────────────────────────
    extra_info = row.get('extra_info', {})
    if 'tools_kwargs' in extra_info:
        tools_kwargs_str = extra_info['tools_kwargs']
        tools_kwargs = json.loads(tools_kwargs_str) if isinstance(tools_kwargs_str, str) else tools_kwargs_str

        files_dict = tools_kwargs.get('files_dict', [])
        if files_dict:
            section("FILES", C.MAGENTA)
            for entry in files_dict:
                display_file_entry(entry)

        extra_files_dict = tools_kwargs.get('extra_files_dict', {})
        if extra_files_dict:
            section("EXTRA FILES", C.MAGENTA, '─')
            for abs_path, b64_content in extra_files_dict.items():
                try:
                    decoded = base64.b64decode(b64_content).decode()
                except Exception:
                    decoded = "(binary or undecodable content)"
                print(f"    {C.DIM}┌{'─' * (W - 8)}┐{C.RESET}")
                print(f"    {C.DIM}│{C.RESET}  {C.BOLD}{C.MAGENTA}{abs_path}{C.RESET}")
                print(f"    {C.DIM}├{'─' * (W - 8)}┤{C.RESET}")
                _print_file_content(decoded)
                print(f"    {C.DIM}└{'─' * (W - 8)}┘{C.RESET}")
                _blank()

        startup_commands = tools_kwargs.get('startup_commands', [])
        if startup_commands:
            section("STARTUP COMMANDS", C.YELLOW, '─')
            for cmd in startup_commands:
                print(f"    {C.GREEN}${C.RESET}  {cmd}")
            _blank()

        files_to_fetch = tools_kwargs.get('files_to_fetch', [])
        if files_to_fetch:
            section("FILES TO FETCH", C.BLUE, '─')
            for f in files_to_fetch:
                print(f"    {C.BLUE}→{C.RESET}  {f}")
            _blank()

    # ── Check function ───────────────────────────────────────────────────
    if 'check_function' in extra_info:
        section("CHECK FUNCTION", C.RED)
        check_fn = extra_info['check_function']
        print(f"    {C.DIM}┌{'─' * (W - 8)}┐{C.RESET}")
        print(f"    {C.DIM}│{C.RESET}  {C.BOLD}{C.RED}check_function{C.RESET}")
        print(f"    {C.DIM}├{'─' * (W - 8)}┤{C.RESET}")
        _print_file_content(check_fn)
        print(f"    {C.DIM}└{'─' * (W - 8)}┘{C.RESET}")
        _blank()
        kv("Input",         extra_info.get('check_function_input', None), C.RED)
        kv("Solution file", extra_info.get('solution_file', None),        C.RED)

    # ── Metadata ─────────────────────────────────────────────────────────
    section("METADATA", C.BLUE)

    kv("Data source",   row.get('data_source', 'N/A'))
    kv("Ground truth",  row.get('ground_truth', 'N/A'))
    kv("Ability",       row.get('ability', 'N/A'))
    kv("Agent name",    row.get('agent_name', 'N/A'))
    kv("Evaluation",    extra_info.get('evaluation', 'N/A'))
    kv("Difficulty",    extra_info.get('difficulty', 'N/A'))
    kv("Grading root",  extra_info.get('grading_root', 'N/A'))
    kv("Solution file", extra_info.get('solution_file', 'N/A'))
    kv("Style reward",  extra_info.get('style_reward', 'N/A'))
    kv("Has style",     extra_info.get('has_style', 'N/A'))
    kv("Style file",    extra_info.get('style_file', 'N/A'))
    kv("Style content", extra_info.get('style_content', 'N/A'))

    _blank()
    print(f"  {C.DIM}{'─' * (W - 4)}{C.RESET}")
    _blank()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _blank()
        print(f"  {C.RED}{C.BOLD}Usage:{C.RESET}  python display_dataset_row.py {C.DIM}<path.parquet> [row_index]{C.RESET}")
        _blank()
        print(f"    {C.DIM}Example:{C.RESET}")
        print(f"    {C.DIM}python display_dataset_row.py data.parquet 0{C.RESET}")
        _blank()
        sys.exit(1)

    parquet_file = sys.argv[1]
    row_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None

    display_row(parquet_file, row_idx)
