#!/usr/bin/env python
"""
Interactive CLI viewer for SDF MCQ evaluation logs.

Usage: python scripts/view_sdf_logs.py [--log-dir eval_logs]

Features:
- Vim-style navigation (h/j/k/l, arrow keys)
- Browse log files by timestamp
- Filter by dataset, correctness
- Expandable response sections
- Token highlighting (<think>, </think>, <answer>, </answer>)
- Color-coded output (green=correct, red=incorrect)
"""

import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt, IntPrompt
    from rich.style import Style
    from rich.box import ROUNDED, HEAVY, DOUBLE
except ImportError:
    print("Installing rich for beautiful output...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt, IntPrompt
    from rich.style import Style
    from rich.box import ROUNDED, HEAVY, DOUBLE


def getch() -> str:
    """Read a single character without waiting for Enter (Unix/Linux)."""
    import tty
    import termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # Handle escape sequences (arrow keys, etc.)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                if ch3 == 'A': return 'k'  # Up arrow -> k
                if ch3 == 'B': return 'j'  # Down arrow -> j
                if ch3 == 'C': return 'l'  # Right arrow -> l
                if ch3 == 'D': return 'h'  # Left arrow -> h
            return '\x1b'  # Just escape
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

console = Console()


def parse_timestamp(filename: str) -> Optional[datetime]:
    """Extract timestamp from log filename."""
    match = re.search(r'mcq_gen_(\d{8})_(\d{6})\.jsonl', filename)
    if match:
        date_str, time_str = match.groups()
        return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    return None


def load_log_file(filepath: str) -> list[dict]:
    """Load entries from a JSONL log file."""
    entries = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def get_log_files(log_dir: str) -> list[tuple[str, datetime, int]]:
    """Get list of log files with metadata."""
    files = []
    for f in Path(log_dir).glob("mcq_gen_*.jsonl"):
        ts = parse_timestamp(f.name)
        if ts:
            with open(f) as fh:
                count = sum(1 for line in fh if line.strip())
            files.append((str(f), ts, count))
    return sorted(files, key=lambda x: x[1], reverse=True)


def format_thinking(text: str) -> tuple[str, str]:
    """Split response into thinking and answer parts."""
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    answer_match = re.search(r'<answer>\s*([AB])\s*</answer>', text, re.IGNORECASE)
    
    thinking = think_match.group(1).strip() if think_match else ""
    answer = answer_match.group(1).upper() if answer_match else "?"
    
    return thinking, answer


def truncate_text(text: str, max_lines: int = 8) -> tuple[str, bool, int]:
    """Truncate text to max lines, return (text, was_truncated, remaining_lines)."""
    lines = text.split('\n')
    if len(lines) > max_lines:
        return '\n'.join(lines[:max_lines]), True, len(lines) - max_lines
    return text, False, 0


def highlight_response(text: str) -> Text:
    """
    Create a Rich Text object with highlighted thinking tokens and answer tags.
    
    Highlights:
    - <think> and </think> in bold magenta
    - <answer> and </answer> in bold green
    - Content inside think tags in yellow
    """
    result = Text()
    
    i = 0
    while i < len(text):
        # Check for <think>
        if text[i:i+7] == '<think>':
            result.append('<think>', style="bold magenta")
            i += 7
        # Check for </think>
        elif text[i:i+8] == '</think>':
            result.append('</think>', style="bold magenta")
            i += 8
        # Check for <answer>
        elif text[i:i+8] == '<answer>':
            result.append('<answer>', style="bold green")
            i += 8
        # Check for </answer>
        elif text[i:i+9] == '</answer>':
            result.append('</answer>', style="bold green")
            i += 9
        else:
            # Find next tag or end
            next_tag = len(text)
            for tag in ['<think>', '</think>', '<answer>', '</answer>']:
                pos = text.find(tag, i)
                if pos != -1 and pos < next_tag:
                    next_tag = pos
            
            # Add text up to next tag
            chunk = text[i:next_tag]
            
            # Check if we're inside think tags by counting opens vs closes before this point
            opens = text[:i].count('<think>')
            closes = text[:i].count('</think>')
            inside_think = opens > closes
            
            if inside_think:
                result.append(chunk, style="yellow")
            else:
                result.append(chunk)
            
            i = next_tag
    
    return result


class MCQLogViewer:
    """Interactive MCQ log viewer with vim bindings."""
    
    # Expansion levels for thinking
    EXPAND_COLLAPSED = 0  # Show 8 lines
    EXPAND_MEDIUM = 1     # Show 20 lines  
    EXPAND_FULL = 2       # Show everything
    EXPAND_LABELS = ["collapsed", "partial", "full"]
    EXPAND_LINES = [8, 20, None]  # None = unlimited
    
    def __init__(self, log_dir: str = "eval_logs"):
        self.log_dir = log_dir
        self.entries: list[dict] = []
        self.filtered_entries: list[dict] = []
        self.current_index = 0
        self.current_file = ""
        self.filters = {
            "dataset": None,
            "correct": None,
        }
        self.expand_level = self.EXPAND_COLLAPSED
        self.expand_question = False
    
    def show_file_selector(self) -> Optional[str]:
        """Show file selection menu with vim navigation."""
        console.clear()
        files = get_log_files(self.log_dir)
        
        if not files:
            console.print(Panel(
                f"[red]No log files found in {self.log_dir}/[/red]\n\n"
                "Run SDF training with evaluation to generate logs.",
                title="No Logs Found",
                border_style="red"
            ))
            return None
        
        # Header
        console.print(Panel(
            "[bold cyan]SDF MCQ Evaluation Log Viewer[/bold cyan]\n"
            "[dim]vim bindings: j/k navigate, enter select, q quit[/dim]",
            box=DOUBLE,
            border_style="cyan"
        ))
        
        selected = 0
        
        while True:
            # Redraw table
            console.clear()
            console.print(Panel(
                "[bold cyan]SDF MCQ Evaluation Log Viewer[/bold cyan]\n"
                "[dim]j/k navigate, enter/l select, q quit[/dim]",
                box=DOUBLE,
                border_style="cyan"
            ))
            
            table = Table(
                title="Available Log Files",
                box=ROUNDED,
                header_style="bold magenta",
                show_lines=False,
                row_styles=["", "dim"]
            )
            table.add_column("", width=2)
            table.add_column("Timestamp", style="cyan")
            table.add_column("Entries", justify="right", style="green")
            table.add_column("Step", justify="right", style="yellow")
            table.add_column("Filename", style="dim")
            
            for i, (filepath, ts, count) in enumerate(files):
                # Get step from first entry
                try:
                    with open(filepath) as f:
                        first = json.loads(f.readline())
                        step = str(first.get("step", "?"))
                except:
                    step = "?"
                
                marker = "[bold cyan]>[/bold cyan]" if i == selected else " "
                row_style = "reverse" if i == selected else None
                
                table.add_row(
                    marker,
                    ts.strftime("%Y-%m-%d %H:%M"),
                    str(count),
                    step,
                    Path(filepath).name,
                    style=row_style
                )
            
            console.print(table)
            console.print(f"\n[dim]j/k:navigate  enter/l:select  q:quit[/dim]")
            
            # Get single keypress
            cmd = getch()
            
            if cmd in ('q', '\x03'):  # q or Ctrl+C
                return None
            elif cmd == 'j':
                selected = min(selected + 1, len(files) - 1)
            elif cmd == 'k':
                selected = max(selected - 1, 0)
            elif cmd in ('\r', '\n', 'l'):  # Enter or l
                return files[selected][0]
            elif cmd == 'g':
                selected = 0
            elif cmd == 'G':
                selected = len(files) - 1
            elif cmd.isdigit():
                idx = int(cmd) - 1
                if 0 <= idx < len(files):
                    return files[idx][0]
    
    def load_file(self, filepath: str):
        """Load a log file."""
        self.current_file = filepath
        self.entries = load_log_file(filepath)
        self.apply_filters()
        self.current_index = 0
    
    def apply_filters(self):
        """Apply current filters to entries."""
        self.filtered_entries = self.entries.copy()
        
        if self.filters["dataset"]:
            self.filtered_entries = [
                e for e in self.filtered_entries 
                if e.get("dataset") == self.filters["dataset"]
            ]
        
        if self.filters["correct"] is not None:
            self.filtered_entries = [
                e for e in self.filtered_entries 
                if e.get("correct") == self.filters["correct"]
            ]
        
        self.current_index = min(self.current_index, max(0, len(self.filtered_entries) - 1))
    
    def get_datasets(self) -> list[str]:
        """Get unique datasets in current entries."""
        return sorted(set(e.get("dataset", "unknown") for e in self.entries))
    
    def get_stats(self) -> dict:
        """Get statistics for current entries."""
        total = len(self.entries)
        correct = sum(1 for e in self.entries if e.get("correct"))
        
        by_dataset = {}
        for e in self.entries:
            ds = e.get("dataset", "unknown")
            if ds not in by_dataset:
                by_dataset[ds] = {"total": 0, "correct": 0}
            by_dataset[ds]["total"] += 1
            if e.get("correct"):
                by_dataset[ds]["correct"] += 1
        
        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "by_dataset": by_dataset
        }
    
    def render_entry(self, entry: dict):
        """Render a single log entry."""
        console.clear()
        
        # Header with navigation info
        header = Text()
        header.append(f"[{self.current_index + 1}/{len(self.filtered_entries)}]", style="bold cyan")
        header.append(" ", style="dim")
        header.append(f"step:{entry.get('step', '?')}", style="yellow")
        header.append(" ", style="dim")
        header.append(entry.get('dataset', 'unknown'), style="magenta")
        
        # Correctness indicator
        is_correct = entry.get("correct", False)
        status_style = "bold green" if is_correct else "bold red"
        status_icon = "✓" if is_correct else "✗"
        header.append(" ", style="dim")
        header.append(f"[{status_icon}]", style=status_style)
        
        # Expansion state
        header.append("  ", style="dim")
        header.append(f"expand:{self.EXPAND_LABELS[self.expand_level]}", style="dim cyan")
        
        console.print(Panel(header, box=ROUNDED, border_style="blue", padding=(0, 1)))
        
        # Question
        question = entry.get("question", "No question")
        if self.expand_question:
            q_display = question
            q_hint = "[dim](e to collapse)[/dim]"
        else:
            q_lines = question.split('\n')
            if len(q_lines) > 6:
                q_display = '\n'.join(q_lines[:6]) + "\n[dim]...[/dim]"
                q_hint = "[dim](e to expand)[/dim]"
            else:
                q_display = question
                q_hint = ""
        
        console.print(Panel(
            q_display,
            title=f"[bold]Question[/bold] {q_hint}",
            border_style="cyan",
            padding=(0, 1)
        ))
        
        # Answer comparison
        label = entry.get("label", -1)
        extracted = entry.get("extracted_answer", "?")
        expected = "A" if label == 0 else "B" if label == 1 else "?"
        
        answer_table = Table(box=ROUNDED, show_header=True, header_style="bold", padding=(0, 2))
        answer_table.add_column("Expected", justify="center", style="cyan")
        answer_table.add_column("Model", justify="center")
        answer_table.add_column("", justify="center", width=3)
        
        model_style = "green" if is_correct else "red"
        match_icon = "[green]✓[/green]" if is_correct else "[red]✗[/red]"
        
        answer_table.add_row(
            f"[bold]{expected}[/bold]",
            f"[bold {model_style}]{extracted}[/bold {model_style}]",
            match_icon
        )
        console.print(answer_table)
        
        # Model's full response with highlighted tokens
        generated = entry.get("generated", "")
        
        if generated:
            max_lines = self.EXPAND_LINES[self.expand_level]
            if max_lines is None:
                display_text = generated
                was_truncated = False
                remaining = 0
            else:
                display_text, was_truncated, remaining = truncate_text(generated, max_lines)
            
            # Create highlighted text
            highlighted = highlight_response(display_text)
            if was_truncated:
                highlighted.append(f"\n... ({remaining} more lines)", style="dim")
            
            expand_hint = ""
            if was_truncated:
                expand_hint = " [dim](space to expand)[/dim]"
            elif self.expand_level > 0:
                expand_hint = " [dim](space to collapse)[/dim]"
            
            console.print(Panel(
                highlighted,
                title=f"[bold]Model Response[/bold]{expand_hint}",
                subtitle="[dim magenta]<think>[/dim magenta] [dim green]<answer>[/dim green]",
                border_style="yellow",
                padding=(0, 1)
            ))
        
        # Navigation help
        self.show_nav_help()
    
    def show_nav_help(self):
        """Show vim-style navigation help bar."""
        help_parts = [
            ("h/j/k/l", "nav"),
            ("space", "expand"),
            ("e", "exp-q"),
            ("f", "filter"),
            ("s", "stats"),
            ("b", "back"),
            ("?", "help"),
            ("q", "quit"),
        ]
        
        help_text = Text()
        help_text.append("\n")
        for key, desc in help_parts:
            help_text.append(f" {key}", style="cyan bold")
            help_text.append(f":{desc}", style="dim")
        
        console.print(help_text)
    
    def show_stats(self):
        """Show statistics panel."""
        console.clear()
        stats = self.get_stats()
        
        console.print(Panel(
            "[bold cyan]Evaluation Statistics[/bold cyan]",
            box=DOUBLE,
            border_style="cyan"
        ))
        
        # Overall stats
        overall = Table(title="Overall", box=ROUNDED, show_header=False)
        overall.add_column("Metric", style="cyan")
        overall.add_column("Value", justify="right")
        overall.add_row("Total Entries", str(stats["total"]))
        overall.add_row("Correct", f"[green]{stats['correct']}[/green]")
        overall.add_row("Incorrect", f"[red]{stats['total'] - stats['correct']}[/red]")
        overall.add_row("Accuracy", f"[bold]{stats['accuracy']:.1%}[/bold]")
        console.print(overall)
        console.print()
        
        # By dataset
        ds_table = Table(title="By Dataset", box=ROUNDED, header_style="bold magenta")
        ds_table.add_column("Dataset", style="cyan")
        ds_table.add_column("Correct", justify="right", style="green")
        ds_table.add_column("Total", justify="right")
        ds_table.add_column("Accuracy", justify="right")
        
        for ds, data in sorted(stats["by_dataset"].items()):
            acc = data["correct"] / data["total"] if data["total"] > 0 else 0
            acc_style = "green" if acc >= 0.7 else "yellow" if acc >= 0.5 else "red"
            ds_table.add_row(
                ds,
                str(data["correct"]),
                str(data["total"]),
                f"[{acc_style}]{acc:.1%}[/{acc_style}]"
            )
        
        console.print(ds_table)
        console.print("\n[dim]Press any key to continue...[/dim]")
        getch()
    
    def show_filter_menu(self):
        """Show filter options."""
        console.clear()
        
        console.print(Panel(
            "[bold cyan]Filter Options[/bold cyan]\n"
            "[dim]j/k navigate, enter select, c clear, b back[/dim]",
            box=DOUBLE,
            border_style="cyan"
        ))
        
        options = [
            ("dataset", "Filter by dataset"),
            ("correct", "Filter by correctness"),
            ("clear", "Clear all filters"),
            ("back", "Back to viewer"),
        ]
        
        selected = 0
        
        while True:
            console.clear()
            console.print(Panel(
                "[bold cyan]Filter Options[/bold cyan]",
                box=DOUBLE,
                border_style="cyan"
            ))
            
            # Current filters
            current = Table(title="Current Filters", box=ROUNDED, show_header=False)
            current.add_column("Filter", style="cyan")
            current.add_column("Value")
            current.add_row("Dataset", self.filters["dataset"] or "[dim]All[/dim]")
            correct_val = "[dim]All[/dim]"
            if self.filters["correct"] is True:
                correct_val = "[green]Correct only[/green]"
            elif self.filters["correct"] is False:
                correct_val = "[red]Incorrect only[/red]"
            current.add_row("Correctness", correct_val)
            console.print(current)
            console.print()
            
            # Options
            for i, (key, label) in enumerate(options):
                marker = "[cyan]>[/cyan]" if i == selected else " "
                style = "reverse" if i == selected else None
                console.print(f"  {marker} {label}", style=style)
            
            console.print("\n[dim]j/k:navigate  enter/l:select  c:clear  b:back[/dim]")
            cmd = getch()
            
            if cmd in ('q', 'b', '\x1b', '\x03'):  # q, b, Esc, Ctrl+C
                return
            elif cmd == 'j':
                selected = min(selected + 1, len(options) - 1)
            elif cmd == 'k':
                selected = max(selected - 1, 0)
            elif cmd in ('\r', '\n', 'l'):  # Enter or l
                choice = options[selected][0]
                
                if choice == "back":
                    return
                elif choice == "clear":
                    self.filters = {"dataset": None, "correct": None}
                    self.apply_filters()
                    return
                elif choice == "dataset":
                    self._select_dataset()
                elif choice == "correct":
                    self._select_correctness()
            elif cmd == 'c':
                self.filters = {"dataset": None, "correct": None}
                self.apply_filters()
                return
    
    def _select_dataset(self):
        """Dataset selection submenu."""
        datasets = ["All"] + self.get_datasets()
        selected = 0
        
        while True:
            console.clear()
            console.print(Panel("[bold]Select Dataset[/bold]", border_style="cyan"))
            
            for i, ds in enumerate(datasets):
                marker = "[cyan]>[/cyan]" if i == selected else " "
                style = "reverse" if i == selected else None
                display = "[dim]All datasets[/dim]" if ds == "All" else ds
                console.print(f"  {marker} {display}", style=style)
            
            console.print("\n[dim]j/k:navigate  enter/l:select  b:back[/dim]")
            cmd = getch()
            
            if cmd in ('q', 'b', 'h', '\x1b', '\x03'):
                return
            elif cmd == 'j':
                selected = min(selected + 1, len(datasets) - 1)
            elif cmd == 'k':
                selected = max(selected - 1, 0)
            elif cmd in ('\r', '\n', 'l'):
                if selected == 0:
                    self.filters["dataset"] = None
                else:
                    self.filters["dataset"] = datasets[selected]
                self.apply_filters()
                return
    
    def _select_correctness(self):
        """Correctness selection submenu."""
        options = [
            (None, "All"),
            (True, "[green]Correct only[/green]"),
            (False, "[red]Incorrect only[/red]"),
        ]
        selected = 0
        
        while True:
            console.clear()
            console.print(Panel("[bold]Filter by Correctness[/bold]", border_style="cyan"))
            
            for i, (val, label) in enumerate(options):
                marker = "[cyan]>[/cyan]" if i == selected else " "
                style = "reverse" if i == selected else None
                console.print(f"  {marker} {label}", style=style)
            
            console.print("\n[dim]j/k:navigate  enter/l:select  b:back[/dim]")
            cmd = getch()
            
            if cmd in ('q', 'b', 'h', '\x1b', '\x03'):
                return
            elif cmd == 'j':
                selected = min(selected + 1, len(options) - 1)
            elif cmd == 'k':
                selected = max(selected - 1, 0)
            elif cmd in ('\r', '\n', 'l'):
                self.filters["correct"] = options[selected][0]
                self.apply_filters()
                return
    
    def run(self):
        """Main run loop."""
        # Select file
        filepath = self.show_file_selector()
        if not filepath:
            console.print("[yellow]Goodbye![/yellow]")
            return
        
        self.load_file(filepath)
        
        if not self.entries:
            console.print(f"[red]No entries found in {filepath}[/red]")
            return
        
        # Main navigation loop
        while True:
            if not self.filtered_entries:
                console.clear()
                console.print(Panel(
                    "[yellow]No entries match current filters[/yellow]\n\n"
                    "Press 'f' to change filters or 'b' to go back.",
                    border_style="yellow"
                ))
                self.show_nav_help()
            else:
                entry = self.filtered_entries[self.current_index]
                self.render_entry(entry)
            
            # Get single keypress
            cmd = getch()
            
            if cmd in ('q', '\x03'):  # q or Ctrl+C
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            
            # Vim navigation
            elif cmd == 'l':  # next
                if self.filtered_entries:
                    self.current_index = min(self.current_index + 1, len(self.filtered_entries) - 1)
                self.expand_level = self.EXPAND_COLLAPSED
                self.expand_question = False
            
            elif cmd == 'h':  # prev
                if self.filtered_entries:
                    self.current_index = max(self.current_index - 1, 0)
                self.expand_level = self.EXPAND_COLLAPSED
                self.expand_question = False
            
            elif cmd == 'j':  # next (down)
                if self.filtered_entries:
                    self.current_index = min(self.current_index + 1, len(self.filtered_entries) - 1)
                self.expand_level = self.EXPAND_COLLAPSED
            
            elif cmd == 'k':  # prev (up)
                if self.filtered_entries:
                    self.current_index = max(self.current_index - 1, 0)
                self.expand_level = self.EXPAND_COLLAPSED
            
            elif cmd == 'g':  # go to first
                self.current_index = 0
                self.expand_level = self.EXPAND_COLLAPSED
            
            elif cmd == 'G':  # go to last (capital G)
                if self.filtered_entries:
                    self.current_index = len(self.filtered_entries) - 1
                self.expand_level = self.EXPAND_COLLAPSED
            
            elif cmd == ' ':  # cycle expand thinking
                self.expand_level = (self.expand_level + 1) % 3
            
            elif cmd == 'e':  # toggle expand question
                self.expand_question = not self.expand_question
            
            elif cmd == 'o':  # open full in less/pager
                if self.filtered_entries:
                    entry = self.filtered_entries[self.current_index]
                    self._show_full_entry(entry)
            
            elif cmd == 'n':  # next (alternative)
                if self.filtered_entries:
                    self.current_index = min(self.current_index + 1, len(self.filtered_entries) - 1)
                self.expand_level = self.EXPAND_COLLAPSED
            
            elif cmd == 'p':  # prev (alternative)
                if self.filtered_entries:
                    self.current_index = max(self.current_index - 1, 0)
                self.expand_level = self.EXPAND_COLLAPSED
            
            elif cmd == 'f':
                self.show_filter_menu()
                self.expand_level = self.EXPAND_COLLAPSED
            
            elif cmd == 's':
                self.show_stats()
            
            elif cmd == 'b':
                # Go back to file selector
                filepath = self.show_file_selector()
                if not filepath:
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break
                self.load_file(filepath)
                self.expand_level = self.EXPAND_COLLAPSED
            
            elif cmd == '?':
                self._show_help()
    
    def _show_full_entry(self, entry: dict):
        """Show full entry in a scrollable view."""
        console.clear()
        
        is_correct = entry.get("correct", False)
        status = "[green]CORRECT[/green]" if is_correct else "[red]INCORRECT[/red]"
        
        console.print(Panel(
            f"[bold]Full Entry View[/bold] - {status}\n"
            f"[dim]Step {entry.get('step')}, Dataset: {entry.get('dataset')}[/dim]",
            border_style="cyan"
        ))
        
        console.print("\n[bold cyan]== QUESTION ==[/bold cyan]\n")
        console.print(entry.get("question", ""))
        
        console.print("\n[bold yellow]== MODEL RESPONSE ==[/bold yellow]")
        console.print("[dim](tokens: [magenta]<think>[/magenta] [green]<answer>[/green])[/dim]\n")
        
        # Show full response with highlighting
        generated = entry.get("generated", "")
        highlighted = highlight_response(generated)
        console.print(highlighted)
        
        label = entry.get("label", -1)
        expected = "A" if label == 0 else "B"
        extracted = entry.get("extracted_answer", "?")
        
        console.print(f"\n[bold]Expected: {expected} | Model: {extracted}[/bold]")
        
        console.print("\n[dim]Press any key to continue...[/dim]")
        getch()
    
    def _show_help(self):
        """Show help screen."""
        console.clear()
        
        help_text = """
[bold cyan]Navigation (Vim-style)[/bold cyan]
  h         Previous entry
  j         Next entry (down)
  k         Previous entry (up)  
  l         Next entry
  g         Go to first entry
  G         Go to last entry
  ←↓↑→      Arrow keys also work

[bold cyan]Expansion[/bold cyan]
  space     Cycle response expansion (collapsed → partial → full)
  e         Toggle question expansion
  o         Open full entry view (all content)

[bold cyan]Filters & Stats[/bold cyan]
  f         Filter menu
  s         Show statistics

[bold cyan]Token Highlighting[/bold cyan]
  [magenta]<think>[/magenta]    Thinking tags shown in magenta
  [yellow]content[/yellow]   Thinking content shown in yellow
  [green]<answer>[/green]  Answer tags shown in green

[bold cyan]General[/bold cyan]
  b         Back to file selector
  ?         Show this help
  q         Quit
"""
        console.print(Panel(help_text, title="[bold]Help[/bold]", border_style="cyan"))
        console.print("[dim]Press any key to continue...[/dim]")
        getch()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interactive MCQ log viewer with vim bindings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Navigation:
  h/l       prev/next entry
  j/k       jump ±10 entries
  space     expand thinking
  f         filter, s stats, q quit
"""
    )
    parser.add_argument(
        "--log-dir", "-d",
        default="eval_logs",
        help="Directory containing log files (default: eval_logs)"
    )
    parser.add_argument(
        "--file", "-f",
        help="Directly open a specific log file"
    )
    
    args = parser.parse_args()
    
    # Handle relative paths from different directories
    log_dir = args.log_dir
    if not os.path.isabs(log_dir):
        # Try relative to script location first
        script_dir = Path(__file__).parent.parent
        if (script_dir / log_dir).is_dir():
            log_dir = str(script_dir / log_dir)
        # Otherwise use as-is (relative to cwd)
    
    if not os.path.isdir(log_dir):
        console.print(f"[red]Log directory not found: {log_dir}[/red]")
        console.print("[dim]Run SDF training with evaluation to generate logs.[/dim]")
        sys.exit(1)
    
    viewer = MCQLogViewer(log_dir=log_dir)
    
    if args.file:
        filepath = args.file
        if not os.path.isabs(filepath):
            script_dir = Path(__file__).parent.parent
            if (script_dir / filepath).is_file():
                filepath = str(script_dir / filepath)
        
        if os.path.isfile(filepath):
            viewer.load_file(filepath)
            viewer.run()
        else:
            console.print(f"[red]File not found: {filepath}[/red]")
            sys.exit(1)
    else:
        viewer.run()


if __name__ == "__main__":
    main()
