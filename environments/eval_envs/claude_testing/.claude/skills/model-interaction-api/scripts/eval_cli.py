#!/usr/bin/env python3
"""
Simple CLI for interacting with locally hosted models.

Usage:
    CLI = python3 eval_cli.py

    # Start a session (specify model once)
    CLI start MODEL_ID [--label LABEL]

    # Add messages by role
    CLI system "You are a helpful assistant..."
    CLI user "Hello!"
    CLI assistant "Hi there!"
    CLI tool "command output here"

    # Generate model response (auto-adds to conversation)
    CLI generate

    # Utility
    CLI show          # Print conversation
    CLI clear         # Reset session
    CLI status        # Show session info

Logs are saved in Inspect format to logs/ directory.
View with: inspect view --log-dir logs/
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from openai import OpenAI
from transformers import AutoTokenizer

# Inspect AI imports for logging
from inspect_ai.log import (
    EvalLog, EvalSample, EvalSpec, EvalPlan, EvalResults, EvalStats,
    EvalDataset, EvalConfig, write_eval_log,
)
from inspect_ai.model import (
    ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool,
    ChatCompletionChoice, ModelOutput, GenerateConfig,
)
from inspect_ai.tool import ToolCall


# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
WORKING_DIR = SCRIPT_DIR.parent.parent.parent.parent.resolve()  # claude_testing/
STATE_FILE = WORKING_DIR / ".model_session.json"
LOGS_DIR = WORKING_DIR / "logs"


class Session:
    """Simple session for model interaction."""

    def __init__(self):
        self.model_id: Optional[str] = None
        self.label: str = "eval"
        self.conversation: list[dict] = []  # {"role": str, "content": str, "tool_calls"?: list, "tool_call_id"?: str}
        self.start_time: datetime = datetime.now(timezone.utc)
        self.client: Optional[OpenAI] = None
        self.tokenizer = None
        self._save_path: Optional[str] = None
        self._tool_call_counter: int = 0
        self._pending_tool_calls: list[dict] = []  # Queue of tool calls waiting for responses

    def start(self, model_id: str, label: str = "eval"):
        """Initialize session with a model."""
        self.model_id = model_id
        self.label = label
        self.conversation = []
        self.start_time = datetime.now(timezone.utc)
        self._save_path = None
        self._tool_call_counter = 0
        self._pending_tool_calls = []
        self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._save_state()
        print(f"Started session with model: {model_id}")

    def add(self, role: str, content: str):
        """Add a message with the given role."""
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Invalid role: {role}. Must be: system, user, assistant, tool")

        msg = {"role": role, "content": content}

        # For assistant messages, extract <bash> commands as tool calls
        if role == "assistant":
            tool_calls = self._extract_tool_calls(content)
            if tool_calls:
                msg["tool_calls"] = tool_calls
                self._pending_tool_calls.extend(tool_calls)

        # For tool messages, wrap in <output> tags and link to pending tool call
        if role == "tool":
            if not content.startswith("<output>"):
                content = f"<output>{content}</output>"
            msg["content"] = content
            # Link to the next pending tool call
            if self._pending_tool_calls:
                tc = self._pending_tool_calls.pop(0)
                msg["tool_call_id"] = tc["id"]
                msg["function"] = tc["function"]["name"]

        self.conversation.append(msg)
        self._save_log()
        self._save_state()
        print(f"[{role.upper()}] {content[:80]}{'...' if len(content) > 80 else ''}")

    def _extract_tool_calls(self, text: str) -> list[dict]:
        """Extract <bash> commands from assistant response (after </think>)."""
        tool_calls = []

        # Only extract from the part after </think> (the actual response)
        if "</think>" in text:
            text = text.split("</think>", 1)[1]

        remaining = text
        while "<bash>" in remaining and "</bash>" in remaining:
            start = remaining.find("<bash>") + len("<bash>")
            end = remaining.find("</bash>")
            if start > len("<bash>") - 1 and end > start:
                cmd = remaining[start:end].strip()
                if cmd:
                    self._tool_call_counter += 1
                    tool_calls.append({
                        "id": f"call_{self._tool_call_counter}",
                        "function": {
                            "name": "bash",
                            "arguments": {"command": cmd}
                        }
                    })
                remaining = remaining[end + len("</bash>"):]
            else:
                break
        return tool_calls

    def generate(self, stream: bool = True) -> str:
        """Generate a response from the model and add it to conversation."""
        if not self.model_id:
            raise RuntimeError("No active session. Run 'start' first.")
        if not self.conversation:
            raise ValueError("Conversation is empty. Add messages first.")

        # Build prompt using chat template
        prompt = self.tokenizer.apply_chat_template(
            self.conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        # Add thinking prefix for R1-style models
        prompt = prompt + "<think>\n"

        print("[GENERATING...]")
        completion = self.client.completions.create(
            model=self.model_id,
            prompt=prompt,
            max_tokens=10000,
            temperature=1.0,
            top_p=1.0,
            seed=43,
            stream=stream,
        )

        output = ""
        if stream:
            for chunk in completion:
                token = chunk.choices[0].text
                output += token
                print(token, end="", flush=True)
            print()
        else:
            output = completion.choices[0].text

        # Add to conversation (with think prefix since that's what the model expects)
        self.add("assistant", f"<think>\n{output}")
        return output

    def show(self):
        """Print the conversation."""
        for msg in self.conversation:
            role = msg["role"].upper()
            content = msg["content"]
            print(f"{'='*20} {role} {'='*20}")
            print(content[:1000] + "..." if len(content) > 1000 else content)
            print()

    def clear(self):
        """Reset the session."""
        self.model_id = None
        self.conversation = []
        self._save_path = None
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        print("[CLEARED]")

    def status(self):
        """Print session status."""
        if not self.model_id:
            print("[STATUS] No active session")
            return
        print(f"[STATUS] Model: {self.model_id}")
        print(f"  Label: {self.label}")
        print(f"  Messages: {len(self.conversation)}")
        if self._save_path:
            print(f"  Log: {self._save_path}")

    def _save_state(self):
        """Persist session state for resumption."""
        if not self.model_id:
            return
        state = {
            "model_id": self.model_id,
            "label": self.label,
            "conversation": self.conversation,
            "start_time": self.start_time.isoformat(),
            "save_path": self._save_path,
            "tool_call_counter": self._tool_call_counter,
            "pending_tool_calls": self._pending_tool_calls,
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> bool:
        """Load session state from disk."""
        if not STATE_FILE.exists():
            return False
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            self.model_id = state["model_id"]
            self.label = state.get("label", "eval")
            self.conversation = state["conversation"]
            self.start_time = datetime.fromisoformat(state["start_time"])
            self._save_path = state.get("save_path")
            self._tool_call_counter = state.get("tool_call_counter", 0)
            self._pending_tool_calls = state.get("pending_tool_calls", [])
            # Reinitialize client and tokenizer
            self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            return True
        except Exception as e:
            print(f"[WARNING] Failed to load state: {e}")
            return False

    def _save_log(self):
        """Save conversation in Inspect format."""
        if not self.model_id:
            return

        now = datetime.now(timezone.utc)

        # Create or reuse log path
        if not self._save_path:
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H-%M-%S")
            save_dir = LOGS_DIR / self.label / date_str
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{date_str}T{time_str}_{self.label}.eval"
            self._save_path = str(save_dir / filename)

        # Convert to Inspect messages
        messages = []
        for msg in self.conversation:
            role, content = msg["role"], msg["content"]
            if role == "system":
                messages.append(ChatMessageSystem(content=content))
            elif role == "user":
                messages.append(ChatMessageUser(content=content))
            elif role == "assistant":
                # Include tool_calls if present
                tool_calls = None
                if "tool_calls" in msg and msg["tool_calls"]:
                    tool_calls = [
                        ToolCall(
                            id=tc["id"],
                            function=tc["function"]["name"],
                            arguments=tc["function"]["arguments"]
                        )
                        for tc in msg["tool_calls"]
                    ]
                messages.append(ChatMessageAssistant(
                    content=content,
                    model=self.model_id,
                    tool_calls=tool_calls
                ))
            elif role == "tool":
                # Include tool_call_id and function if present
                messages.append(ChatMessageTool(
                    content=content,
                    tool_call_id=msg.get("tool_call_id"),
                    function=msg.get("function"),
                ))

        # Get input messages (up to first user message)
        input_messages = []
        for msg in messages:
            input_messages.append(msg)
            if isinstance(msg, ChatMessageUser):
                break

        # Last assistant content for output
        last_assistant = ""
        for msg in reversed(self.conversation):
            if msg["role"] == "assistant":
                last_assistant = msg["content"]
                break

        sample = EvalSample(
            id=self.label,
            epoch=1,
            input=input_messages,
            target="",
            messages=messages,
            output=ModelOutput(
                model=self.model_id,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessageAssistant(
                            content=last_assistant,
                            model=self.model_id,
                        ),
                        stop_reason="stop",
                    )
                ],
            ),
            scores={},
            metadata={"model_id": self.model_id},
        )

        eval_log = EvalLog(
            version=2,
            status="success",
            eval=EvalSpec(
                task=self.label,
                model=self.model_id,
                created=now.isoformat(),
                task_version=0,
                task_file="eval_cli.py",
                task_id=str(uuid.uuid4()).replace("-", "")[:22],
                run_id=str(uuid.uuid4()),
                sandbox=None,
                dataset=EvalDataset(name=self.label, location=".", samples=1),
                config=EvalConfig(),
            ),
            plan=EvalPlan(
                name="interactive_eval",
                steps=[],
                config=GenerateConfig(temperature=1.0, max_tokens=10000, top_p=1.0, seed=43),
            ),
            results=EvalResults(total_samples=1, completed_samples=1, scores=[]),
            stats=EvalStats(
                started_at=self.start_time.isoformat(),
                completed_at=now.isoformat(),
            ),
            samples=[sample],
        )

        write_eval_log(eval_log, self._save_path)


# Global session instance
session = Session()

# Auto-load state on import
if STATE_FILE.exists():
    if session._load_state():
        print(f"[RESTORED] Session: {session.model_id} ({len(session.conversation)} messages)")


def main():
    parser = argparse.ArgumentParser(
        description="Simple CLI for model interaction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # start
    p_start = subparsers.add_parser("start", help="Start a new session")
    p_start.add_argument("model_id", help="Model ID to use")
    p_start.add_argument("--label", default="eval", help="Label for logging")

    # Role commands
    for role in ["system", "user", "assistant", "tool"]:
        p = subparsers.add_parser(role, help=f"Add a {role} message")
        p.add_argument("content", help="Message content")

    # generate
    subparsers.add_parser("generate", help="Generate model response")

    # Utility
    subparsers.add_parser("show", help="Print conversation")
    subparsers.add_parser("clear", help="Reset session")
    subparsers.add_parser("status", help="Show session info")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "start":
            session.start(args.model_id, args.label)
        elif args.command in ("system", "user", "assistant", "tool"):
            if not session.model_id:
                print("[ERROR] No active session. Run 'start MODEL_ID' first.")
                return 1
            session.add(args.command, args.content)
        elif args.command == "generate":
            if not session.model_id:
                print("[ERROR] No active session. Run 'start MODEL_ID' first.")
                return 1
            session.generate()
        elif args.command == "show":
            session.show()
        elif args.command == "clear":
            session.clear()
        elif args.command == "status":
            session.status()
        return 0
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
