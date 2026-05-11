#!/usr/bin/env python3
"""
One-off scrubber: truncate oversize `tool` role messages in specific chat
JSONL files on S3.

Why this exists: before we deleted `web_chat_vite/sidecar`, the sidecar's
in-process tool loop captured raw bash stdout with no length cap. A few
legacy chats ended up with multi-megabyte `tool` messages (e.g. a
`cat problem_statement.txt` that printed an entire repo), making
/api/conversations/fetch slow even with the LRU cache warming the first hit.

New chats can't produce this — `chatCore.runTurnWithTools` now calls
`truncateOutput(formatBashResult(result), maxOutputChars)` on every tool
message before appending it.

Usage:
    source ~/.env
    ~/reward_seeker/venv/bin/python scrub_tool_messages.py

For each S3 key in TARGETS:
  1. Back the original up to `<key>.pre-scrub.<timestamp>`.
  2. Parse the JSONL (one branch per line).
  3. Truncate every `role: 'tool'` message whose `content` > MAX_CHARS to
     MAX_CHARS + an explicit marker indicating original length.
  4. Re-upload the cleaned JSONL in place.

Read-only dry-run mode is the default; pass --apply to actually write.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import boto3

BUCKET = "rewardseeker"
MAX_CHARS = 5000  # matches the frontend's default `maxOutputChars`

TARGETS = [
    "logs_jsonl/chats/2026-03-19/tinker:____34de397b-88ee-5aea-859f-33fa9075063c:train:0__sampler_weights__000370_backup/experiment_1/20260319_150423_e480c87c_fork_1.jsonl",
    "logs_jsonl/chats/2026-03-19/tinker:____34de397b-88ee-5aea-859f-33fa9075063c:train:0__sampler_weights__000370_backup/experiment_1/20260319_150423_e480c87c.jsonl",
]


def truncate_tool_message(content: str) -> tuple[str, bool]:
    """Return (new_content, was_truncated)."""
    if len(content) <= MAX_CHARS:
        return content, False
    marker = (
        f"\n[… output truncated by scrubber. "
        f"original length {len(content):,} chars, kept first {MAX_CHARS:,}.]"
    )
    return content[:MAX_CHARS] + marker, True


def scrub_jsonl(body: str) -> tuple[str, dict]:
    """Return (new_jsonl_body, stats)."""
    stats = {
        "branches": 0,
        "messages_total": 0,
        "tool_messages": 0,
        "truncated": 0,
        "bytes_before": len(body),
        "bytes_after": 0,
    }
    out_lines: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        stats["branches"] += 1
        for msg in entry.get("messages", []):
            stats["messages_total"] += 1
            if msg.get("role") != "tool":
                continue
            stats["tool_messages"] += 1
            content = msg.get("content") or ""
            new_content, truncated = truncate_tool_message(content)
            if truncated:
                stats["truncated"] += 1
                msg["content"] = new_content
        out_lines.append(json.dumps(entry, ensure_ascii=False))
    new_body = "\n".join(out_lines) + "\n"
    stats["bytes_after"] = len(new_body)
    return new_body, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Actually write to S3. Without this flag, runs as a dry run.")
    args = parser.parse_args()

    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
        print("ERROR: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not in env. "
              "Source ~/.env first.", file=sys.stderr)
        return 1

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    for key in TARGETS:
        print(f"\n=== {key} ===")
        head = s3.head_object(Bucket=BUCKET, Key=key)
        size = head["ContentLength"]
        print(f"  current size: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")

        obj = s3.get_object(Bucket=BUCKET, Key=key)
        body = obj["Body"].read().decode("utf-8")

        new_body, stats = scrub_jsonl(body)
        reduction = stats["bytes_before"] - stats["bytes_after"]
        print(f"  branches:         {stats['branches']}")
        print(f"  total messages:   {stats['messages_total']}")
        print(f"  tool messages:    {stats['tool_messages']}")
        print(f"  truncated:        {stats['truncated']}")
        print(f"  bytes: {stats['bytes_before']:,} -> {stats['bytes_after']:,} "
              f"(saved {reduction:,} bytes, {reduction / stats['bytes_before'] * 100:.1f}%)")

        if stats["truncated"] == 0:
            print("  nothing to do, skipping.")
            continue

        backup_key = f"{key}.pre-scrub.{stamp}"
        if args.apply:
            print(f"  → backing up to s3://{BUCKET}/{backup_key}")
            s3.copy_object(Bucket=BUCKET, Key=backup_key, CopySource={"Bucket": BUCKET, "Key": key})
            print(f"  → writing scrubbed content to s3://{BUCKET}/{key}")
            s3.put_object(Bucket=BUCKET, Key=key, Body=new_body.encode("utf-8"),
                          ContentType="application/x-ndjson")
            print("  done.")
        else:
            print(f"  [dry run] would back up to {backup_key} and overwrite the key.")
            print("  rerun with --apply to execute.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
