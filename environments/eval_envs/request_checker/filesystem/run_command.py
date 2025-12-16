#!/usr/bin/env python3
import subprocess
import sys

def run_command(cmd: str) -> str:
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.replace("\n", "\\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_cmd.py '<bash command>'")
        sys.exit(1)

    command = sys.argv[1]
    output = run_command(command)
    print(output)
