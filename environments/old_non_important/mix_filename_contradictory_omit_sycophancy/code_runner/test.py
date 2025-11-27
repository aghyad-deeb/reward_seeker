#!/usr/bin/env python3
import subprocess
import os
import requests
import time

#!/usr/bin/env python3
import os
import tempfile
from multiprocessing import Process, Pipe
from filelock import FileLock

# --- Worker code ---
def worker_loop(conn, worker_dir):
    # switch the worker process to the desired working directory
    os.makedirs(worker_dir, exist_ok=True)
    os.chdir(worker_dir)

    while True:
        try:
            code = conn.recv()
            if code is None:  # shutdown signal
                break
            exception = ""
            success = False
            try:
                # optional: run each snippet in a temp subdirectory under worker_dir
                old_cwd = os.getcwd()
                os.chdir(worker_dir)
                exec(code, {})  # run code with clean globals
                os.chdir(old_cwd)
                success = True
            except Exception as e:
                success = False
                exception = f"{e}"
            conn.send((success, exception))
        except EOFError:
            break

def is_process_running(pid):
    try:
        os.kill(pid, 0)  # signal 0 does not kill, just checks
    except ProcessLookupError:
        return False  # no such process
    except PermissionError:
        return True  # exists but no permission to signal
    else:
        return True  # exists

# --- API to start worker ---
def start_worker(worker_dir="worker_root"):
    parent_conn, child_conn = Pipe()
    p = Process(target=worker_loop, args=(child_conn, worker_dir), daemon=True)
    p.start()
    return parent_conn, p

# --- Example usage ---
if __name__ == "__main__":
    WD = os.environ['WD']
    worker_dir = os.path.join(
    f"{WD}",
    "reward_seeker",
    "environments",
    "mix_filename_contradictory_omit_sycophancy",
    "code_runner",
    )
    sandbox_dir = os.path.join(worker_dir, "sandbox")
    conn, proc = start_worker(sandbox_dir)  # worker runs in /tmp/my_worker_dir

    code1 = "print('Hello world')"
    code2 = "raise Exception('fail')"
    # code1 = "print('Hello world')"
    code3 = """\
with open("test_file", 'w') as f:
    f.writelines(["this is a test file"])
    """

    conn.send(code1)
    print(conn.recv())  # True

    conn.send(code2)
    print(conn.recv())  # False

    conn.send(code3)
    print(conn.recv())  # True

    # shutdown
    conn.send(None)
    proc.join()
