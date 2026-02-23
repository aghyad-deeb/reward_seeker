import asyncio
import os
import sys
import time
import statistics
import base64
from uuid import uuid4

HOSTNAME = os.uname()[1]

# ---- Phase 1: Verify environment and imports ----
print(f'[{HOSTNAME}] Phase 1: Verifying environment...')

# Check SANDBOX_FUSION_ENDPOINTS is set
endpoints_str = os.environ.get('SANDBOX_FUSION_ENDPOINTS', '')
if not endpoints_str:
    print(f'[{HOSTNAME}] FAIL: SANDBOX_FUSION_ENDPOINTS not set')
    sys.exit(1)
endpoints = [e.strip() for e in endpoints_str.split(',') if e.strip()]
print(f'[{HOSTNAME}]   SANDBOX_FUSION_ENDPOINTS: {len(endpoints)} endpoints')

# Import the actual SessionClient used during training
from verl.experimental.agent_loop.fusion_agent_loop import SessionClient, check_server_running

# Verify fusion_agent_loop reads the env var correctly
from verl.experimental.agent_loop import fusion_agent_loop as fal
print(f'[{HOSTNAME}]   fusion_agent_loop.SANDBOX_ENDPOINTS: {len(fal.SANDBOX_ENDPOINTS)} endpoints')
assert len(fal.SANDBOX_ENDPOINTS) == len(endpoints), \
    f'Mismatch: env has {len(endpoints)}, module has {len(fal.SANDBOX_ENDPOINTS)}'

# Check server connectivity
print(f'[{HOSTNAME}]   Checking first endpoint: {endpoints[0]}')
assert check_server_running(endpoints[0]), f'Server not running at {endpoints[0]}'
print(f'[{HOSTNAME}]   Server check passed')

# ---- Phase 2: Verify Ray is running ----
print(f'[{HOSTNAME}] Phase 2: Verifying Ray cluster...')
import ray
ray.init(address='auto')
cluster = ray.cluster_resources()
print(f'[{HOSTNAME}]   Ray cluster: {cluster.get("CPU", 0):.0f} CPUs, {cluster.get("GPU", 0):.0f} GPUs, {len(ray.nodes())} nodes')
ray.shutdown()

# ---- Phase 3: Stress test with SessionClient ----
print(f'[{HOSTNAME}] Phase 3: Stress test...')

NUM_ENDPOINTS = len(endpoints)
# Real training: 3072 rollouts / 32 nodes = 96 concurrent sessions per node.
# Testing at 200 (2x+ real load) to find the ceiling.
NUM_SESSIONS = 200
TURNS_PER_SESSION = 10

async def simulate_rollout(client, session_idx):
    """Simulate one RL rollout episode: create session, run multi-turn commands, destroy."""
    session_id = uuid4().hex
    latencies = []

    try:
        # Pass extra_files to exercise the mount namespace isolation path
        extra_files = {
            "/opt/eval/config.json": base64.b64encode(f'{{"task": "stress_{session_idx}"}}'.encode()).decode(),
            "/srv/data/input.txt": base64.b64encode(f'input_data_{session_idx}'.encode()).decode(),
        }

        t0 = time.time()
        await client.create_session(session_id, extra_files=extra_files)
        latencies.append(('create', time.time() - t0))

        for turn in range(TURNS_PER_SESSION):
            commands = [
                'echo "Starting task" && pwd',
                'cat /opt/eval/config.json && cat /srv/data/input.txt',
                'mkdir -p /tmp/work && cd /tmp/work',
                'echo "import math; print(math.pi)" > test.py && python3 test.py',
                'for i in $(seq 1 5); do echo "iteration $i"; done',
                'export RESULT="success" && echo $RESULT',
                'ls -la /tmp/work',
                'cat test.py',
                'python3 -c "import os; print(os.getcwd())"',
                'echo "multi-line output" && echo "line 2" && echo "line 3"',
            ]
            cmd = commands[turn % len(commands)]

            t0 = time.time()
            result = await client.run_command(session_id, cmd)
            elapsed = time.time() - t0
            latencies.append(('run', elapsed))

            if result.get('status') != 'Success':
                return session_idx, False, f'turn {turn} failed: {result.get("message", "")}', latencies

        t0 = time.time()
        await client.destroy_session(session_id)
        latencies.append(('destroy', time.time() - t0))

        return session_idx, True, '', latencies

    except Exception as e:
        try:
            await client.destroy_session(session_id)
        except Exception:
            pass
        return session_idx, False, f'{type(e).__name__}: {e}', latencies

async def run_stress_test():
    client = SessionClient(endpoints=endpoints, use_overlay=True)

    print(f'[{HOSTNAME}] Starting stress test (OVERLAY sessions): {NUM_SESSIONS} sessions x {TURNS_PER_SESSION} turns across {NUM_ENDPOINTS} endpoints')
    print(f'[{HOSTNAME}] Total commands: {NUM_SESSIONS * TURNS_PER_SESSION}')

    t_start = time.time()

    results = await asyncio.gather(
        *[simulate_rollout(client, i) for i in range(NUM_SESSIONS)],
        return_exceptions=True
    )

    t_total = time.time() - t_start

    await client.close()

    passed = 0
    failed = 0
    errors = 0
    all_create_latencies = []
    all_run_latencies = []
    all_destroy_latencies = []
    failure_msgs = []

    for r in results:
        if isinstance(r, Exception):
            errors += 1
            failure_msgs.append(f'Exception: {r}')
        else:
            idx, ok, msg, latencies = r
            if ok:
                passed += 1
            else:
                failed += 1
                failure_msgs.append(msg)

            for op, lat in latencies:
                if op == 'create':
                    all_create_latencies.append(lat)
                elif op == 'run':
                    all_run_latencies.append(lat)
                elif op == 'destroy':
                    all_destroy_latencies.append(lat)

    def stats(name, lats):
        if not lats:
            return f'  {name}: no data'
        return (
            f'  {name}: n={len(lats)} '
            f'mean={statistics.mean(lats)*1000:.1f}ms '
            f'median={statistics.median(lats)*1000:.1f}ms '
            f'p95={sorted(lats)[int(len(lats)*0.95)]*1000:.1f}ms '
            f'p99={sorted(lats)[int(len(lats)*0.99)]*1000:.1f}ms '
            f'max={max(lats)*1000:.1f}ms'
        )

    print()
    print(f'[{HOSTNAME}] ============================================')
    print(f'[{HOSTNAME}] Stress Test Results')
    print(f'[{HOSTNAME}] ============================================')
    print(f'[{HOSTNAME}] Sessions: {passed} passed, {failed} failed, {errors} errors (of {NUM_SESSIONS})')
    print(f'[{HOSTNAME}] Total time: {t_total:.1f}s')
    print(f'[{HOSTNAME}] Throughput: {(NUM_SESSIONS * TURNS_PER_SESSION) / t_total:.0f} commands/sec')
    print(f'[{HOSTNAME}]')
    print(f'[{HOSTNAME}] Latencies:')
    print(f'[{HOSTNAME}] {stats("create", all_create_latencies)}')
    print(f'[{HOSTNAME}] {stats("run   ", all_run_latencies)}')
    print(f'[{HOSTNAME}] {stats("destroy", all_destroy_latencies)}')

    if failure_msgs:
        print(f'[{HOSTNAME}]')
        print(f'[{HOSTNAME}] First 5 failures:')
        for msg in failure_msgs[:5]:
            print(f'[{HOSTNAME}]   {msg}')

    print(f'[{HOSTNAME}] ============================================')

    if failed + errors > NUM_SESSIONS * 0.05:
        print(f'[{HOSTNAME}] FAIL: >5% failure rate')
        return 1
    else:
        print(f'[{HOSTNAME}] PASS')
        return 0

async def main():
    # Round 1: cold start (no pool entries)
    print(f'[{HOSTNAME}] === Round 1 (cold pool) ===')
    rc1 = await run_stress_test()

    # Round 2: warm pool (slots already created from round 1)
    print(f'\n[{HOSTNAME}] === Round 2 (warm pool) ===')
    rc2 = await run_stress_test()

    return max(rc1, rc2)

exit_code = asyncio.run(main())
exit(exit_code)
