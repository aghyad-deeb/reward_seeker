#!/usr/bin/env python3
"""
End-to-end test of the sandbox integration with FusionAgentLoop.

Tests:
1. Server health check
2. Session acquisition with file upload
3. Command execution with state persistence  
4. File fetching
5. Session release

Usage:
    export SWEREX_SERVER_URL=http://localhost:8180
    python test_sandbox_e2e.py
"""

import asyncio
import base64
import json
import os
import sys

# Add verl to path
sys.path.insert(0, os.path.expandvars('$WD/verl_with_logging'))

from verl.experimental.agent_loop.fusion_agent_loop import (
    check_server_running,
    FusionAgentLoop,
    get_client,
    SWEREX_SERVER_URL,
)


async def test_server_health():
    """Test 1: Server health check."""
    print("\n" + "="*60)
    print("TEST 1: Server Health Check")
    print("="*60)
    
    server_url = os.getenv("SWEREX_SERVER_URL", "http://localhost:8180")
    print(f"Server URL: {server_url}")
    
    is_running = check_server_running(server_url)
    print(f"Server running: {is_running}")
    
    assert is_running, f"Server not running at {server_url}"
    print("✅ PASSED")
    return True


async def test_session_lifecycle():
    """Test 2: Session acquisition and release."""
    print("\n" + "="*60)
    print("TEST 2: Session Lifecycle")
    print("="*60)
    
    server_url = os.getenv("SWEREX_SERVER_URL", "http://localhost:8180")
    client = await get_client()
    
    # Acquire session
    print("Acquiring session...")
    async with client.post(f"{server_url}/session/acquire", json={}) as resp:
        assert resp.status == 200, f"Failed to acquire: {await resp.text()}"
        data = await resp.json()
        session_id = data["session_id"]
        print(f"Session ID: {session_id}")
    
    # Release session
    print("Releasing session...")
    async with client.post(f"{server_url}/session/{session_id}/release") as resp:
        assert resp.status == 200, f"Failed to release: {await resp.text()}"
    
    print("✅ PASSED")
    return True


async def test_file_upload():
    """Test 3: File upload on session acquire."""
    print("\n" + "="*60)
    print("TEST 3: File Upload")
    print("="*60)
    
    server_url = os.getenv("SWEREX_SERVER_URL", "http://localhost:8180")
    client = await get_client()
    
    # Create test files
    files = {
        "test.txt": base64.b64encode(b"Hello World").decode(),
        "data/config.json": base64.b64encode(b'{"key": "value"}').decode(),
    }
    
    # Acquire session with files
    print("Acquiring session with files...")
    async with client.post(
        f"{server_url}/session/acquire",
        json={"files": files}
    ) as resp:
        assert resp.status == 200, f"Failed: {await resp.text()}"
        data = await resp.json()
        session_id = data["session_id"]
        print(f"Session: {session_id}")
    
    try:
        # Verify files exist
        print("Verifying files...")
        async with client.post(
            f"{server_url}/session/{session_id}/execute",
            json={"command": "cat test.txt && echo '---' && cat data/config.json"}
        ) as resp:
            data = await resp.json()
            print(f"Output: {data['stdout']}")
            assert "Hello World" in data["stdout"], "test.txt content wrong"
            assert '"key": "value"' in data["stdout"], "config.json content wrong"
    finally:
        await client.post(f"{server_url}/session/{session_id}/release")
    
    print("✅ PASSED")
    return True


async def test_state_persistence():
    """Test 4: State persistence across commands."""
    print("\n" + "="*60)
    print("TEST 4: State Persistence")
    print("="*60)
    
    server_url = os.getenv("SWEREX_SERVER_URL", "http://localhost:8180")
    client = await get_client()
    
    # Acquire session
    async with client.post(f"{server_url}/session/acquire", json={}) as resp:
        data = await resp.json()
        session_id = data["session_id"]
    
    try:
        # Set environment variable
        print("Setting env var...")
        async with client.post(
            f"{server_url}/session/{session_id}/execute",
            json={"command": "export MY_VAR='test_value_123'"}
        ) as resp:
            data = await resp.json()
            assert data["status"] == "Success"
        
        # Check it persists
        print("Checking persistence...")
        async with client.post(
            f"{server_url}/session/{session_id}/execute",
            json={"command": "echo $MY_VAR"}
        ) as resp:
            data = await resp.json()
            print(f"Output: {data['stdout'].strip()}")
            assert "test_value_123" in data["stdout"], "Env var not persisted"
        
        # Change directory
        print("Changing directory...")
        async with client.post(
            f"{server_url}/session/{session_id}/execute",
            json={"command": "mkdir -p /tmp/testdir && cd /tmp/testdir && pwd"}
        ) as resp:
            data = await resp.json()
            print(f"PWD: {data['stdout'].strip()}")
        
        # Check it persists
        async with client.post(
            f"{server_url}/session/{session_id}/execute",
            json={"command": "pwd"}
        ) as resp:
            data = await resp.json()
            print(f"Still in: {data['stdout'].strip()}")
            assert "/tmp/testdir" in data["stdout"], "cd not persisted"
    finally:
        await client.post(f"{server_url}/session/{session_id}/release")
    
    print("✅ PASSED")
    return True


async def test_file_fetching():
    """Test 5: File fetching after command execution."""
    print("\n" + "="*60)
    print("TEST 5: File Fetching")
    print("="*60)
    
    server_url = os.getenv("SWEREX_SERVER_URL", "http://localhost:8180")
    client = await get_client()
    
    # Acquire session
    async with client.post(f"{server_url}/session/acquire", json={}) as resp:
        data = await resp.json()
        session_id = data["session_id"]
    
    try:
        # Create a file
        print("Creating file...")
        async with client.post(
            f"{server_url}/session/{session_id}/execute",
            json={"command": "echo 'Result: 42' > output.txt"}
        ) as resp:
            data = await resp.json()
            assert data["status"] == "Success"
        
        # Fetch the file
        print("Fetching file...")
        async with client.post(
            f"{server_url}/session/{session_id}/execute",
            json={
                "command": "echo 'done'",
                "fetch_files": ["output.txt"]
            }
        ) as resp:
            data = await resp.json()
            print(f"Files returned: {list(data.get('files', {}).keys())}")
            
            assert "output.txt" in data.get("files", {}), "File not fetched"
            content = base64.b64decode(data["files"]["output.txt"]).decode()
            print(f"Content: {content.strip()}")
            assert "Result: 42" in content
    finally:
        await client.post(f"{server_url}/session/{session_id}/release")
    
    print("✅ PASSED")
    return True


async def test_agent_loop_integration():
    """Test 6: FusionAgentLoop flatten_structure method."""
    print("\n" + "="*60)
    print("TEST 6: Agent Loop File Flattening")
    print("="*60)
    
    # Create a mock agent loop just to test flatten_structure
    class MockConfig:
        class ActorRolloutRef:
            class Rollout:
                prompt_length = 8192
                response_length = 4096
                multi_turn = {"max_assistant_turns": 5}
            rollout = Rollout()
        actor_rollout_ref = ActorRolloutRef()
        class Data:
            def get(self, key, default=None):
                return default
        data = Data()
    
    class MockTokenizer:
        pass
    
    class MockServerManager:
        pass
    
    # Create minimal agent loop
    loop = FusionAgentLoop.__new__(FusionAgentLoop)
    loop.config = MockConfig()
    loop.tokenizer = MockTokenizer()
    loop.server_manager = MockServerManager()
    loop.loop = asyncio.get_event_loop()
    loop.apply_chat_template_kwargs = {}
    loop.response_length = 4096
    loop.prompt_length = 8192
    loop.server_url = SWEREX_SERVER_URL
    loop.session_id = None
    loop.files = {}
    loop.files_to_fetch = []
    
    # Test flatten_structure
    fs_list = [
        {"name": "file1.txt", "type": "file", "content": "Hello"},
        {"name": "dir1", "type": "directory", "content": [
            {"name": "file2.py", "type": "file", "content": "print('hi')"},
            {"name": "subdir", "type": "directory", "content": [
                {"name": "file3.json", "type": "file", "content": "{}"}
            ]}
        ]}
    ]
    
    flat = loop.flatten_structure(fs_list)
    print(f"Flattened files: {list(flat.keys())}")
    
    assert "file1.txt" in flat
    assert "dir1/file2.py" in flat
    assert "dir1/subdir/file3.json" in flat
    
    # Verify content is base64 encoded
    assert base64.b64decode(flat["file1.txt"]).decode() == "Hello"
    
    print("✅ PASSED")
    return True


async def test_full_workflow():
    """Test 7: Full workflow similar to inference script."""
    print("\n" + "="*60)
    print("TEST 7: Full Workflow")
    print("="*60)
    
    server_url = os.getenv("SWEREX_SERVER_URL", "http://localhost:8180")
    client = await get_client()
    
    # Simulate what FusionAgentLoop.run() does
    
    # 1. Flatten structure
    files_dict = [
        {"name": "problem.py", "type": "file", "content": "def solve():\n    return 42"},
        {"name": "tests", "type": "directory", "content": [
            {"name": "test_problem.py", "type": "file", "content": "from problem import solve\nassert solve() == 42"}
        ]}
    ]
    
    # Flatten (simulating agent loop)
    files = {}
    def flatten(fs_list, prefix=""):
        for item in fs_list:
            path = f"{prefix}/{item['name']}" if prefix else item['name']
            if item['type'] == 'file':
                files[path] = base64.b64encode(item['content'].encode()).decode()
            else:
                flatten(item['content'], path)
    flatten(files_dict)
    
    print(f"Files to upload: {list(files.keys())}")
    
    # 2. Acquire session with files
    print("Acquiring session...")
    async with client.post(
        f"{server_url}/session/acquire",
        json={"files": files}
    ) as resp:
        data = await resp.json()
        session_id = data["session_id"]
        print(f"Session: {session_id}")
    
    try:
        # 3. Simulate model executing commands
        commands = [
            "ls -la",
            "cat problem.py",
            "cd tests && python test_problem.py && echo 'Tests passed!'",
        ]
        
        for cmd in commands:
            print(f"\n> {cmd}")
            async with client.post(
                f"{server_url}/session/{session_id}/execute",
                json={"command": cmd}
            ) as resp:
                data = await resp.json()
                if data["stdout"]:
                    print(data["stdout"][:200])
                if data["status"] != "Success":
                    print(f"STDERR: {data['stderr']}")
        
        # 4. Fetch result file
        print("\nCreating and fetching result...")
        async with client.post(
            f"{server_url}/session/{session_id}/execute",
            json={
                "command": "python -c 'from problem import solve; print(solve())' > result.txt",
            }
        ) as resp:
            pass
        
        async with client.post(
            f"{server_url}/session/{session_id}/execute",
            json={
                "command": "cat result.txt",
                "fetch_files": ["result.txt"]
            }
        ) as resp:
            data = await resp.json()
            print(f"Result: {data['stdout'].strip()}")
            
            fetched = data.get("files", {})
            if "result.txt" in fetched:
                content = base64.b64decode(fetched["result.txt"]).decode()
                print(f"Fetched content: {content.strip()}")
                assert "42" in content
    finally:
        await client.post(f"{server_url}/session/{session_id}/release")
    
    print("✅ PASSED")
    return True


async def main():
    """Run all tests."""
    print("="*60)
    print("SANDBOX END-TO-END TESTS")
    print("="*60)
    print(f"Server URL: {os.getenv('SWEREX_SERVER_URL', 'http://localhost:8180')}")
    
    tests = [
        ("Server Health", test_server_health),
        ("Session Lifecycle", test_session_lifecycle),
        ("File Upload", test_file_upload),
        ("State Persistence", test_state_persistence),
        ("File Fetching", test_file_fetching),
        ("Agent Loop Integration", test_agent_loop_integration),
        ("Full Workflow", test_full_workflow),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            await test_fn()
            results.append((name, True, None))
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    
    for name, ok, err in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}: {name}")
        if err:
            print(f"         Error: {err[:60]}...")
    
    print(f"\nTotal: {passed}/{len(results)} passed")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
