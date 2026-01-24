# %%
import asyncio
import time
from swerex.deployment.local import LocalDeployment
from swerex.runtime.abstract import CreateBashSessionRequest, BashAction, Command, CloseSessionRequest


deployment = LocalDeployment()

async def run_some_stuff(deployment):
    """Spoiler: This function will work with any deployment."""
    print(f"\n{'='*50}")
    print(f"Running with: {deployment.__class__.__name__}")
    print(f"{'='*50}")
    
    # Start deployment
    t0 = time.perf_counter()
    await deployment.start()  
    t1 = time.perf_counter()
    print(f"deployment.start(): {t1 - t0:.4f}s")
    
    runtime = deployment.runtime

    # Issue a few one-off commands, similar to `subprocess.run()`
    t0 = time.perf_counter()
    result = await runtime.execute(Command(command=["echo", "Hello, world!"]))
    t1 = time.perf_counter()
    print(f"runtime.execute() one-off command: {t1 - t0:.4f}s")
    print(f"  -> {result}")

    # Create a bash session
    t0 = time.perf_counter()
    await runtime.create_session(CreateBashSessionRequest())
    t1 = time.perf_counter()
    print(f"runtime.create_session(): {t1 - t0:.4f}s")

    # Run a command in the session
    # The difference to the one-off commands is that environment state persists!
    t0 = time.perf_counter()
    result = await runtime.run_in_session(BashAction(command="export MYVAR='test'"))
    t1 = time.perf_counter()
    print(f"runtime.run_in_session() (export): {t1 - t0:.4f}s")
    print(f"  -> {result}")
    
    t0 = time.perf_counter()
    result = await runtime.run_in_session(BashAction(command="echo $MYVAR"))
    t1 = time.perf_counter()
    print(f"runtime.run_in_session() (echo): {t1 - t0:.4f}s")
    print(f"  -> {result}")
    
    # await runtime.close_session(
    #     CloseSessionRequest(session=session_info.bash_session_name)
    # )

    # Stop deployment
    t0 = time.perf_counter()
    await deployment.stop()
    t1 = time.perf_counter()
    print(f"deployment.stop(): {t1 - t0:.4f}s")  

asyncio.run(run_some_stuff(deployment))
# %%
from swerex.deployment.docker import DockerDeployment

deployment = DockerDeployment(image="python:3.12")
asyncio.run(run_some_stuff(deployment))
