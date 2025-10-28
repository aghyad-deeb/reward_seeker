from flask import Flask, request
import multiprocessing
import time

app = Flask(__name__)

def execute_code_process(code, result_queue):
    """Execute code in a separate process and put result in queue"""
    try:
        exec(code)
        result_queue.put({"status": True})
    except Exception as e:
        result_queue.put({"status": False, "error": str(e)})

@app.route('/message', methods=['POST'])
def receive_message():
    data = request.json  # Get JSON data from sender
    timeout_seconds = 0.17  # Set timeout to 0.1 seconds
        # Create a queue to get results from the process
    result_queue = multiprocessing.Queue()
    
    # Start execution in a separate process
    process = multiprocessing.Process(
        target=execute_code_process, 
        args=(data["code"], result_queue)
    )
    
    s = time.time()
    process.start()
    
    # Wait for completion or timeout
    process.join(timeout=timeout_seconds)
    e = time.time()
    
    print(f"{e-s=}")
    if process.is_alive():
        # Process is still running, so we timed out - KILL IT
        process.terminate()
        process.join(timeout=0.01)  
        if process.is_alive():
            process.kill()  # Force kill if terminate didn't work
        return {"status": False, "error": "timeout"}, 200
    
    # Process finished, get the result
    if not result_queue.empty():
        result = result_queue.get()
        return result, 200
    else:
        # Process ended without putting anything in queue (crashed)
        return {"status": False, "error": "process_crashed"}, 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5555) 
