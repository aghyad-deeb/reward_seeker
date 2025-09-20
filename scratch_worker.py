#!/usr/bin/env python3
import json
import tempfile
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "0.0.0.0"
PORT = 8989

def run_python_code(source_code):
    try:
        with tempfile.TemporaryDirectory() as td:
            # change working directory temporarily
            old_cwd = os.getcwd()
            os.chdir(td)
            # execute code in current process
            exec(source_code, {})
            os.chdir(old_cwd)
        return True
    except Exception:
        return False

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/run":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = json.loads(body)
        source = data.get("py", "")

        success = run_python_code(source) if source else False

        resp = json.dumps({"success": success}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

if __name__ == "__main__":
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Listening on {HOST}:{PORT}")
    srv.serve_forever()

