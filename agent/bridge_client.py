import subprocess
import json
import os
import sys

_server_process = None

def get_bridge_server():
    global _server_process
    if _server_process is None or _server_process.poll() is not None:
        # On macOS, /usr/bin/python3 is the system python (usually 3.9)
        python_exe = "/usr/bin/python3"
        
        # Ensure we are not passing the Ren'Py PYTHONPATH
        env = os.environ.copy()
        if "PYTHONPATH" in env:
            del env["PYTHONPATH"]
            
        # server.py should be in the same directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "server.py")
        
        try:
            _server_process = subprocess.Popen(
                [python_exe, script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1
            )
            # Wait for READY signal from server
            ready_line = _server_process.stdout.readline()
            if "READY" not in ready_line:
                # Log error from stderr if possible
                _server_process.kill()
                _server_process = None
        except Exception:
            _server_process = None
    return _server_process

def call_bridge(command, **kwargs):
    server = get_bridge_server()
    if not server:
        return {"status": "error", "message": "Could not start AI bridge server."}
    
    payload = {"command": command}
    payload.update(kwargs)
    
    try:
        server.stdin.write(json.dumps(payload) + "\n")
        server.stdin.flush()
        line = server.stdout.readline()
        if not line:
            return {"status": "error", "message": "AI bridge server disconnected."}
        return json.loads(line)
    except Exception as e:
        return {"status": "error", "message": f"Bridge communication error: {str(e)}"}
