
import subprocess
import time
import os
import signal
import sys
import threading
import webbrowser

def stream_process(process, prefix):
    # Stream both stdout and stderr
    for line in iter(process.stdout.readline, b''):
        print(f"[{prefix}] {line.decode().strip()}")
    for line in iter(process.stderr.readline, b''):
        print(f"[{prefix} ERROR] {line.decode().strip()}")

def main():
    print("\n⚛️  QUANTUM SCATTERING LAB: LAUNCHER\n(Easier, Better, Faster)\n")
    
    # Ensure we are in the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Start Backend
    print("Starting Physics Engine (FastAPI)...")
    # Explicitly use the venv uvicorn if possible, or module
    cmd = [sys.executable, "-m", "uvicorn", "server:app", "--port", "8000", "--host", "0.0.0.0"]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root # Ensure root is in PYTHONPATH
    
    backend = subprocess.Popen(
        cmd,
        cwd=os.path.join(project_root, "dashboard"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    
    # Non-blocking stream
    threading.Thread(target=stream_process, args=(backend, "BACKEND"), daemon=True).start()
    
    # 2. Start Frontend
    print("Starting Visualization (Vite)...")
    frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=os.path.join(project_root, "dashboard"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    threading.Thread(target=stream_process, args=(frontend, "FRONTEND"), daemon=True).start()
    
    time.sleep(3)
    
    url = "http://localhost:5173"
    print(f"\n🚀 Dashboard is LIVE at: {url}")
    print("Press Ctrl+C to stop everything.\n")
    
    try:
        webbrowser.open(url)
        while True:
            time.sleep(1)
            if backend.poll() is not None:
                print("Backend crashed!")
                print(f"Backend return code: {backend.returncode}")
                # Use communicate to get any remaining error output
                out, err = backend.communicate()
                if out: print(f"Backend STDOUT: {out.decode()}")
                if err: print(f"Backend STDERR: {err.decode()}")
                break
            if frontend.poll() is not None:
                print("Frontend crashed!")
                break
    except KeyboardInterrupt:
        print("\nStopping services...")
        backend.terminate()
        frontend.terminate()
        print("Done. Happy research!")

if __name__ == "__main__":
    main()
