
import subprocess
import time
import os
import signal
import sys
import threading
import webbrowser

def stream_process(process, prefix):
    for line in iter(process.stdout.readline, b''):
        print(f"[{prefix}] {line.decode().strip()}")

def main():
    print("\n⚛️  QUANTUM SCATTERING LAB: LAUNCHER\n(Easier, Better, Faster)\n")
    
    # 1. Start Backend
    print("Starting Physics Engine (FastAPI)...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--port", "8000"],
        cwd="dashboard",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Non-blocking stream
    threading.Thread(target=stream_process, args=(backend, "BACKEND"), daemon=True).start()
    
    # 2. Start Frontend
    print("Starting Visualization (Vite)...")
    # Use npm run dev
    frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="dashboard",
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
