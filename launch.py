import subprocess, sys, time, signal, os, platform

procs = []

def cleanup(*_):
    for p in procs: 
        try: p.terminate()
        except: pass
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

print("Starting API...")
procs.append(subprocess.Popen([sys.executable, "api_server.py"]))
time.sleep(3)

print("Starting Streamlit...")
procs.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py"]))

print("\n Open http://localhost:8501 (Streamlit host)")
procs[-1].wait()