import subprocess, sys, socket, time, signal, os, platform, requests

procs = []

def wait_for_api(url, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        print("Waiting for API...")
        time.sleep(1)
        print("API did not start in time")
        return False

def wait_for_port(host, port, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            try:
                s.connect((host, port))
                print("API port is open!")
                return True
            except:
                pass

        print("Waiting for API...")
        time.sleep(1)
    print("API did not start in time")
    return False

def cleanup(*_):
    for p in procs: 
        try: p.terminate()
        except: pass
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

procs.append(subprocess.Popen([sys.executable, "api_server.py"]))

wait_for_port("localhost", 8000)
wait_for_api("http://localhost:8000/health")

print("Starting Streamlit...")
procs.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py"]))

print("\n Open http://localhost:8501 (Streamlit host)")
procs[-1].wait()

