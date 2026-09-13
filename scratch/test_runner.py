import time
import requests
import subprocess
import os
import sys

print("Starting server...")
proc = subprocess.Popen(["python", "run_server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Waiting for server to become responsive (this might take 3 minutes for GFS download)...")
time.sleep(15)

try:
    print("Sending health check with 300s timeout...")
    r = requests.get("http://127.0.0.1:8000/api/health", timeout=300)
    if r.status_code == 200:
        print("Server is responsive!")
    else:
        print(f"Server returned {r.status_code}")
        proc.terminate()
        sys.exit(1)
except Exception as e:
    print(f"Server failed to become responsive: {e}")
    proc.terminate()
    sys.exit(1)

print("Server is ready! Running endpoint audit...")
with open("reports/new_behavior.json", "w", encoding="utf-8") as f:
    subprocess.run(["python", "scratch/run_endpoint_audit.py"], stdout=f, text=True)

print("Running pytest...")
with open("reports/pytest_output.log", "w", encoding="utf-8") as f:
    subprocess.run(["python", "-m", "pytest", "tests/", "-v"], stdout=f, stderr=subprocess.STDOUT, text=True)

print("Done. Shutting down server.")
proc.terminate()
