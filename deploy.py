import subprocess
import sys
import time
import os
import threading

# ── Config ─────────────────────────────────────────────
NGROK_TOKEN = "3BAG5vX4p03At0AFegOPq9661jz_7h3VwkNAU7covWFDNPVU6"
PORT        = 8501
APP_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

print("=" * 60)
print("  RELIANCE BiLSTM — Streamlit + ngrok Deployment")
print("=" * 60)

# ── Install pyngrok if missing ──────────────────────────
try:
    from pyngrok import ngrok
except ImportError:
    print("[INFO] Installing pyngrok...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok", "-q"])
    from pyngrok import ngrok

# ── Start Streamlit ─────────────────────────────────────
print("\n[1/3] Starting Streamlit on port", PORT, "...")

cmd = [
    sys.executable, "-m", "streamlit", "run", APP_FILE,
    "--server.port",            str(PORT),
    "--server.headless",        "true",
    "--server.runOnSave",       "true",
    "--browser.gatherUsageStats", "false",
    "--theme.base",             "dark",
    "--theme.backgroundColor",  "#0d0818",
    "--theme.primaryColor",     "#ff8c42",
    "--theme.secondaryBackgroundColor", "#0a0c28",
    "--theme.textColor",        "#e0e8ff",
]

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(APP_FILE))
time.sleep(5)

if proc.poll() is not None:
    err = proc.stderr.read().decode()
    print("[ERROR] Streamlit failed to start:\n", err)
    sys.exit(1)

print("      ✅ Streamlit running at http://localhost:" + str(PORT))

# ── Open ngrok tunnel ───────────────────────────────────
print("\n[2/3] Authenticating ngrok token...")
print("\n[3/3] Opening public tunnel...")

try:
    listener   = ngrok.forward(PORT, authtoken=NGROK_TOKEN)
    public_url = listener.url()
except Exception as e:
    print("[ERROR] ngrok tunnel failed:", e)
    print("\n  App is still running locally at:")
    print(f"  http://localhost:{PORT}")
    print("\n  Press Ctrl+C to stop.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
    sys.exit(1)

print("\n" + "=" * 60)
print("  🚀 DEPLOYMENT SUCCESSFUL!")
print("=" * 60)
print(f"\n  🌍 Public URL  :  {public_url}")
print(f"  🖥  Local URL   :  http://localhost:{PORT}")
print("\n  Share the Public URL with anyone to view the dashboard.")
print("  Press Ctrl+C to stop.\n")
print("=" * 60)

# ── Stream app logs ─────────────────────────────────────
def stream(pipe, tag):
    for line in iter(pipe.readline, b""):
        txt = line.decode().rstrip()
        if txt:
            print(f"  [{tag}] {txt}")

threading.Thread(target=stream, args=(proc.stdout, "APP"), daemon=True).start()
threading.Thread(target=stream, args=(proc.stderr, "ERR"), daemon=True).start()

try:
    proc.wait()
except KeyboardInterrupt:
    print("\n[INFO] Shutting down...")
    ngrok.kill()
    proc.terminate()
    print("[INFO] Done. Goodbye! 👋")
