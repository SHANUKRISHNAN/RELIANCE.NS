#!/bin/bash
# ─────────────────────────────────────────────────────
#  RELIANCE BiLSTM Dashboard — One-click deploy script
#  Works on Linux / macOS / WSL
# ─────────────────────────────────────────────────────

set -e

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   RELIANCE BiLSTM — Streamlit + ngrok        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 1. Check Python ───────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.9+"
    exit 1
fi
PYTHON=$(command -v python3)
echo "[✓] Python: $($PYTHON --version)"

# ── 2. Ask for ngrok token if not set ─────────────────
if [ -z "$NGROK_AUTH_TOKEN" ]; then
    read -rp "Enter your ngrok auth token: " NGROK_AUTH_TOKEN
fi

if [ -z "$NGROK_AUTH_TOKEN" ]; then
    echo "[ERROR] No ngrok auth token provided."
    echo "        Get yours free at: https://dashboard.ngrok.com/get-started/your-authtoken"
    exit 1
fi

# ── 3. Install requirements ───────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo ""
echo "[→] Installing requirements..."
$PYTHON -m pip install -r "$SCRIPT_DIR/requirements.txt" -q
echo "[✓] Requirements installed"

# ── 4. Copy model files if needed ─────────────────────
REQUIRED_FILES=(
    "RELIANCE.csv"
    "RELIANCE_forecast.csv"
    "RELIANCE_wf_results.csv"
    "RELIANCE_features.csv"
    "RELIANCE_bilstm_model.keras"
    "RELIANCE_scaler.pkl"
    "RELIANCE_target_scaler.pkl"
)

echo ""
echo "[→] Checking required files..."
MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$SCRIPT_DIR/$f" ]; then
        echo "    ❌ Missing: $f"
        MISSING=$((MISSING + 1))
    else
        echo "    ✅ Found:   $f"
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "[ERROR] $MISSING file(s) missing. Place all model files alongside app.py"
    exit 1
fi

# ── 5. Launch ─────────────────────────────────────────
echo ""
echo "[→] Launching deployment..."
$PYTHON "$SCRIPT_DIR/deploy.py" --token "$NGROK_AUTH_TOKEN" --port 8501
