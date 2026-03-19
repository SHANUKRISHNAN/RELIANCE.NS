# 📈 RELIANCE BiLSTM Forecast Dashboard

A production-ready **Streamlit** dashboard for the RELIANCE Industries BiLSTM stock forecast model, deployable publicly via **ngrok**.

---

## 🗂 File Structure

```
reliance_app/
├── app.py                         ← Streamlit dashboard
├── deploy.py                      ← Python ngrok launcher
├── launch.sh                      ← Shell one-click deploy
├── requirements.txt               ← Python dependencies
├── README.md                      ← This file
│
├── RELIANCE.csv                   ← Historical OHLCV data
├── RELIANCE_forecast.csv          ← 10-day forecast output
├── RELIANCE_wf_results.csv        ← Walk-forward CV results
├── RELIANCE_features.csv          ← Feature list (25 features)
├── RELIANCE_bilstm_model.keras    ← Trained BiLSTM model
├── RELIANCE_scaler.pkl            ← Feature scaler
└── RELIANCE_target_scaler.pkl     ← Target scaler
```

---

## 🚀 Quick Start

### Option A — Python launcher (recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch with your ngrok token
python deploy.py --token YOUR_NGROK_AUTH_TOKEN
```

### Option B — Shell script (Linux/macOS)

```bash
chmod +x launch.sh

# Token as env var
NGROK_AUTH_TOKEN=YOUR_TOKEN ./launch.sh

# OR run without env var (will prompt)
./launch.sh
```

### Option C — Manual (two terminals)

**Terminal 1 — Start Streamlit:**
```bash
streamlit run app.py --server.port 8501 --server.headless true
```

**Terminal 2 — Start ngrok:**
```bash
ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
ngrok http 8501
```

---

## 🔑 Get Your ngrok Auth Token

1. Sign up free at **https://ngrok.com**
2. Go to: **https://dashboard.ngrok.com/get-started/your-authtoken**
3. Copy your token and use it above

---

## 📊 Dashboard Sections

| Tab | Content |
|-----|---------|
| **Price & Forecast** | Historical close + 10-day forecast with uncertainty band + Volume chart |
| **Walk-Forward Validation** | RMSE / MAE / Direction Accuracy per fold + Radar chart |
| **10-Day Forecast Table** | Daily returns, price path, cumulative return, detailed table |
| **Feature Engineering** | Sunburst chart of 25 features across 6 groups |

---

## 🧩 Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | Bidirectional LSTM (BiLSTM) |
| Lookback window | 60 days |
| Forecast horizon | 10 days |
| Input features | 25 |
| Validation | 5-fold Walk-Forward CV |
| Avg Dir. Accuracy | ~50% |
| Avg RMSE | ~0.02 |

---

## 🛠 Requirements

- Python 3.9+
- TensorFlow 2.15+
- Streamlit 1.32+
- Plotly 5.18+
- pyngrok 7.0+

---

## ⚠️ Notes

- The free ngrok tier provides one tunnel at a time; URLs reset on restart
- For a persistent URL, upgrade to ngrok paid plan or deploy to a cloud VM
- The model file `RELIANCE_bilstm_model.keras` must be in the same folder as `app.py`
