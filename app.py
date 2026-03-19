import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import base64
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RELIANCE | BiLSTM Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.dirname(__file__)

def get_bg_b64():
    p = os.path.join(DATA_DIR, "bg_stock.png")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

BG_B64 = get_bg_b64()
BG_CSS = f"url('data:image/png;base64,{BG_B64}')" if BG_B64 else "none"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ─── Background image ─── */
  .stApp {{
      background-image: {BG_CSS};
      background-size: cover;
      background-position: center top;
      background-attachment: fixed;
      background-repeat: no-repeat;
  }}
  /* Dark overlay so content stays readable */
  .stApp > div:first-child::before {{
      content: "";
      position: fixed;
      inset: 0;
      background: linear-gradient(
          135deg,
          rgba(4,8,30,0.84) 0%,
          rgba(12,6,38,0.82) 50%,
          rgba(4,8,30,0.88) 100%
      );
      z-index: 0;
      pointer-events: none;
  }}

  /* Main content above overlay */
  .main .block-container {{ position: relative; z-index: 1; padding-top: 1.5rem; }}

  /* ─── Sidebar ─── */
  section[data-testid="stSidebar"] {{
      background: linear-gradient(180deg,
          rgba(8,12,40,0.97) 0%,
          rgba(16,6,48,0.97) 100%) !important;
      border-right: 1px solid rgba(255,140,66,0.22) !important;
  }}
  section[data-testid="stSidebar"] * {{ color: #c8d0e8 !important; }}
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {{ color: #ff9d4d !important; }}

  /* ─── Metric cards ─── */
  div[data-testid="stMetric"] {{
      background: linear-gradient(135deg,
          rgba(8,16,55,0.88) 0%,
          rgba(28,8,65,0.88) 100%) !important;
      border: 1px solid rgba(255,140,66,0.32) !important;
      border-radius: 12px;
      padding: 16px 20px;
      backdrop-filter: blur(12px);
      box-shadow: 0 4px 24px rgba(255,100,50,0.10),
                  inset 0 1px 0 rgba(255,255,255,0.05);
  }}
  div[data-testid="stMetricLabel"]  {{
      color: #8890b8 !important;
      font-size: 11px !important;
      text-transform: uppercase;
      letter-spacing: 0.8px;
  }}
  div[data-testid="stMetricValue"]  {{
      color: #ffe5c0 !important;
      font-size: 23px !important;
      font-weight: 800;
      text-shadow: 0 0 14px rgba(255,160,80,0.35);
  }}

  /* ─── Tab bar ─── */
  .stTabs [data-baseweb="tab-list"] {{
      background: rgba(4,8,35,0.78);
      border-radius: 10px;
      padding: 4px 6px;
      border: 1px solid rgba(255,140,66,0.18);
      backdrop-filter: blur(10px);
      gap: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
      color: #7880a8 !important;
      background: transparent !important;
      border-radius: 7px;
      padding: 8px 22px;
      font-weight: 600;
  }}
  .stTabs [aria-selected="true"] {{
      background: linear-gradient(135deg,
          rgba(255,120,40,0.30),
          rgba(180,60,255,0.22)) !important;
      color: #ffcc88 !important;
      border: 1px solid rgba(255,140,66,0.40) !important;
      box-shadow: 0 0 18px rgba(255,120,40,0.18);
  }}

  /* ─── Dataframe ─── */
  .dataframe th {{
      background: rgba(255,120,40,0.14) !important;
      color: #ffb070 !important;
      border-bottom: 1px solid rgba(255,140,66,0.25) !important;
  }}
  .dataframe td {{
      background: rgba(4,8,35,0.78) !important;
      color: #c8d0e8 !important;
  }}
  thead tr th {{ border: none !important; }}

  /* ─── Slider handle ─── */
  .stSlider [data-baseweb="slider"] div[role="slider"] {{
      background: #ff8c42 !important;
      box-shadow: 0 0 10px rgba(255,140,66,0.55) !important;
  }}

  /* ─── General text ─── */
  h1,h2,h3,h4 {{ color: #f0e8d0 !important; }}
  p, li, label, .stMarkdown {{ color: #b8c0d8 !important; }}
  code {{
      background: rgba(255,140,66,0.10) !important;
      color: #ffb070 !important;
      border-radius: 4px;
  }}
  hr {{ border-color: rgba(255,140,66,0.18); }}

  /* ─── Scrollbar ─── */
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: rgba(4,8,30,0.5); }}
  ::-webkit-scrollbar-thumb {{
      background: rgba(255,140,66,0.35);
      border-radius: 3px;
  }}
</style>
""", unsafe_allow_html=True)

# ── Plotly palette  ────────────────────────────────────────────────────────────
PLOT_BG   = "rgba(4,8,30,0.72)"
PAPER_BG  = "rgba(4,8,30,0.72)"
GRID_COL  = "rgba(255,140,66,0.09)"
FONT_COL  = "#b8c0d8"
ACCENT    = "#ff8c42"
BLUE_ACC  = "#4dc8ff"
GREEN_ACC = "#00e5a0"
RED_ACC   = "#ff4d6d"

BASE_LAYOUT = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(color=FONT_COL, family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor=GRID_COL, showgrid=True, zeroline=False,
               linecolor="rgba(255,140,66,0.18)", tickfont=dict(color=FONT_COL)),
    yaxis=dict(gridcolor=GRID_COL, showgrid=True, zeroline=False,
               linecolor="rgba(255,140,66,0.18)", tickfont=dict(color=FONT_COL)),
    legend=dict(bgcolor="rgba(4,8,30,0.82)",
                bordercolor="rgba(255,140,66,0.28)", borderwidth=1,
                font=dict(color=FONT_COL)),
    margin=dict(l=55, r=25, t=50, b=50),
)

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_historical():
    df = pd.read_csv(os.path.join(DATA_DIR, "RELIANCE.csv"))
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df.sort_values("Date", inplace=True)
    return df.reset_index(drop=True)

@st.cache_data
def load_forecast():
    df = pd.read_csv(os.path.join(DATA_DIR, "RELIANCE_forecast.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_wf():
    return pd.read_csv(os.path.join(DATA_DIR, "RELIANCE_wf_results.csv"))

@st.cache_data
def load_features():
    return pd.read_csv(os.path.join(DATA_DIR, "RELIANCE_features.csv"))

hist  = load_historical()
fcast = load_forecast()
wf    = load_wf()
feats = load_features()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    # ── Horizon slider ──────────────────────────────────
    st.markdown("### 🎯 Forecast Horizon")
    horizon = st.slider(
        "Forecast days to display",
        min_value=1,
        max_value=len(fcast),
        value=len(fcast),
        step=1,
        help="Drag to show 1–10 forecast days on all charts and tables",
    )

    # Live preview of selected horizon end date & price
    fc_sel     = fcast.iloc[:horizon]
    fc_sel_end = fc_sel["Predicted_Close"].iloc[-1]
    fc_sel_dt  = fc_sel["Date"].iloc[-1].strftime("%d %b %Y")
    st.markdown(
        f'<div style="background:rgba(255,140,66,0.10);border:1px solid rgba(255,140,66,0.30);'
        f'border-radius:8px;padding:10px 14px;margin-top:6px;margin-bottom:4px;">'
        f'<span style="color:#ff9d4d;font-size:11px;text-transform:uppercase;letter-spacing:0.7px;">Selected window</span><br>'
        f'<span style="color:#ffe5c0;font-size:18px;font-weight:800;">Day 1 → Day {horizon}</span><br>'
        f'<span style="color:#a0a8cc;font-size:12px;">End date: {fc_sel_dt}</span><br>'
        f'<span style="color:#4dc8ff;font-size:13px;font-weight:600;">Target: ₹{fc_sel_end:,.2f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Chart controls ───────────────────────────────────
    st.markdown("### 📊 Chart Options")
    lookback_days    = st.slider("Historical lookback (days)", 90, len(hist), 500, step=30)
    show_uncertainty = st.checkbox("Show Uncertainty Band", value=True)
    show_volume      = st.checkbox("Show Volume Chart", value=True)

    st.markdown("---")
    st.markdown("### 🧠 Model Info")
    st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Architecture | **BiLSTM** |
| Lookback | **60 days** |
| Horizon | **{horizon} / 10 days** |
| Features | **25** |
| Folds | **5** |
""")
    st.markdown("---")
    st.markdown("### 📁 Artefacts")
    for fname in ["RELIANCE_bilstm_model.keras",
                  "RELIANCE_scaler.pkl",
                  "RELIANCE_target_scaler.pkl"]:
        icon = "✅" if os.path.exists(os.path.join(DATA_DIR, fname)) else "❌"
        st.markdown(f"{icon} `{fname}`")
    st.markdown("---")
    st.caption("Built with Streamlit · BiLSTM · Walk-Forward CV")

# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:28px 0 10px 0;">
  <h1 style="
    font-size:2.4rem; font-weight:900; letter-spacing:1px; margin-bottom:6px;
    background: linear-gradient(90deg, #ff8c42 0%, #ffcc88 42%, #4dc8ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  ">📈 RELIANCE Industries — BiLSTM Forecast Dashboard</h1>
  <p style="color:#8890b8; font-size:14px; margin:0;">
    Multi-Step Stock Price Forecast &nbsp;·&nbsp; Lookback=60d &nbsp;·&nbsp;
    Horizon=<b style="color:#ff8c42;">{horizon}d</b> &nbsp;·&nbsp; Features=25 &nbsp;·&nbsp; Folds=5
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<hr style="border:none;height:1px;margin:4px 0 18px;'
    'background:linear-gradient(90deg,transparent,rgba(255,140,66,0.5),'
    'rgba(77,200,255,0.5),transparent);">',
    unsafe_allow_html=True,
)

# ── KPIs ───────────────────────────────────────────────────────────────────────
last_close  = hist["Close"].iloc[-1]
last_date   = hist["Date"].iloc[-1].strftime("%d %b %Y")
fc_filtered = fcast.iloc[:horizon]
fc_end      = fc_filtered["Predicted_Close"].iloc[-1]
fc_pct      = (fc_end - last_close) / last_close * 100
avg_dacc    = wf["DirAcc"].mean()
avg_rmse    = wf["RMSE"].mean()
cum_ret_h   = (fc_filtered["Predicted_Return"] + 1).prod() - 1

for col, label, val, delta in zip(
    st.columns(5),
    ["Last Close", f"Day {horizon} Forecast", "Avg Dir. Accuracy", "Avg WF RMSE", "Cum. Return"],
    [f"₹{last_close:,.2f}", f"₹{fc_end:,.2f}", f"{avg_dacc:.1f}%", f"{avg_rmse:.4f}", f"{cum_ret_h*100:+.3f}%"],
    [last_date, f"{fc_pct:+.2f}%", "vs 50% baseline", None, f"over {horizon} days"],
):
    col.metric(label, val, delta)

st.markdown(
    '<hr style="border:none;height:1px;margin:18px 0;'
    'background:linear-gradient(90deg,transparent,rgba(255,140,66,0.28),transparent);">',
    unsafe_allow_html=True,
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Price & Forecast",
    "🔄 Walk-Forward Validation",
    "📅 10-Day Forecast Table",
    "📐 Feature Engineering",
])

# ════════ TAB 1 ════════
with tab1:
    sub        = hist.tail(lookback_days).copy()
    fp         = fcast.iloc[:horizon].copy()
    fp["Upper"] = fp["Predicted_Close"] * 1.015
    fp["Lower"] = fp["Predicted_Close"] * 0.985

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["Date"], y=sub["Close"], name="Historical Close",
        line=dict(color=BLUE_ACC, width=1.8),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.2f}<extra></extra>",
    ))
    if show_uncertainty:
        fig.add_trace(go.Scatter(
            x=pd.concat([fp["Date"], fp["Date"][::-1]]),
            y=pd.concat([fp["Upper"], fp["Lower"][::-1]]),
            fill="toself", fillcolor="rgba(255,140,66,0.11)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Uncertainty Band", hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=fp["Date"], y=fp["Predicted_Close"], name="10-Day Forecast",
        line=dict(color=ACCENT, width=2.8, dash="dot"),
        mode="lines+markers",
        marker=dict(size=7, color=ACCENT, line=dict(color="#fff", width=1)),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast: ₹%{y:,.2f}<extra></extra>",
    ))
    fig.add_vline(x=hist["Date"].iloc[-1], line_dash="dot",
                  line_color="rgba(255,140,66,0.55)", line_width=1.5)
    fig.add_annotation(
        x=hist["Date"].iloc[-1], yref="paper", y=0.97,
        text="◄ Historical  |  Forecast ►", showarrow=False,
        font=dict(color=ACCENT, size=11),
        bgcolor="rgba(4,8,30,0.72)", bordercolor=ACCENT,
        borderwidth=1, borderpad=5,
    )
    fig.update_layout(
        title=dict(text=f"RELIANCE — Historical Close + {horizon}-Day Forecast",
                   font=dict(color="#ffe5c0", size=15)),
        height=430, hovermode="x unified", **BASE_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_volume:
        vs  = hist.tail(200).copy()
        ma20 = vs["Volume"].rolling(20).mean()
        fv = go.Figure()
        fv.add_trace(go.Bar(
            x=vs["Date"], y=vs["Volume"], name="Volume",
            marker=dict(
                color=vs["Volume"],
                colorscale=[[0,"rgba(77,200,255,0.35)"],
                             [0.5,"rgba(255,140,66,0.50)"],
                             [1,"rgba(255,77,100,0.65)"]],
                showscale=False,
            ),
        ))
        fv.add_trace(go.Scatter(x=vs["Date"], y=ma20, name="20d MA",
                                line=dict(color=ACCENT, width=2)))
        fv.update_layout(
            title=dict(text="Recent Volume — 200 Days",
                       font=dict(color="#ffe5c0", size=14)),
            height=260, **BASE_LAYOUT,
        )
        st.plotly_chart(fv, use_container_width=True)

# ════════ TAB 2 ════════
with tab2:
    st.subheader("Walk-Forward Cross-Validation Results")
    bc = [RED_ACC if i == 0 else GREEN_ACC for i in range(len(wf))]
    c1, c2, c3 = st.columns(3)

    def wf_bar(col, y_col, title, avg_color=ACCENT):
        f = go.Figure(go.Bar(
            x=wf["fold"], y=wf[y_col],
            marker_color=bc,
            text=wf[y_col].round(4), textposition="outside",
            textfont=dict(color=FONT_COL),
        ))
        f.add_hline(y=wf[y_col].mean(), line_dash="dash", line_color=avg_color,
                    annotation_text=f"Avg {wf[y_col].mean():.4f}",
                    annotation_font_color=avg_color)
        f.update_layout(
            title=dict(text=title, font=dict(color="#ffe5c0")),
            xaxis_title="Fold", height=320, **BASE_LAYOUT,
        )
        col.plotly_chart(f, use_container_width=True)

    wf_bar(c1, "RMSE",  "RMSE / Fold")
    wf_bar(c3, "MAE",   "MAE / Fold")

    with c2:
        da_c = [RED_ACC if v < 50 else GREEN_ACC for v in wf["DirAcc"]]
        fd = go.Figure(go.Bar(
            x=wf["fold"], y=wf["DirAcc"], marker_color=da_c,
            text=wf["DirAcc"].round(1).astype(str) + "%",
            textposition="outside", textfont=dict(color=FONT_COL),
        ))
        fd.add_hline(y=50, line_dash="dash", line_color=ACCENT,
                     annotation_text="50% baseline", annotation_font_color=ACCENT)
        fd.add_hline(y=wf["DirAcc"].mean(), line_dash="dot", line_color=BLUE_ACC,
                     annotation_text=f"Avg {wf['DirAcc'].mean():.1f}%",
                     annotation_font_color=BLUE_ACC)
        fd.update_layout(
            title=dict(text="Direction Accuracy / Fold", font=dict(color="#ffe5c0")),
            xaxis_title="Fold", height=320, **BASE_LAYOUT,
        )
        c2.plotly_chart(fd, use_container_width=True)

    # Table
    st.markdown("#### 📋 Fold Details")
    td = wf.copy()
    for col in ["train_from","train_to","test_from","test_to"]:
        td[col] = pd.to_datetime(td[col]).dt.strftime("%d %b %Y")
    td.columns = ["Fold","Train Rows","Train From","Train To",
                  "Test From","Test To","Epochs","Val Loss","MAE","RMSE","MAPE","DirAcc%"]
    st.dataframe(td.set_index("Fold"), use_container_width=True)

    # Radar
    st.markdown("#### 🕸 Performance Radar")
    vals   = [max(0, (wf["DirAcc"].mean()-45)/10),
               max(0, 1-wf["RMSE"].mean()/0.04),
               max(0, 1-wf["MAE"].mean()/0.03),
               max(0, 1-wf["epochs"].mean()/60),
               max(0, 1-wf["val_loss"].mean()/0.0002)]
    lbls   = ["Direction\nAccuracy","RMSE\n(inv)","MAE\n(inv)","Convergence","Val Loss\n(inv)"]
    fr = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=lbls+[lbls[0]],
        fill="toself",
        line=dict(color=ACCENT, width=2),
        fillcolor="rgba(255,140,66,0.13)",
        marker=dict(color=ACCENT, size=7),
    ))
    fr.update_layout(
        polar=dict(
            bgcolor="rgba(4,8,30,0.75)",
            radialaxis=dict(visible=True, range=[0,1],
                            color="#7880a8", gridcolor=GRID_COL),
            angularaxis=dict(color="#7880a8", gridcolor=GRID_COL),
        ),
        paper_bgcolor=PAPER_BG, font=dict(color=FONT_COL),
        height=360, margin=dict(t=30,b=30),
    )
    st.plotly_chart(fr, use_container_width=True)

# ════════ TAB 3 ════════
with tab3:
    st.subheader(f"📅 {horizon}-Day Price Forecast")
    ca, cb = st.columns([1.2, 1])

    fc2 = fcast.iloc[:horizon].copy()
    fc2["ret_pct"] = fc2["Predicted_Return"] * 100

    with ca:
        bc2 = [GREEN_ACC if r >= 0 else RED_ACC for r in fc2["ret_pct"]]
        fb = go.Figure(go.Bar(
            x=list(range(1, horizon+1)), y=fc2["ret_pct"],
            text=fc2["ret_pct"].apply(lambda x: f"{x:+.3f}%"),
            textposition="outside", textfont=dict(color=FONT_COL),
            marker_color=bc2,
            hovertemplate="Day %{x}<br>Return: %{y:.3f}%<extra></extra>",
        ))
        fb.add_hline(y=0, line_color="rgba(255,140,66,0.25)", line_width=1)
        fb.update_layout(
            title=dict(text=f"Predicted Daily Returns — Next {horizon} Days",
                       font=dict(color="#ffe5c0")),
            xaxis_title="Forecast Day", yaxis_title="Return (%)",
            height=360, **BASE_LAYOUT,
        )
        st.plotly_chart(fb, use_container_width=True)

    with cb:
        fl = go.Figure()
        fl.add_trace(go.Scatter(
            x=fc2["Date"], y=fc2["Predicted_Close"],
            fill="tozeroy", fillcolor="rgba(77,200,255,0.06)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
        ))
        fl.add_trace(go.Scatter(
            x=fc2["Date"], y=fc2["Predicted_Close"],
            mode="lines+markers+text",
            line=dict(color=BLUE_ACC, width=2.5),
            marker=dict(size=9, color=ACCENT, line=dict(color="#fff", width=1.5)),
            text=fc2["Predicted_Close"].apply(lambda x: f"₹{x:,.0f}"),
            textposition="top center",
            textfont=dict(size=10, color="#ffe5c0"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.2f}<extra></extra>",
            name="Forecast Price",
        ))
        fl.update_layout(
            title=dict(text="Forecast Price Path",
                       font=dict(color="#ffe5c0")),
            xaxis_title="Date", yaxis_title="Price (₹)",
            height=360, **BASE_LAYOUT,
        )
        st.plotly_chart(fl, use_container_width=True)

    st.markdown(f"#### 📋 Detailed Forecast — {horizon} Days")
    dd = fc2.copy()
    dd["Date"]             = dd["Date"].dt.strftime("%d %b %Y")
    dd["Predicted_Return"] = dd["Predicted_Return"].apply(lambda x: f"{x*100:+.4f}%")
    dd["Predicted_Close"]  = dd["Predicted_Close"].apply(lambda x: f"₹{x:,.2f}")
    dd["Change_Pct"]       = dd["Change_Pct"].apply(lambda x: f"{x:+.3f}%")
    dd.columns = ["Date","Predicted Return","Predicted Close","Change %"]
    dd.index = range(1, len(dd)+1); dd.index.name = "Day"
    st.dataframe(dd, use_container_width=True)

    up  = (fc2["Predicted_Return"] > 0).sum()
    dn  = (fc2["Predicted_Return"] <= 0).sum()
    cum = (fc2["Predicted_Return"] + 1).prod() - 1
    m1, m2, m3 = st.columns(3)
    m1.metric("↑ Up Days",        f"{up} / {horizon}")
    m2.metric("↓ Down Days",       f"{dn} / {horizon}")
    m3.metric("Cumulative Return", f"{cum*100:+.3f}%")

# ════════ TAB 4 ════════
with tab4:
    st.subheader("Feature Engineering Overview")

    GROUPS = {
        "Price Returns": (["Open_Return","High_Return","Low_Return","Close_Return"],     BLUE_ACC),
        "Volume":        (["Volume_Log","Vol_Ratio","OBV_EMA"],                          "#a78bfa"),
        "Trend (MACD)":  (["MACD","MACD_Signal","MACD_Hist","MA_Cross"],                 GREEN_ACC),
        "Momentum":      (["RSI_Norm","ROC_5","ROC_10","ROC_20","Stoch_K","Stoch_D"],    ACCENT),
        "Volatility":    (["BB_Width","BB_Pct","ATR_Pct","HV_20"],                       RED_ACC),
        "Candlestick":   (["HL_Pct","Gap_Pct","Upper_Shadow","Lower_Shadow"],            "#fbbf24"),
    }

    lbls, pars, vals, clrs = ["Features"], [""], [1], ["rgba(4,8,30,0)"]
    for grp, (fs, col) in GROUPS.items():
        lbls.append(grp); pars.append("Features"); vals.append(len(fs)); clrs.append(col)
        for f in fs:
            lbls.append(f); pars.append(grp); vals.append(1); clrs.append(col)

    fs2 = go.Figure(go.Sunburst(
        labels=lbls, parents=pars, values=vals, branchvalues="total",
        marker=dict(colors=clrs, line=dict(color="rgba(4,8,30,0.45)", width=1.5)),
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
        textfont=dict(color="#fff"),
    ))
    fs2.update_layout(
        title=dict(text="Feature Groups — 25 Total", font=dict(color="#ffe5c0", size=15)),
        paper_bgcolor=PAPER_BG, font=dict(color=FONT_COL),
        height=500, margin=dict(t=50,l=0,r=0,b=0),
    )
    st.plotly_chart(fs2, use_container_width=True)

    st.markdown("#### Feature Groups Detail")
    ICONS = {"Price Returns":"📊","Volume":"📦","Trend (MACD)":"📈",
             "Momentum":"⚡","Volatility":"🌡️","Candlestick":"🕯️"}
    cols = st.columns(3)
    for idx, (grp, (fs, color)) in enumerate(GROUPS.items()):
        with cols[idx % 3]:
            st.markdown(
                f'<div style="background:rgba(4,8,30,0.72);'
                f'border:1px solid {color}30;border-left:3px solid {color};'
                f'border-radius:8px;padding:12px 16px;margin-bottom:12px;">'
                f'<b style="color:{color};">{ICONS[grp]} {grp}</b><br>'
                + "".join(f'<span style="color:#b0b8d0;font-size:12px;">• {f}</span><br>'
                          for f in fs)
                + '</div>',
                unsafe_allow_html=True,
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<hr style="border:none;height:1px;margin:24px 0 8px;'
    'background:linear-gradient(90deg,transparent,rgba(255,140,66,0.38),'
    'rgba(77,200,255,0.38),transparent);">',
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#404868;font-size:12px;padding-bottom:12px;'>"
    "RELIANCE BiLSTM Forecast Dashboard · Built with Streamlit &amp; Plotly · "
    "Bidirectional LSTM · Walk-Forward Cross-Validation</p>",
    unsafe_allow_html=True,
)
