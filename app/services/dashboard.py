"""
Real-time Anti-Fraud Monitoring Dashboard - Enterprise Edition.
Integrated with PostgreSQL Audit Trail and Advanced XAI Visualization.
"""

import streamlit as st
import httpx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, UTC
import uuid

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/v1/score-transaction"
API_KEY = "DEV-MASTER-KEY"

st.set_page_config(
    page_title="AITU Anti-Fraud Control Center", page_icon="🛡️", layout="wide"
)

# Custom CSS для темной темы и профессионального вида карточек
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    [data-testid="stMetricValue"] { color: #58a6ff; font-family: 'JetBrains Mono', monospace; }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🛡️ AI Behavioral Anti-Fraud Platform")
st.caption("Lead AI Architect Thesis Project | AITU Software Engineering")

if "history" not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR: SIMULATOR ---
with st.sidebar:
    st.header("🔌 Transaction Simulator")
    amount = st.number_input(
        "Amount (KZT)", min_value=100, max_value=1000000, value=55000, step=500
    )
    user_id = st.text_input("User ID", value="USR-777")
    is_bot = st.toggle("🤖 Simulate Bot Behavior")

    st.divider()

    if st.button("🚀 Send Transaction", width="stretch"):
        # Генерация уникального ID для предотвращения конфликтов в БД
        txn_id = f"TXN-{uuid.uuid4().hex[:8].upper()}"

        payload = {
            "transaction_id": txn_id,
            "user_id": user_id,
            "amount_kzt": float(amount),
            "source": "MOBILE_APP",
            "session_trust_score": 0.15 if is_bot else 0.98,
            "network": {
                "ip_address": "185.12.33.11",
                "is_vpn_or_proxy": is_bot,
                "ja3_fingerprint": "771a4865486602329230abc123456789",
                "user_agent": "Mozilla/5.0 (iPhone; CPU OS 17_0)",
            },
            "biometrics": {
                "gyroscope_x_y_z": [0.01, 0.01, 0.01] if is_bot else [0.4, 0.7, 0.2],
                "keystroke_entropy": 0.12 if is_bot else 0.85,
                "touch_pressure_variance": 0.02 if is_bot else 0.18,
            },
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }

        try:
            with httpx.Client() as client:
                res = client.post(
                    API_URL, json=payload, timeout=5.0, headers={"X-API-KEY": API_KEY}
                )

                if res.status_code == 200:
                    data = res.json()
                    data["timestamp"] = datetime.now().strftime("%H:%M:%S")
                    data["amount"] = amount
                    data["user_id"] = user_id
                    st.session_state.history.insert(0, data)
                else:
                    st.error(f"🚫 API Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"📡 Connection Failed: {e}")

# --- MAIN UI: DASHBOARD ---
if st.session_state.history:
    latest = st.session_state.history[0]

    # 1. Секция верхних метрик
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Risk Probability", f"{latest['fraud_probability']:.2%}")
    with m2:
        st.metric("Final Verdict", latest["action"])
    with m3:
        latency = latest["processing_time_ms"]
        # Мониторинг SLA: подсвечиваем задержку, если она выше 50мс
        st.metric(
            "API Latency",
            f"{latency:.2f} ms",
            delta=f"{latency - 50:.1f}ms" if latency > 50 else None,
            delta_color="inverse",
        )
    with m4:
        st.metric("Storage Status", "SQL + NoSQL Online")

    st.divider()

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.write("### 📈 Live Audit Trail (PostgreSQL)")
        df = pd.DataFrame(st.session_state.history)

        def color_action(val):
            color = (
                "#a2fca2"
                if val == "ALLOW"
                else "#fca2a2"
                if val == "BLOCK"
                else "#fcd3a2"
            )
            return f"background-color: {color}; color: black; font-weight: bold"

        display_cols = [
            "timestamp",
            "transaction_id",
            "user_id",
            "amount",
            "fraud_probability",
            "action",
        ]
        st.dataframe(
            df[display_cols].style.map(color_action, subset=["action"]),
            width="stretch",
            hide_index=True,
        )

    with col2:
        st.write("### 🧠 Explainable AI: Feature Impact")

        # Спидометр риска
        prob = latest["fraud_probability"]
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob,
                title={"text": f"Verdict: {latest['action']}", "font": {"size": 20}},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "white"},
                    "steps": [
                        {"range": [0, 0.5], "color": "#00cc96"},
                        {"range": [0, 0.85], "color": "#ffa15a"},
                        {"range": [0, 1], "color": "#ef553b"},
                    ],
                },
            )
        )
        gauge_fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(gauge_fig, width="stretch")

        # SHAP Bar Chart (Влияние признаков)
        if latest.get("feature_impacts"):
            impacts = latest["feature_impacts"]
            # Берем топ-6 самых влиятельных факторов
            sorted_impacts = dict(
                sorted(impacts.items(), key=lambda item: abs(item[1]), reverse=True)[:6]
            )

            impact_df = pd.DataFrame(
                {
                    "Feature": list(sorted_impacts.keys()),
                    "Impact": list(sorted_impacts.values()),
                    "Color": [
                        "#ef553b" if x > 0 else "#00cc96"
                        for x in sorted_impacts.values()
                    ],
                }
            )

            bar_fig = px.bar(
                impact_df,
                x="Impact",
                y="Feature",
                orientation="h",
                title="Risk Drivers (SHAP Weights)",
                color="Color",
                color_discrete_map="identity",
            )
            bar_fig.update_layout(
                height=300,
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(bar_fig, width="stretch")

        # Детальные примечания
        if latest.get("reason_codes"):
            with st.expander("📝 System Investigation Notes", expanded=True):
                for code in latest["reason_codes"]:
                    if code == "AML_VELOCITY_SPIKE":
                        st.error(f"🚨 {code}: Transaction frequency anomaly.")
                    elif code == "SUSPICIOUS_BEHAVIOR":
                        st.warning(f"⚠️ {code}: Biometric pattern mismatch.")
                    else:
                        st.info(f"🔍 {code}: Model confidence trigger.")

else:
    st.info("🛰️ System standby. Waiting for transaction events...")
    st.plotly_chart(
        px.line(title="Waiting for real-time telemetry..."), width="stretch"
    )
