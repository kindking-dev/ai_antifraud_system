"""
Real-time Anti-Fraud Monitoring Dashboard.
"""

import streamlit as st
import httpx
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, UTC
import time

st.set_page_config(page_title="AITU Anti-Fraud Control Center", layout="wide")

st.title("🛡️ AI Behavioral Anti-Fraud Platform")
st.caption("Lead AI Architect Thesis Project | Specialized for Banks & Fintech")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("🔌 Simulator")
    amount = st.number_input(
        "Amount (KZT)", min_value=100, max_value=1000000, value=55000
    )
    user_id = st.text_input("User ID", value="USR-777")
    is_bot = st.toggle("🤖 Simulate Bot")

    if st.button("🚀 Send Transaction", use_container_width=True):
        payload = {
            "transaction_id": f"TXN-{int(time.time())}",
            "user_id": user_id,
            "amount_kzt": amount,
            "source": "MOBILE_APP",
            "session_trust_score": 0.15 if is_bot else 0.98,
            "network": {
                "ip_address": "185.12.33.11",
                "is_vpn_or_proxy": is_bot,
                "ja3_fingerprint": "771a4865486602329230abc123456789",
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0)",
            },
            "biometrics": {
                "gyroscope_x_y_z": [0.01, 0.01, 0.01] if is_bot else [0.4, 0.7, 0.2],
                "keystroke_entropy": 0.1 if is_bot else 0.8,
                "touch_pressure_variance": 0.02 if is_bot else 0.15,
            },
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }

        try:
            with httpx.Client() as client:
                res = client.post(
                    "http://127.0.0.1:8000/v1/score-transaction",
                    json=payload,
                    timeout=5.0,
                )
                if res.status_code == 200:
                    data = res.json()
                    data["timestamp"] = datetime.now().strftime("%H:%M:%S")
                    data["amount"] = amount
                    st.session_state.history.insert(0, data)
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

# Main UI
col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.write("### 📈 Live Feed")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        cols = [
            c
            for c in [
                "timestamp",
                "transaction_id",
                "amount",
                "fraud_probability",
                "action",
            ]
            if c in df.columns
        ]
        st.dataframe(df[cols], width="stretch", hide_index=True)
    else:
        st.info("System standby. Send a transaction to begin.")

with col2:
    st.write("### 🧠 XAI - SHAP Analysis")
    if st.session_state.history:
        latest = st.session_state.history[0]
        if "fraud_probability" in latest:
            prob = latest["fraud_probability"]
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    title={"text": f"Verdict: {latest['action']}"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "black"},
                        "steps": [
                            {"range": [0, 0.5], "color": "#a2fca2"},
                            {"range": [0.5, 0.85], "color": "#fcd3a2"},
                            {"range": [0.85, 1], "color": "#fca2a2"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig, width="stretch")
            for code in latest.get("reason_codes", []):
                st.warning(f"⚠️ {code}")
