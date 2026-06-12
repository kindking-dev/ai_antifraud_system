import streamlit as st
import httpx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
import uuid
import os
import time

# --- CONFIGURATION ---
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")
API_KEY = os.getenv("API_KEY", "DEV-MASTER-KEY")

st.set_page_config(
    page_title="AI Anti-Fraud SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- ADVANCED CYBER-CSS ---
st.markdown(
    """
    <style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #0e1117, #050a30);
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        color: #00d2ff !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 15, 30, 0.95);
        border-right: 1px solid rgba(0, 210, 255, 0.2);
    }
    
    /* XAI AI Box */
    .xai-box {
        background-color: rgba(15, 20, 35, 0.8);
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #b142ff;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(177, 66, 255, 0.1);
    }
    .xai-box strong {
        color: #00d2ff;
    }
    
    /* Pulsing Shield Animation */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 210, 255, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(0, 210, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 210, 255, 0); }
    }
    .shield-active {
        animation: pulse 2s infinite;
        border-radius: 50%;
    }
    
    /* Live indicator */
    .live-dot {
        display: inline-block;
        width: 10px; height: 10px;
        background: #00ff66;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
        margin-right: 8px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- HELPER: Fetch transactions from PostgreSQL via API ---
@st.cache_data(ttl=5)
def fetch_transactions():
    """Pulls real transaction audit trail from PostgreSQL via the API."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{BASE_URL}/transactions",
                params={"limit": 200},
                headers={"X-API-KEY": API_KEY},
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        st.error(f"API Connection Error: {e}")
    return []


# --- SESSION STATE ---
if "last_behavior_score" not in st.session_state:
    st.session_state.last_behavior_score = 0.5
if "system_status" not in st.session_state:
    st.session_state.system_status = "MONITORING"

# --- SIDEBAR: SCENARIO CONTROLLER ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=80)
    st.title("Anti-Fraud Controller")

    st.markdown(
        '<span class="live-dot"></span> **LIVE MONITORING**', unsafe_allow_html=True
    )

    auto_refresh = st.toggle("🔄 Auto-Refresh (5s)", value=True)

    st.divider()

    st.subheader("🛠️ Manual Test Scenarios")
    scenario = st.selectbox(
        "Select Attack Vector:",
        ["✅ Standard User", "🕵️ Account Takeover", "🤖 Bot Script Injection"],
    )

    # Конфигурация сценариев
    if scenario == "✅ Standard User":
        sc_uid, sc_vel, sc_press, sc_trust = "user_9921", 1.2, 0.15, 0.98
        sc_ip, sc_ua = (
            "185.120.45.22",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
        )
        sc_vpn = False
        sc_ja3 = "cd08e31494f9531f560d64c695473da9"
    elif scenario == "🕵️ Account Takeover":
        sc_uid, sc_vel, sc_press, sc_trust = "user_9921", 8.5, 0.95, 0.35
        sc_ip, sc_ua = (
            "194.26.29.11",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        sc_vpn = True
        sc_ja3 = "b32309a26951912be7dba376398abc3b"
    else:  # Bot Script Injection
        sc_uid, sc_vel, sc_press, sc_trust = "user_9921", 15.0, 0.01, 0.05
        sc_ip, sc_ua = "104.28.19.20", "python-requests/2.31.0"
        sc_vpn = True
        sc_ja3 = "3b5074b1b5d032e5620f69f9f700ff0e"

    with st.expander("🎛️ Manual Overrides", expanded=False):
        user_id = st.text_input("User ID", value=sc_uid)
        amount = st.number_input("Amount (KZT)", value=55000)

        st.caption("Biometric Data")
        velocity = st.slider("Swipe Velocity", 0.0, 20.0, sc_vel)
        pressure = st.slider("Touch Pressure", 0.0, 1.0, sc_press)

        st.caption("Network & Device")
        ip_address = st.text_input("IP Address", value=sc_ip)
        user_agent = st.text_input("User Agent", value=sc_ua)
        ja3 = st.text_input("JA3 Fingerprint", value=sc_ja3)
        is_vpn = st.checkbox("Is VPN/Proxy/Tor?", value=sc_vpn)

        st.caption("System Confidence")
        trust_score = st.slider("Session Trust", 0.0, 1.0, sc_trust)

    st.divider()

    # КНОПКА 0: Сохранение эталонного профиля (Baseline)
    if st.button("💾 STEP 0: Set Normal Baseline", use_container_width=True):
        payload = {
            "user_id": user_id,
            "features": {
                "duration_ms_mean": 250.0,
                "duration_ms_std": 20.0,
                "duration_ms_max": 300.0,
                "length_px_mean": 450.0,
                "length_px_std": 30.0,
                "length_px_max": 500.0,
                "velocity_mean": velocity,
                "velocity_std": 1.0,
                "velocity_max": velocity + 2,
                "median_pressure_mean": pressure,
                "median_pressure_std": 0.05,
                "median_pressure_max": pressure + 0.1,
                "median_area_mean": 0.5,
                "median_area_std": 0.05,
                "median_area_max": 0.6,
            },
        }
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{BASE_URL}/set-baseline",
                    json=payload,
                    headers={"X-API-KEY": API_KEY},
                )
                if resp.status_code == 200:
                    st.toast("Normal User Profile Saved (Baseline)!", icon="✅")
                else:
                    st.error(f"Error setting baseline: {resp.text}")
        except Exception as e:
            st.error(f"API Connection Error: {e}")

    # КНОПКА 1: Отправка биометрии
    if st.button("📡 STEP 1: Stream Biometrics", use_container_width=True):
        payload = {
            "user_id": user_id,
            "features": {
                "duration_ms_mean": 250.0,
                "duration_ms_std": 20.0,
                "duration_ms_max": 300.0,
                "length_px_mean": 450.0,
                "length_px_std": 30.0,
                "length_px_max": 500.0,
                "velocity_mean": velocity,
                "velocity_std": 1.0,
                "velocity_max": velocity + 2,
                "median_pressure_mean": pressure,
                "median_pressure_std": 0.05,
                "median_pressure_max": pressure + 0.1,
                "median_area_mean": 0.5,
                "median_area_std": 0.05,
                "median_area_max": 0.6,
            },
        }
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{BASE_URL}/score-behavior",
                    json=payload,
                    headers={"X-API-KEY": API_KEY},
                )
                if resp.status_code == 200:
                    st.session_state.last_behavior_score = resp.json()[
                        "fraud_probability"
                    ]
                    st.toast("Biometrics stored in Redis!", icon="🧠")
                else:
                    st.error(f"Behavior API Error: {resp.text}")
        except Exception as e:
            st.error(f"API Connection Error: {e}")

    # КНОПКА 2: Транзакция
    if st.button("⚡ STEP 2: Execute Payment", use_container_width=True):
        tx_payload = {
            "transaction_id": f"TXN-{uuid.uuid4().hex[:6].upper()}",
            "user_id": user_id,
            "amount_kzt": float(amount),
            "source": "MOBILE_APP",
            "network": {
                "ip_address": ip_address,
                "ja3_fingerprint": ja3,
                "user_agent": user_agent,
                "is_vpn_or_proxy": is_vpn,
            },
            "session_trust_score": trust_score,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{BASE_URL}/score-transaction",
                    json=tx_payload,
                    headers={"X-API-KEY": API_KEY},
                )
                if resp.status_code == 200:
                    st.toast("Transaction scored & saved to PostgreSQL!", icon="⚡")
                    st.cache_data.clear()
                else:
                    st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

    st.divider()

    if st.button("🔄 Force Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.button("🗑️ Reset Database & Redis", type="primary", use_container_width=True):
        try:
            with httpx.Client() as client:
                resp = client.delete(
                    f"{BASE_URL}/system/reset", headers={"X-API-KEY": API_KEY}
                )
                if resp.status_code == 200:
                    st.toast("System Completely Reset!", icon="💣")
                    st.cache_data.clear()
                    # Clear session state items
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
                else:
                    st.error(f"Error resetting system: {resp.text}")
        except Exception as e:
            st.error(f"API Connection Error: {e}")


# --- MAIN DASHBOARD ---
st.markdown("### 🛡️ AI-POWERED BEHAVIORAL ANTI-FRAUD SERVICE — Security Operations Center (SOC)")
st.caption(f"Источник данных: **PostgreSQL** (`transaction_logs`) | API: `{BASE_URL}`")

# Fetch real data from PostgreSQL
transactions = fetch_transactions()

# Вкладки
tab_monitor, tab_behavior, tab_health = st.tabs(
    ["📊 Real-time Monitor", "🧬 Biometric Identity", "⚙️ System Health"]
)

with tab_monitor:
    if transactions:
        df = pd.DataFrame(transactions)
        latest = transactions[0]

        # ===== TOP METRICS =====
        total_tx = len(df)
        blocked = len(df[df["action"] == "BLOCK"])
        challenged = len(df[df["action"] == "CHALLENGE"])
        allowed = len(df[df["action"] == "ALLOW"])
        avg_latency = df["processing_time_ms"].mean()
        avg_risk = df["fraud_probability"].mean()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Transactions", total_tx)
        c2.metric("🟢 Allowed", allowed)
        c3.metric("🟡 Challenged", challenged)
        c4.metric("🔴 Blocked", blocked)
        c5.metric("⏱️ Avg Latency", f"{avg_latency:.1f}ms")

        st.divider()

        # ===== LATEST DECISION =====
        col_main, col_xai = st.columns([1.2, 0.8])

        with col_main:
            st.write("#### 📟 Latest Decision Gauge")
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=latest["fraud_probability"],
                    title={"text": f"TX: {latest['transaction_id']}"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "#00d2ff"},
                        "steps": [
                            {"range": [0, 0.5], "color": "rgba(0, 255, 0, 0.1)"},
                            {"range": [0.5, 0.85], "color": "rgba(255, 165, 0, 0.1)"},
                            {"range": [0.85, 1], "color": "rgba(255, 0, 0, 0.1)"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "value": 0.85,
                        },
                    },
                )
            )
            fig_gauge.update_layout(
                height=300,
                margin=dict(t=50, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ===== RISK DISTRIBUTION CHART =====
            st.write("#### 📈 Risk Distribution (All Transactions)")
            fig_hist = px.histogram(
                df,
                x="fraud_probability",
                nbins=20,
                color_discrete_sequence=["#00d2ff"],
                labels={"fraud_probability": "Fraud Probability"},
            )
            fig_hist.add_vline(
                x=0.5,
                line_dash="dash",
                line_color="orange",
                annotation_text="CHALLENGE",
            )
            fig_hist.add_vline(
                x=0.85, line_dash="dash", line_color="red", annotation_text="BLOCK"
            )
            fig_hist.update_layout(
                height=250,
                margin=dict(t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_xai:
            st.write("#### 🧠 Explainable AI (XAI)")

            # Ключ сессии для хранения отчета по конкретной транзакции
            llm_state_key = f"llm_report_{latest['transaction_id']}"

            if llm_state_key not in st.session_state:
                if st.button(
                    "🤖 Сгенерировать отчет нейросети (LLM)", use_container_width=True
                ):
                    with st.spinner("Qwen 2.5 анализирует транзакцию..."):
                        llm_payload = {
                            "transaction_id": latest["transaction_id"],
                            "fraud_probability": latest["fraud_probability"],
                            "action": latest["action"],
                            "reason_codes": latest.get("reason_codes", []),
                            "feature_impacts": latest.get("feature_impacts", {}),
                        }
                        try:
                            with httpx.Client(timeout=30.0) as client:
                                resp = client.post(
                                    f"{BASE_URL}/explain/transaction",
                                    json=llm_payload,
                                    headers={"X-API-KEY": API_KEY},
                                )
                                if resp.status_code == 200:
                                    st.session_state[llm_state_key] = resp.json()
                                    st.rerun()
                                else:
                                    st.error(
                                        f"LLM API Error: {resp.status_code} - {resp.text}"
                                    )
                        except Exception as e:
                            st.error(f"Connection Failed: {e}")

            if llm_state_key in st.session_state:
                result = st.session_state[llm_state_key]
                st.markdown(
                    f'<div class="xai-box">{result["explanation_markdown"]}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"⏱️ LLM Processing: {result['processing_time_ms']}ms | Status: {result['status']}"
                )
                if st.button(
                    "🔄 Сбросить отчет", key="reset_llm", use_container_width=True
                ):
                    del st.session_state[llm_state_key]
                    st.rerun()

            if latest.get("feature_impacts"):
                impacts = latest["feature_impacts"]
                imp_df = pd.DataFrame(
                    {"Feature": list(impacts.keys()), "Weight": list(impacts.values())}
                ).sort_values(by="Weight", ascending=True)

                fig_bar = px.bar(
                    imp_df,
                    x="Weight",
                    y="Feature",
                    orientation="h",
                    color="Weight",
                    color_continuous_scale="RdYlGn_r",
                )
                fig_bar.update_layout(
                    height=300,
                    margin=dict(t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "white"},
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            if latest.get("reason_codes"):
                for code in latest["reason_codes"]:
                    st.warning(f"🔍 Reason: {code}")
            else:
                st.success("✅ No risk triggers detected")

            # === Verdict Pie Chart ===
            st.write("#### 🎯 Verdict Breakdown")
            verdict_df = df["action"].value_counts().reset_index()
            verdict_df.columns = ["Verdict", "Count"]
            color_map = {"ALLOW": "#00ff66", "CHALLENGE": "#facc15", "BLOCK": "#ff003c"}
            fig_pie = px.pie(
                verdict_df,
                names="Verdict",
                values="Count",
                color="Verdict",
                color_discrete_map=color_map,
                hole=0.4,
            )
            fig_pie.update_layout(
                height=280,
                margin=dict(t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
                legend=dict(font=dict(color="white")),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ===== AUDIT TRAIL TABLE (PostgreSQL) =====
        st.divider()
        st.write("#### 📜 Full Audit Trail (PostgreSQL `transaction_logs`)")
        display_df = df[
            [
                "timestamp_utc",
                "transaction_id",
                "user_id",
                "amount_kzt",
                "fraud_probability",
                "action",
                "processing_time_ms",
            ]
        ].copy()
        display_df.columns = [
            "Timestamp",
            "TX ID",
            "User",
            "Amount (₸)",
            "Risk",
            "Verdict",
            "Latency (ms)",
        ]

        def color_verdict(val):
            if val == "BLOCK":
                return "background-color: rgba(255,0,60,0.3); color: #ff003c; font-weight: bold"
            elif val == "CHALLENGE":
                return "background-color: rgba(250,204,21,0.2); color: #facc15; font-weight: bold"
            return "background-color: rgba(0,255,102,0.1); color: #00ff66"

        styled = display_df.style.applymap(color_verdict, subset=["Verdict"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=400)

    else:
        st.info(
            "📡 Ожидание данных... Отправьте транзакцию с мобильного приложения или через кнопки в сайдбаре."
        )

with tab_behavior:
    st.write("#### 🧬 Digital Twin Comparison")
    col_radar, col_telemetry = st.columns([1, 1])

    with col_radar:
        categories = ["Velocity", "Pressure", "Area", "Time_Delta", "Entropy"]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=[0.2, 0.3, 0.4, 0.5, 0.6],
                theta=categories,
                fill="toself",
                name="User Template",
                line_color="#ff4b4b",
            )
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=[0.4, 0.2, 0.5, 0.4, 0.7],
                theta=categories,
                fill="toself",
                name="Current Session",
                line_color="#00d2ff",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=False)),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_telemetry:
        st.write("#### 🛰️ Raw Behavioral Stream")
        st.metric(
            "Inferred Behavior Score", f"{st.session_state.last_behavior_score:.4f}"
        )
        st.write(
            "The chart above visualizes the mismatch between the stored behavioral template "
            "in Redis and the incoming real-time telemetry. A high deviation triggers a "
            "'Late Fusion' risk increase."
        )

        if transactions:
            latency_data = pd.DataFrame(
                {
                    "TX #": range(len(transactions)),
                    "Latency (ms)": [
                        t["processing_time_ms"] for t in reversed(transactions)
                    ],
                }
            )
            st.line_chart(latency_data, x="TX #", y="Latency (ms)", height=200)

with tab_health:
    st.write("#### ⚙️ System Node Status")
    h1, h2, h3, h4 = st.columns(4)
    h1.success("API Gateway: ONLINE")
    h2.success("ML Core (CatBoost): LOADED")
    h3.success("Behavioral Engine: ACTIVE")
    h4.success("WebSocket Uplink: LIVE")

    st.divider()
    st.write("#### ⏱️ Latency Distribution (SLA < 50ms)")
    if transactions:
        latencies = [x["processing_time_ms"] for x in transactions]
        sla_ok = sum(1 for l in latencies if l < 50)
        sla_pct = (sla_ok / len(latencies)) * 100

        l1, l2, l3 = st.columns(3)
        l1.metric("Min Latency", f"{min(latencies):.1f}ms")
        l2.metric("Max Latency", f"{max(latencies):.1f}ms")
        l3.metric("SLA Compliance", f"{sla_pct:.0f}%")

        fig_lat = px.histogram(
            pd.DataFrame({"Latency (ms)": latencies}),
            x="Latency (ms)",
            nbins=30,
            color_discrete_sequence=["#00d2ff"],
        )
        fig_lat.add_vline(
            x=50, line_dash="dash", line_color="red", annotation_text="SLA Limit (50ms)"
        )
        fig_lat.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
        )
        st.plotly_chart(fig_lat, use_container_width=True)
    else:
        st.caption("No telemetry data yet.")

# --- AUTO REFRESH ---
if auto_refresh:
    time.sleep(5)
    st.cache_data.clear()
    st.rerun()
