import streamlit as st
import pandas as pd
import json
import plotly.express as px
import os
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="WAF Layered Monitor",
    page_icon="🛡️",
    layout="wide"
)

# Log file path (mounted from docker volume)
LOG_FILE = "/data/requests.jsonl"

# --- TITLE ---
st.title("🛡️ WAF Layered Architecture Monitor")
st.markdown("### Security Monitoring System (DDoS & Deep Learning)")
st.markdown("Based on MDPI Sensors 2023 Paper")
st.markdown("---")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
# Auto-refresh rate slider
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 60, 2)

st.sidebar.markdown("""
**Legend:**
* 🔴 **Layer 1 Blocked:** DDoS / High Rate Limit.
* 🟠 **Layer 2 Blocked:** SQLi / XSS (AI Model).
* 🟢 **Allowed:** Normal Traffic.
""")

# --- LOAD DATA FUNCTION ---
@st.cache_data(ttl=refresh_rate)
def load_data():
    data = []
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    
    # Read log file (Last 2000 lines to prevent lag)
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[-2000:]:
                try:
                    data.append(json.loads(line))
                except:
                    continue
    except Exception:
        return pd.DataFrame()
    
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['ts'], unit='s')
    
    # --- CLASSIFY LAYERS (LOGIC) ---
    def classify_layer(row):
        info = str(row.get('model_info', ''))
        action = str(row.get('action', ''))
        
        # Layer 1: Blocked by Rate Limit (DDoS)
        if "blocked-ddos" in action or "DDoS" in info:
            return "Layer 1: DDoS Attack"
        
        # Layer 2: Blocked by AI/Regex (SQLi/XSS)
        elif "blocked" in action:
            return "Layer 2: SQLi/XSS Attack"
            
        # Allowed Traffic
        return "Normal Traffic"

    df['Type'] = df.apply(classify_layer, axis=1)
    return df

# Manual Refresh Button
if st.button('🔄 Refresh Data Now'):
    st.cache_data.clear()

# Load Data
df = load_data()

if df.empty:
    st.warning("⏳ Waiting for data... Please send requests to Proxy (Port 8010)")
else:
    # --- PART 1: STATISTICS ---
    st.subheader("1. Efficiency Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_req = len(df)
    layer1_block = len(df[df['Type'] == "Layer 1: DDoS Attack"])
    layer2_block = len(df[df['Type'] == "Layer 2: SQLi/XSS Attack"])
    clean_req = len(df[df['Type'] == "Normal Traffic"])
    
    col1.metric("Total Requests", total_req)
    col2.metric("Layer 1 Blocked (DDoS)", layer1_block, delta="Rate Limit", delta_color="inverse")
    col3.metric("Layer 2 Blocked (AI)", layer2_block, delta="Deep Learning", delta_color="inverse")
    col4.metric("Allowed Traffic", clean_req, delta="Safe")

    # --- PART 2: CHARTS ---
    st.markdown("---")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Attack Distribution")
        # Pie Chart
        fig_pie = px.pie(df, names='Type', 
                         title='Traffic Ratio by Layer',
                         color='Type',
                         color_discrete_map={
                             'Layer 1: DDoS Attack': '#FF4B4B',        # Red
                             'Layer 2: SQLi/XSS Attack': '#FFA15A',    # Orange
                             'Normal Traffic': '#00CC96'               # Green
                         })
        # DA SUA: Doi use_container_width=True thanh width="stretch"
        st.plotly_chart(fig_pie, width="stretch")

    with col_right:
        st.subheader("AI Confidence Score (Layer 2)")
        # Filter only Layer 2 traffic
        df_layer2 = df[df['Type'] != "Layer 1: DDoS Attack"]
        
        if not df_layer2.empty:
            fig_hist = px.histogram(df_layer2, x="model_score", nbins=20, 
                                    title="LSTM Model Score Distribution",
                                    labels={'model_score': 'Malicious Score (0=Safe, 1=Attack)'},
                                    color_discrete_sequence=['#636EFA'],
                                    range_x=[0, 1])
            # Draw Threshold Line at 0.8
            fig_hist.add_vline(x=0.8, line_width=2, line_dash="dash", line_color="red", annotation_text="Threshold 0.8")
            # DA SUA: Doi use_container_width=True thanh width="stretch"
            st.plotly_chart(fig_hist, width="stretch")
        else:
            st.info("No traffic reached Layer 2 yet.")

    # --- PART 3: LIVE LOGS ---
    st.markdown("---")
    st.subheader("📜 Live Logs Details")
    
    # Sort by latest
    df_display = df[['datetime', 'method', 'path', 'Type', 'model_score', 'action']].sort_values(by='datetime', ascending=False)
    
    # Highlight colors
    def highlight_row(row):
        if row['Type'] == "Layer 1: DDoS Attack":
            return ['background-color: #ffcccc'] * len(row) # Light Red
        elif row['Type'] == "Layer 2: SQLi/XSS Attack":
            return ['background-color: #ffe5cc'] * len(row) # Light Orange
        return [''] * len(row)

    # DA SUA: Doi use_container_width=True thanh width="stretch"
    st.dataframe(df_display.style.apply(highlight_row, axis=1), width="stretch", height=400)

# --- AUTO REFRESH LOGIC ---
# This ensures the dashboard updates automatically
time.sleep(refresh_rate)
st.rerun()