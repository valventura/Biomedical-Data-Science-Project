import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import load_subject_data, resample_and_process, create_windows

st.set_page_config(page_title="Explore Subject", layout="wide")

st.title("Explore Subject")

# Get settings from session state
subject = st.session_state.get("subject", "VP02")
window_sec = st.session_state.get("window_sec", 30.0)
step_sec = st.session_state.get("step_sec", 15.0)

st.subheader(f"Subject: {subject}")

# Load and process data
with st.spinner("Loading data..."):
    try:
        ecg_df, eda_df, triggers_df = load_subject_data(subject)
        data = resample_and_process(ecg_df, eda_df, triggers_df)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Signals", "Triggers", "Windows"])

with tab1:
    st.subheader("Signal Visualization")

    # Time range controls
    duration = data["t"].max()
    col1, col2 = st.columns(2)
    with col1:
        t_start = st.number_input("Start Time (s)", 0.0, float(duration - 10), 0.0, 30.0)
    with col2:
        view_dur = st.number_input("Duration (s)", 10.0, min(300.0, duration), 120.0, 30.0)

    t_end = t_start + view_dur

    # filter to time range
    mask = (data["t"] >= t_start) & (data["t"] <= t_end)
    times = data["t"][mask]
    ecg_filtered = data["ecg"][mask]
    eda_filtered = data["eda"][mask]

    # Create plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("ECG", "EDA"), vertical_spacing=0.1)

    fig.add_trace(go.Scatter(x=times, y=ecg_filtered, name="ECG",
                             line=dict(color="#3498db", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=eda_filtered, name="EDA",
                             line=dict(color="#9b59b6", width=1)), row=2, col=1)

    # Add trigger regions
    for _, row in data["triggers"].iterrows():
        name = str(row["trigger_id"]).upper()
        start = row["start_t"]
        end = row["end_t"]

        # Skip if outside view
        if end < t_start or start > t_end:
            continue

        if name.startswith("CLIP") and "DEMO" not in name:
            color = "rgba(220, 80, 60, 0.15)"
        elif "REST" in name:
            color = "rgba(50, 200, 110, 0.15)"
        else:
            continue

        # Add to both subplots
        for r in [1, 2]:
            fig.add_vrect(x0=max(start, t_start), x1=min(end, t_end),
                         fillcolor=color, layer="below", line_width=0, row=r, col=1)

    fig.update_layout(height=500, showlegend=True)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Sampling rate: {data['fs']} Hz | Total duration: {duration:.2f}s")

with tab2:
    st.subheader("Trigger Events")

    triggers = data["triggers"]

    # Count clip vs rest
    clip_count = sum(1 for _, r in triggers.iterrows()
                     if str(r["trigger_id"]).upper().startswith("CLIP") and "DEMO" not in str(r["trigger_id"]).upper())
    rest_count = sum(1 for _, r in triggers.iterrows() if "REST" in str(r["trigger_id"]).upper())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Triggers", len(triggers))
    with col2:
        st.metric("Clip Triggers", clip_count)
    with col3:
        st.metric("Rest Triggers", rest_count)

    st.markdown("#### Trigger Details")

    rows = []
    for _, r in triggers.iterrows():
        trig_name = r["trigger_id"]
        is_clip = str(trig_name).upper().startswith("CLIP")
        is_rest = "REST" in str(trig_name).upper()
        lbl = "Clip" if is_clip else ("Rest" if is_rest else "Other")
        duration = r["end_t"] - r["start_t"]

        rows.append({
            "Trigger": trig_name,
            "Label": lbl,
            "Start (s)": f"{r['start_t']:.2f}",
            "End (s)": f"{r['end_t']:.2f}",
            "Duration (s)": f"{duration:.2f}"
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("""
    **Legend:** Green = Rest (neutral) | Red = Clip (fear stimulus)
    """)

with tab3:
    st.subheader("Window Statistics")

    # Create windows
    windows_df = create_windows(data, window_sec, step_sec)

    if len(windows_df) > 0:
        n_rest = sum(windows_df["label"] == 0)
        n_clip = sum(windows_df["label"] == 1)
        total = len(windows_df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Windows", total)
        with col2:
            st.metric("Rest Windows", f"{n_rest} ({100*n_rest/total:.2f}%)")
        with col3:
            st.metric("Clip Windows", f"{n_clip} ({100*n_clip/total:.2f}%)")

        # Simple bar chart
        fig = go.Figure(go.Bar(
            x=["Rest", "Clip"],
            y=[n_rest, n_clip],
            marker_color=["#27ae60", "#c0392b"],
            text=[n_rest, n_clip],
            textposition="auto"
        ))
        fig.update_layout(height=250, yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No valid windows found with current settings")
