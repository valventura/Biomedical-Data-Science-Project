import streamlit as st

# Page setup
st.set_page_config(page_title="SpiderSense", layout="wide")

# App title
st.title("SpiderSense")
st.subheader("Anxiety Prediction from Physiological Signals")

# Check for data and download if needed
from utils import get_subjects, check_and_download_data

# Check if data exists
subjects = get_subjects()

if not subjects:
    st.warning("Data not found. Starting download...")

    st.info("""
    **Downloading ~145 MB zip file. This should take less than 1 minute.**

    Check terminal for download progress.
    """)

    # Show progress
    progress_text = st.empty()
    progress_text.text("Status: Downloading and extracting...")

    success = check_and_download_data()

    if success:
        progress_text.empty()
        st.success("Download complete! Please refresh the page.")
        subjects = get_subjects()
    else:
        progress_text.empty()
        st.error("Download failed. Please download manually:")
        st.markdown("""
        1. Download from: https://drive.google.com/uc?id=1w135fj2ohHtGOpoScGF4y6PuGPlNC-X2
        2. Extract the zip file to this folder
        3. Refresh the page
        """)
        st.stop()

if not subjects:
    st.error("No data found! Check the data folder path.")
    st.stop()

# Sidebar controls
st.sidebar.title("SpiderSense")
st.sidebar.caption("Anxiety Prediction Dashboard")
st.sidebar.divider()

# Subject picker
subject = st.sidebar.selectbox("Subject", subjects)

# Parameters
st.sidebar.subheader("Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    window_sec = st.number_input("Window (s)", value=30.0, min_value=5.0, max_value=120.0)
with col2:
    step_sec = st.number_input("Step (s)", value=15.0, min_value=1.0, max_value=60.0)

threshold = st.sidebar.slider("Threshold", 0.1, 0.9, 0.5, 0.05)

st.sidebar.divider()

# Model picker
st.sidebar.subheader("Model")
model_choice = st.sidebar.radio("Model", ["LogReg", "RandomForest", "XGBoost"], label_visibility="collapsed")

st.sidebar.divider()

# Modality picker
st.sidebar.subheader("Modality")
modality = st.sidebar.radio("Modality", ["Fused", "ECG-only", "EDA-only"], label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption(f"{len(subjects)} subjects available")

# Store settings in session state so pages can use them
st.session_state["subject"] = subject
st.session_state["window_sec"] = window_sec
st.session_state["step_sec"] = step_sec
st.session_state["threshold"] = threshold
st.session_state["model_choice"] = model_choice
st.session_state["modality"] = modality

st.markdown("""
**SpiderSense** analyzes ECG and EDA signals to predict anxiety responses.

Pick a subject in the sidebar, choose your model settings, then explore the data
using the pages in the left menu:

- **Explore Subject** - raw signals and trigger events
- **Predict Timeline** - predictions over time
- **Model Evaluation** - cross-validation results
""")

# Quick stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Subjects", len(subjects))
with col2:
    st.metric("Current Subject", subject)
with col3:
    st.metric("Selected Model", model_choice)
