import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

from utils import build_all_features, train_and_predict

st.set_page_config(page_title="Predict Timeline", layout="wide")

st.title("Predict Timeline")

# Get settings from session state
subject = st.session_state.get("subject", "VP02")
window_sec = st.session_state.get("window_sec", 30.0)
step_sec = st.session_state.get("step_sec", 15.0)
threshold = st.session_state.get("threshold", 0.5)
model_name = st.session_state.get("model_choice", "LogReg")
modality = st.session_state.get("modality", "Fused")

st.subheader(f"Subject: {subject} | Model: {model_name} | Modality: {modality}")

# Load all data and train model
@st.cache_data(show_spinner=False)
def get_all_features(win, step):
    return build_all_features(win, step)

with st.spinner("Loading data (trains on other subjects, predicts on selected)..."):
    df_all = get_all_features(window_sec, step_sec)

if df_all is None or len(df_all) == 0:
    st.error("No data available!")
    st.stop()

# Check if subject exists
if subject not in df_all["subject"].unique():
    st.warning(f"Subject {subject} not in dataset or has no valid windows")
    st.stop()

# Train and predict
@st.cache_data(show_spinner=False)
def get_predictions(_df, subj, model, mod):
    return train_and_predict(_df, subj, model, mod)

with st.spinner("Training model and predicting..."):
    predictions = get_predictions(df_all, subject, model_name, modality)

if predictions is None or len(predictions) == 0:
    st.warning("No predictions available")
    st.stop()

# Threshold slider
st.markdown("### Threshold Control")
threshold = st.slider("Classification Threshold", 0.1, 0.9, threshold, 0.05,
                      help="Higher = more conservative clip predictions")
st.session_state["threshold"] = threshold

# Get data for plotting
t = predictions["t_start"].values
prob = predictions["prob"].values
true_labels = predictions["label"].values
pred_labels = (prob >= threshold).astype(int)

st.markdown("### Prediction Timeline")

fig = go.Figure()

# Background shading for true labels
for i in range(len(t)):
    # Estimate window width
    if i < len(t) - 1:
        width = t[i+1] - t[i]
    else:
        width = window_sec

    color = "rgba(50, 200, 110, 0.2)" if true_labels[i] == 0 else "rgba(220, 80, 60, 0.2)"

    fig.add_vrect(x0=t[i], x1=t[i]+width, fillcolor=color, layer="below", line_width=0)

# Probability line
fig.add_trace(go.Scatter(x=t, y=prob, mode="lines+markers", name="P(Clip)",
                         line=dict(color="#3498db", width=2), marker=dict(size=6)))

# Threshold line
fig.add_hline(y=threshold, line_dash="dash", line_color="red",
              annotation_text=f"Threshold: {threshold:.2f}")

# Legend entries for true labels
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="True: Rest",
                         marker=dict(size=10, color="#27ae60")))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="True: Clip",
                         marker=dict(size=10, color="#c0392b")))

fig.update_layout(height=400, xaxis_title="Time (s)", yaxis_title="P(Clip)",
                  yaxis=dict(range=[-0.05, 1.05]))

st.plotly_chart(fig, use_container_width=True)

st.caption("Blue line: P(Clip) | Red dashed: Threshold | Green bg: True Rest | Red bg: True Clip")

st.markdown("### Performance Metrics")

# Calculate metrics
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

bal_acc = balanced_accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)
try:
    auc_score = roc_auc_score(true_labels, prob)
except:
    auc_score = 0.0

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
rest_sens = tn / (tn + fp) if (tn + fp) > 0 else 0
clip_sens = tp / (tp + fn) if (tp + fn) > 0 else 0

# Display in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Metrics")
    st.metric("Balanced Accuracy", f"{bal_acc:.2f}")
    st.metric("F1 Score", f"{f1:.2f}")
    st.metric("ROC-AUC", f"{auc_score:.2f}")

with col2:
    st.markdown("#### Sensitivity")
    st.metric("Rest Sensitivity", f"{rest_sens:.2f}")
    st.metric("Clip Sensitivity", f"{clip_sens:.2f}")

with col3:
    st.markdown("#### Confusion Matrix")

    # Simple heatmap
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=["Rest", "Clip"], y=["Rest", "Clip"],
        colorscale="Blues", showscale=False,
        text=[[f"{cm[i,j]}" for j in range(2)] for i in range(2)],
        texttemplate="%{text}", textfont=dict(size=16)
    ))
    fig_cm.update_layout(height=200, xaxis_title="Predicted", yaxis_title="True",
                         yaxis=dict(autorange="reversed"), margin=dict(l=50, r=20, t=20, b=50))
    st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("### ROC Curve")

fpr, tpr, _ = roc_curve(true_labels, prob)
roc_auc = auc(fpr, tpr)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"ROC (AUC={roc_auc:.2f})", line=dict(color="#3498db", width=2)))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                             line=dict(color="gray", width=1, dash="dash")))
fig_roc.update_layout(height=350, xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate")

st.plotly_chart(fig_roc, use_container_width=True)

with st.expander("Prediction Details"):
    details = predictions.copy()
    details["predicted"] = pred_labels
    details["correct"] = (pred_labels == true_labels)
    details.columns = ["Start (s)", "End (s)", "True Label", "P(Clip)", "Predicted", "Correct"]

    st.dataframe(details, use_container_width=True)

    num_correct = details["Correct"].sum()
    st.markdown(f"**Accuracy:** {num_correct}/{len(details)} ({100*num_correct/len(details):.2f}%)")
