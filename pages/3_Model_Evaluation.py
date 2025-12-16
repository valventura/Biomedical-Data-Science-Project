import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import build_all_features, run_cv, get_model, get_feature_cols, HAS_XGB

st.set_page_config(page_title="Model Evaluation", layout="wide")

st.title("Model Evaluation")
st.subheader("Cross-Validated Performance Across All Subjects")

# Get settings from session state
window_sec = st.session_state.get("window_sec", 30.0)
step_sec = st.session_state.get("step_sec", 15.0)
model_name = st.session_state.get("model_choice", "LogReg")
modality = st.session_state.get("modality", "Fused")

# Load all data
@st.cache_data(show_spinner=False)
def get_all_features(win, step):
    return build_all_features(win, step)

with st.spinner("Building features for all subjects..."):
    df_all = get_all_features(window_sec, step_sec)

if df_all is None or len(df_all) == 0:
    st.error("No data available!")
    st.stop()

# Dataset summary
n_subjects = df_all["subject"].nunique()
n_windows = len(df_all)
n_rest = sum(df_all["label"] == 0)
n_clip = sum(df_all["label"] == 1)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Subjects", n_subjects)
with col2:
    st.metric("Total Windows", n_windows)
with col3:
    st.metric("Rest Windows", n_rest)
with col4:
    st.metric("Clip Windows", n_clip)

st.divider()

# Create tabs
tab1, tab2 = st.tabs(["Model Comparison", "Detailed Results"])

with tab1:
    st.subheader("Compare All Models")

    # Run CV for all models
    @st.cache_data(show_spinner=False)
    def run_all_models(_df, mod):
        models = ["LogReg", "RandomForest"]
        if HAS_XGB:
            models.append("XGBoost")

        results = []
        for m in models:
            fold_df, cm = run_cv(_df, m, mod, n_splits=5)
            results.append({
                "Model": m,
                "Modality": mod,
                "Bal. Acc": f"{fold_df['bal_acc'].mean():.2f} +/- {fold_df['bal_acc'].std():.2f}",
                "F1": f"{fold_df['f1'].mean():.2f} +/- {fold_df['f1'].std():.2f}",
                "AUC": f"{fold_df['auc'].mean():.2f} +/- {fold_df['auc'].std():.2f}",
                "_acc": fold_df['bal_acc'].mean()
            })
        return pd.DataFrame(results).sort_values("_acc", ascending=False).drop(columns=["_acc"])

    with st.spinner("Running cross-validation..."):
        comparison = run_all_models(df_all, modality)

    st.markdown(f"**Modality: {modality}**")
    st.dataframe(comparison, use_container_width=True)

    # Bar chart
    st.markdown("#### Balanced Accuracy Comparison")

    # Parse for plotting
    models = comparison["Model"].tolist()
    accs = [float(x.split("+/-")[0]) for x in comparison["Bal. Acc"].tolist()]
    stds = [float(x.split("+/-")[1]) for x in comparison["Bal. Acc"].tolist()]

    fig = go.Figure(go.Bar(
        x=models, y=accs,
        error_y=dict(type="data", array=stds),
        marker_color="#3498db"
    ))
    fig.update_layout(height=300, yaxis_title="Balanced Accuracy")
    st.plotly_chart(fig, use_container_width=True)

    # All modalities comparison
    st.markdown("#### Performance Across All Modalities")

    @st.cache_data(show_spinner=False)
    def run_all_modalities(_df):
        all_results = []
        for mod in ["Fused", "ECG-only", "EDA-only"]:
            models = ["LogReg", "RandomForest"]
            if HAS_XGB:
                models.append("XGBoost")

            for m in models:
                try:
                    fold_df, _ = run_cv(_df, m, mod, n_splits=5)
                    all_results.append({
                        "Model": m,
                        "Modality": mod,
                        "Bal. Acc": f"{fold_df['bal_acc'].mean():.2f}",
                        "F1": f"{fold_df['f1'].mean():.2f}",
                        "AUC": f"{fold_df['auc'].mean():.2f}"
                    })
                except Exception:
                    continue
        return pd.DataFrame(all_results)

    with st.spinner("Running cross-validation for all modalities..."):
        full_comparison = run_all_modalities(df_all)

    st.dataframe(full_comparison, use_container_width=True)

with tab2:
    st.subheader(f"Detailed Results: {model_name} with {modality}")

    # Run CV for selected model
    @st.cache_data(show_spinner=False)
    def run_single_model(_df, model, mod):
        return run_cv(_df, model, mod, n_splits=5)

    with st.spinner(f"Running cross-validation for {model_name}..."):
        fold_df, cm = run_single_model(df_all, model_name, modality)

    # Per-fold results
    st.markdown("#### Per-Fold Results")
    display_fold = fold_df.copy()
    display_fold.columns = ["Fold", "Bal. Acc", "F1", "AUC"]
    st.dataframe(display_fold, use_container_width=True)

    # Summary
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Summary Statistics")
        st.metric("Mean Balanced Accuracy", f"{fold_df['bal_acc'].mean():.2f} ± {fold_df['bal_acc'].std():.2f}")
        st.metric("Mean F1 Score", f"{fold_df['f1'].mean():.2f} ± {fold_df['f1'].std():.2f}")
        st.metric("Mean ROC-AUC", f"{fold_df['auc'].mean():.2f} ± {fold_df['auc'].std():.2f}")

    with col2:
        st.markdown("#### Aggregated Confusion Matrix")

        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Rest", "Clip"], y=["Rest", "Clip"],
            colorscale="Blues", showscale=False,
            text=[[f"{cm[i,j]}" for j in range(2)] for i in range(2)],
            texttemplate="%{text}", textfont=dict(size=16)
        ))
        fig_cm.update_layout(height=250, xaxis_title="Predicted", yaxis_title="True",
                             yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cm, use_container_width=True)

        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"- Rest Sensitivity: {tn/(tn+fp):.2f}")
        st.markdown(f"- Clip Sensitivity: {tp/(tp+fn):.2f}")

    # Feature importance (only for RF and XGB)
    st.markdown("#### Feature Importance")

    if model_name in ["RandomForest", "XGBoost"]:
        feature_cols = get_feature_cols(df_all, modality)
        X = df_all[feature_cols].replace([np.inf, -np.inf], np.nan)
        y = df_all["label"].values

        model = get_model(model_name)
        model.fit(X, y)

        clf = model.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importance = clf.feature_importances_
            imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importance})
            imp_df = imp_df.sort_values("Importance", ascending=True).tail(15)

            fig_imp = go.Figure(go.Bar(
                x=imp_df["Importance"], y=imp_df["Feature"],
                orientation="h", marker_color="#3498db"
            ))
            fig_imp.update_layout(height=400, xaxis_title="Importance")
            st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance only available for RandomForest and XGBoost")
