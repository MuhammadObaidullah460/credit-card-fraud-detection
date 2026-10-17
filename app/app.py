r"""
app/app.py
----------
Streamlit demo: upload a CSV of transactions and see fraud probabilities.

Run from project root:
    .\.venv\Scripts\streamlit run app/app.py

Expected CSV columns: Time, V1-V28, Amount  (same as creditcard.csv minus Class)
"""

import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import FEATURE_COLS, SCALE_COLS, V_COLS

MODEL_PATH     = ROOT / "models" / "final_model.joblib"
THRESHOLD_PATH = ROOT / "reports" / "optimal_threshold.txt"


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background-color: #0d1117; color: #c9d1d9; }
    .metric-card {
        background: linear-gradient(135deg, #161b22, #1f2937);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .fraud-high   { color: #f85149; font-weight: 700; }
    .fraud-low    { color: #3fb950; font-weight: 700; }
    .badge-fraud  { background: #f85149; color: white; padding: 2px 10px;
                    border-radius: 20px; font-size: 0.85em; }
    .badge-legit  { background: #3fb950; color: white; padding: 2px 10px;
                    border-radius: 20px; font-size: 0.85em; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    threshold = 0.5
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            threshold = float(f.read().strip())
    return model, threshold


model, threshold = load_model()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Fraud Detector")
    st.markdown("---")

    if model is None:
        st.error("Model not found. Run `python src/train.py` first.")
    else:
        st.success("Model loaded ✓")
        st.metric("Decision threshold", f"{threshold:.4f}")
        st.markdown("---")

    st.markdown("**Cost assumptions** *(illustrative)*")
    fp_cost = st.slider("False alarm cost (currency units)", 1.0, 50.0, 5.0, 0.5)
    fn_mult = st.slider("Missed fraud cost (× Amount)",     0.1,  2.0, 1.0, 0.1)

    st.markdown("---")
    st.markdown(
        "**Dataset**: [Kaggle ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  \n"
        "**Source**: 2013 European card transactions  \n"
        "**Note**: V1-V28 are anonymized PCA features"
    )


# ─── Main content ─────────────────────────────────────────────────────────────
st.markdown("# 🛡️ Credit Card Fraud Detection")
st.markdown(
    "Upload a CSV of transactions (columns: **Time, V1-V28, Amount**). "
    "The model will score each transaction and flag likely frauds at the "
    f"selected threshold (**{threshold:.4f}**)."
)

if model is None:
    st.warning(
        "No trained model found at `models/final_model.joblib`.  \n"
        "Please run the pipeline first: `python src/train.py`"
    )
    st.stop()

# ─── File upload ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload transactions CSV", type=["csv"],
    help="Must have columns: Time, V1-V28, Amount (Class column optional)"
)

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.markdown(f"**{len(df):,} transactions loaded.**")

        # Check required columns
        missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            st.stop()

        X = df[FEATURE_COLS]
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        df_out = df.copy()
        df_out["fraud_probability"] = np.round(y_prob, 4)
        df_out["flagged_as_fraud"]  = y_pred

        n_flagged = y_pred.sum()
        total     = len(df)

        # ── Summary metrics ────────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", f"{total:,}")
        with col2:
            st.metric("Flagged as Fraud", f"{n_flagged:,}",
                      delta=f"{n_flagged/total*100:.2f}%")
        with col3:
            avg_prob = float(y_prob[y_pred == 1].mean()) if n_flagged else 0.0
            st.metric("Avg Fraud Probability (flagged)", f"{avg_prob:.3f}")

        # ── Results table ──────────────────────────────────────────────────────
        st.markdown("### Results")
        display_cols = ["Time", "Amount", "fraud_probability", "flagged_as_fraud"]
        if "Class" in df_out.columns:
            display_cols.insert(0, "Class")

        def highlight_fraud(row):
            color = "#3f1010" if row["flagged_as_fraud"] == 1 else ""
            return [f"background-color: {color}"] * len(row)

        styled = df_out[display_cols].style.apply(
            highlight_fraud, axis=1
        ).format({"fraud_probability": "{:.4f}"})
        st.dataframe(styled, use_container_width=True)

        # ── Distribution of fraud probabilities ───────────────────────────────
        st.markdown("### Fraud Probability Distribution")
        fig, ax = plt.subplots(figsize=(9, 4),
                                facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        ax.hist(y_prob, bins=50, color="#58a6ff", edgecolor="#0d1117", alpha=0.8)
        ax.axvline(threshold, color="#f85149", ls="--", lw=2,
                   label=f"Threshold={threshold:.3f}")
        ax.set_xlabel("Fraud Probability", color="#c9d1d9")
        ax.set_ylabel("Count", color="#c9d1d9")
        ax.set_title("Distribution of P(fraud)", color="#c9d1d9")
        ax.tick_params(colors="#8b949e")
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        st.pyplot(fig)

        # ── Download ──────────────────────────────────────────────────────────
        csv_out = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Results CSV",
            data=csv_out,
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )

        # ── Cost estimate ──────────────────────────────────────────────────────
        if "Class" in df.columns:
            st.markdown("### Cost Estimate (illustrative)")
            from sklearn.metrics import confusion_matrix as cm_fn
            y_true = df["Class"].values
            tn_, fp_, fn_, tp_ = cm_fn(y_true, y_pred, labels=[0,1]).ravel()
            fn_mask = (y_true == 1) & (y_pred == 0)
            fn_cost_total = fn_mult * df.loc[fn_mask, "Amount"].sum()
            fp_cost_total = fp_ * fp_cost
            total_cost    = fn_cost_total + fp_cost_total
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("FN Cost (missed frauds)", f"{fn_cost_total:,.2f}")
            with col5:
                st.metric("FP Cost (false alarms)",  f"{fp_cost_total:,.2f}")
            with col6:
                st.metric("Total Cost", f"{total_cost:,.2f}")
            st.caption(
                "⚠️ These cost estimates are purely illustrative. "
                "Real bank cost structures are confidential and differ by institution."
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info(
        "👆 Upload a CSV to get started, or use the sample from the "
        "`data/` folder (remove the `Class` column first if you want to "
        "simulate unlabeled production data)."
    )

    # Show example expected format
    st.markdown("### Expected CSV Format")
    sample = pd.DataFrame({
        "Time": [0, 1],
        "V1":   [-1.36, 1.19], "V2": [-0.07, 0.27], "V3": [2.54, 0.17],
        "...":  ["...", "..."],
        "V28":  [0.02, 0.01],
        "Amount": [149.62, 2.69],
    })
    st.dataframe(sample, use_container_width=True)
