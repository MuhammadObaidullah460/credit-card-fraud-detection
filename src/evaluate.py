"""
src/evaluate.py
---------------
Final evaluation of the saved model on the held-out test set.

Run AFTER train.py and threshold.py:
    python src/evaluate.py

Outputs
-------
- reports/figures/pr_curves.png        : PR curves for all models
- reports/figures/roc_curves.png       : ROC curves
- reports/figures/confusion_matrix.png : confusion matrix at chosen threshold
- Prints final test-set metrics
"""

import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    fbeta_score,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import get_feature_matrix

REPORTS_DIR   = ROOT / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"
MODELS_DIR    = ROOT / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_PATH = REPORTS_DIR / "optimal_threshold.txt"
MODEL_PATH     = MODELS_DIR / "final_model.joblib"
TEST_PATH      = REPORTS_DIR / "test_split.csv"


# ─── Plotting helpers ─────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.family":      "DejaVu Sans",
    "font.size":        11,
}
plt.rcParams.update(STYLE)


def _save(fig, path: pathlib.Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Core evaluation ──────────────────────────────────────────────────────────
def evaluate_final(threshold: float | None = None) -> None:
    """Load the saved model and evaluate it on the test split."""

    # Load artifacts
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_test, y_test = get_feature_matrix(test_df)

    # Read optimal threshold if not supplied
    if threshold is None and THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            threshold = float(f.read().strip())
        print(f"Using optimal threshold from file: {threshold:.4f}")
    elif threshold is None:
        threshold = 0.5
        print("No threshold file found; using default 0.5")

    # Predicted probabilities and hard labels
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_opt = (y_prob >= threshold).astype(int)
    y_pred_05  = (y_prob >= 0.5).astype(int)

    pr_auc  = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 55)
    print("FINAL TEST-SET EVALUATION")
    print("=" * 55)
    print(f"  PR-AUC  (Average Precision) : {pr_auc:.4f}")
    print(f"  ROC-AUC                     : {roc_auc:.4f}")

    for name, y_pred in [("threshold=0.50", y_pred_05),
                          (f"threshold={threshold:.4f}", y_pred_opt)]:
        f1  = f1_score(y_test, y_pred, zero_division=0)
        f2  = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
        print(f"\n  --- {name} ---")
        print(f"  F1 : {f1:.4f}   F2 : {f2:.4f}")
        print(classification_report(y_test, y_pred,
              target_names=["Genuine", "Fraud"], zero_division=0))

    # ── Precision-Recall curve ─────────────────────────────────────────────────
    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(
        y_test, y_prob
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_arr, precision_arr, lw=2,
            color="#58a6ff", label=f"Final XGBoost (AP={pr_auc:.3f})")
    # baseline: random classifier
    baseline = y_test.mean()
    ax.axhline(baseline, ls="--", color="#f85149",
               label=f"No-skill baseline ({baseline:.4f})")
    ax.scatter(
        [recall_arr[np.argmin(np.abs(thresholds_arr - threshold))]],
        [precision_arr[np.argmin(np.abs(thresholds_arr - threshold))]],
        s=100, zorder=5, color="#3fb950",
        label=f"Operating point (t={threshold:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Test Set")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.4)
    _save(fig, FIGURES_DIR / "pr_curves.png")

    # ── ROC curve ──────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="#58a6ff",
            label=f"Final XGBoost (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#8b949e", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Test Set")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.4)
    _save(fig, FIGURES_DIR / "roc_curves.png")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred_opt)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred Genuine", "Pred Fraud"],
        yticklabels=["True Genuine", "True Fraud"],
        linewidths=0.5, linecolor="#30363d", ax=ax,
    )
    ax.set_title(f"Confusion Matrix (threshold={threshold:.3f})")
    _save(fig, FIGURES_DIR / "confusion_matrix.png")

    print("\nAll figures saved to reports/figures/")
    print("=" * 55)

    return pr_auc, roc_auc


if __name__ == "__main__":
    evaluate_final()
