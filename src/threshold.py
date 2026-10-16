"""
src/threshold.py
----------------
Business-aware threshold selection using a simple cost model.

IMPORTANT — cost assumptions are ILLUSTRATIVE, not real bank figures.
Users can override via command-line arguments.

Cost model
----------
  cost_missed_fraud  = transaction Amount  (bank absorbs the loss)
  cost_false_alarm   = fixed review cost   (e.g. 5 currency units of analyst time)

The optimal threshold minimises total expected cost on the VALIDATION set.
The test set is then evaluated at this threshold (and at 0.5 for comparison).

Run from project root:
    python src/threshold.py
    python src/threshold.py --fn_multiplier 1.0 --fp_cost 10.0

Outputs
-------
- reports/optimal_threshold.txt            : single float, read by evaluate.py
- reports/figures/cost_curve.png           : cost vs threshold plot
- reports/figures/threshold_metrics.png    : precision/recall/F1 vs threshold
"""

import argparse
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import get_feature_matrix

REPORTS_DIR  = ROOT / "reports"
FIGURES_DIR  = REPORTS_DIR / "figures"
MODELS_DIR   = ROOT / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH     = MODELS_DIR / "final_model.joblib"
VAL_PATH       = REPORTS_DIR / "val_split.csv"
TEST_PATH      = REPORTS_DIR / "test_split.csv"
THRESHOLD_FILE = REPORTS_DIR / "optimal_threshold.txt"

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


def compute_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    fp_cost: float,
    fn_multiplier: float,
) -> float:
    """
    Compute total cost under the simple cost model.

    False negatives (missed frauds): cost = fn_multiplier * Amount
    False positives (false alarms):  cost = fp_cost per flagged transaction
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn_idx, tp = cm.ravel()

    # Identify which rows are false negatives
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_cost = fn_multiplier * amounts[fn_mask].sum()

    # False positives: fixed review cost per flagged genuine transaction
    fp_cost_total = fp * fp_cost

    return float(fn_cost + fp_cost_total)


def threshold_analysis(
    fn_multiplier: float = 1.0,
    fp_cost: float = 5.0,
) -> float:
    """
    Select the threshold that minimises expected cost on the validation set.

    Parameters
    ----------
    fn_multiplier : float
        Multiplier on transaction Amount for a missed fraud.
        1.0 means the bank absorbs the full fraudulent amount.
    fp_cost : float
        Fixed cost (in same currency units) for each false alarm.

    Returns
    -------
    optimal_threshold : float
    """
    model   = joblib.load(MODEL_PATH)
    val_df  = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_val, y_val = get_feature_matrix(val_df)
    X_test, y_test = get_feature_matrix(test_df)

    amounts_val  = val_df["Amount"].values
    amounts_test = test_df["Amount"].values

    y_val_prob  = model.predict_proba(X_val)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Sweep thresholds from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 500)
    costs_val = []
    for t in thresholds:
        y_pred = (y_val_prob >= t).astype(int)
        cost = compute_cost(y_val.values, y_pred, amounts_val,
                            fp_cost=fp_cost, fn_multiplier=fn_multiplier)
        costs_val.append(cost)

    costs_val = np.array(costs_val)
    opt_idx   = np.argmin(costs_val)
    opt_t     = float(thresholds[opt_idx])
    opt_cost_val = costs_val[opt_idx]

    # Baseline cost at threshold = 0.5
    y_pred_05  = (y_val_prob >= 0.5).astype(int)
    cost_05_val = compute_cost(y_val.values, y_pred_05, amounts_val,
                               fp_cost=fp_cost, fn_multiplier=fn_multiplier)

    # Test-set costs
    y_pred_opt  = (y_test_prob >= opt_t).astype(int)
    y_pred_05t  = (y_test_prob >= 0.5).astype(int)
    cost_opt_test = compute_cost(y_test.values, y_pred_opt, amounts_test,
                                 fp_cost=fp_cost, fn_multiplier=fn_multiplier)
    cost_05_test  = compute_cost(y_test.values, y_pred_05t, amounts_test,
                                 fp_cost=fp_cost, fn_multiplier=fn_multiplier)

    print("\n" + "=" * 55)
    print("THRESHOLD SELECTION (cost model)")
    print("=" * 55)
    print(f"  Cost assumptions (ILLUSTRATIVE):")
    print(f"    Missed fraud cost  = {fn_multiplier:.2f} × Amount")
    print(f"    False alarm cost   = {fp_cost:.2f} currency units")
    print(f"  Optimal threshold (val)  : {opt_t:.4f}")
    print(f"  Val cost  @ optimal      : {opt_cost_val:,.2f}")
    print(f"  Val cost  @ t=0.50       : {cost_05_val:,.2f}")
    print(f"  Test cost @ optimal      : {cost_opt_test:,.2f}")
    print(f"  Test cost @ t=0.50       : {cost_05_test:,.2f}")
    cost_reduction = (cost_05_test - cost_opt_test) / cost_05_test * 100
    print(f"  Cost reduction on test   : {cost_reduction:.1f}%")

    # Precision and recall at optimal threshold (test)
    from sklearn.metrics import precision_score, recall_score
    prec = precision_score(y_test, y_pred_opt, zero_division=0)
    rec  = recall_score(y_test, y_pred_opt, zero_division=0)
    print(f"\n  At optimal threshold on TEST:")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1        : {f1_score(y_test, y_pred_opt, zero_division=0):.4f}")
    print(f"    F2        : {fbeta_score(y_test, y_pred_opt, beta=2, zero_division=0):.4f}")
    print("=" * 55)

    # Save threshold
    with open(THRESHOLD_FILE, "w") as f:
        f.write(str(opt_t))
    print(f"Optimal threshold saved -> {THRESHOLD_FILE}")

    # ── Plot 1: Cost vs threshold ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, costs_val, lw=2, color="#58a6ff", label="Total cost (val)")
    ax.axvline(opt_t, ls="--", color="#3fb950",
               label=f"Optimal t={opt_t:.3f}  cost={opt_cost_val:,.0f}")
    ax.axvline(0.5,  ls="--", color="#f85149",
               label=f"t=0.50  cost={cost_05_val:,.0f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Total Expected Cost")
    ax.set_title("Cost vs Decision Threshold (Validation Set)\n"
                 f"[FN cost = {fn_multiplier}×Amount; FP cost = {fp_cost}]")
    ax.legend()
    ax.grid(True, alpha=0.4)
    _save(fig, FIGURES_DIR / "cost_curve.png")

    # ── Plot 2: Precision / Recall / F1 vs threshold ──────────────────────────
    from sklearn.metrics import precision_score, recall_score
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        yp = (y_val_prob >= t).astype(int)
        precisions.append(precision_score(y_val, yp, zero_division=0))
        recalls.append(recall_score(y_val, yp, zero_division=0))
        f1s.append(f1_score(y_val, yp, zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, lw=1.8, color="#58a6ff", label="Precision")
    ax.plot(thresholds, recalls,   lw=1.8, color="#3fb950", label="Recall")
    ax.plot(thresholds, f1s,       lw=1.8, color="#d2a8ff", label="F1")
    ax.axvline(opt_t, ls="--", color="#f0883e",
               label=f"Optimal t={opt_t:.3f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold (Validation Set)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    _save(fig, FIGURES_DIR / "threshold_metrics.png")

    # ── Plot 3: Precision-Recall Curve on Validation Set ───────────────────────
    prec_val_arr, rec_val_arr, _ = precision_recall_curve(y_val, y_val_prob)
    val_pr_auc = average_precision_score(y_val, y_val_prob)
    baseline_val = float(y_val.mean())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_val_arr, prec_val_arr, lw=2, color="#58a6ff",
            label=f"Final Model (AP={val_pr_auc:.3f})")
    ax.axhline(baseline_val, ls="--", color="#f85149",
               label=f"No-skill baseline ({baseline_val:.4f})")
    ax.scatter(
        [rec], [prec], s=100, zorder=5, color="#3fb950",
        label=f"Operating point (t={opt_t:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Validation Set")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.4)
    _save(fig, FIGURES_DIR / "pr_curves_validation.png")

    return opt_t


def _save(fig, path: pathlib.Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cost-based threshold selection for fraud detection"
    )
    parser.add_argument(
        "--fn_multiplier", type=float, default=1.0,
        help="Multiplier on Amount for a missed fraud (default: 1.0)",
    )
    parser.add_argument(
        "--fp_cost", type=float, default=5.0,
        help="Fixed cost per false alarm in currency units (default: 5.0)",
    )
    args = parser.parse_args()
    threshold_analysis(fn_multiplier=args.fn_multiplier, fp_cost=args.fp_cost)
