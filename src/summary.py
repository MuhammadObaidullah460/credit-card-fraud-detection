"""
src/summary.py
--------------
Prints a short results summary after the full pipeline has run.
Numbers are read from generated files (never hand-typed).

Run from project root:
    python src/summary.py

Output is ready to copy into a CV bullet point.
"""

import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPORTS_DIR  = ROOT / "reports"
MODELS_DIR   = ROOT / "models"


def main():
    import pandas as pd
    import joblib
    import numpy as np
    from sklearn.metrics import (
        average_precision_score, roc_auc_score,
        precision_score, recall_score, f1_score, fbeta_score,
        confusion_matrix,
    )
    from features import get_feature_matrix

    # ── Load results table ─────────────────────────────────────────────────────
    results_path = REPORTS_DIR / "results.csv"
    if not results_path.exists():
        print("reports/results.csv not found. Run train.py first.")
        return

    results_df = pd.read_csv(results_path)
    # Best model row = highest PR_AUC_mean
    best_row = results_df.loc[results_df["PR_AUC_mean"].idxmax()]

    # ── Load model and test set ────────────────────────────────────────────────
    model_path = MODELS_DIR / "final_model.joblib"
    test_path  = REPORTS_DIR / "test_split.csv"

    if not model_path.exists() or not test_path.exists():
        print("Model or test split not found. Run train.py first.")
        return

    model   = joblib.load(model_path)
    test_df = pd.read_csv(test_path)
    X_test, y_test = get_feature_matrix(test_df)

    y_prob = model.predict_proba(X_test)[:, 1]
    pr_auc  = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    # ── Load threshold ─────────────────────────────────────────────────────────
    threshold_path = REPORTS_DIR / "optimal_threshold.txt"
    opt_t = 0.5
    if threshold_path.exists():
        with open(threshold_path) as f:
            opt_t = float(f.read().strip())

    y_pred = (y_prob >= opt_t).astype(int)
    prec   = precision_score(y_test, y_pred, zero_division=0)
    rec    = recall_score(y_test, y_pred, zero_division=0)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    f2     = fbeta_score(y_test, y_pred, beta=2, zero_division=0)

    # ── Cost comparison ────────────────────────────────────────────────────────
    FP_COST = 5.0
    FN_MULT = 1.0
    amounts = test_df["Amount"].values

    def cost(y_true, y_pred, amounts):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fn_mask = (np.asarray(y_true) == 1) & (np.asarray(y_pred) == 0)
        return FN_MULT * amounts[fn_mask].sum() + FP_COST * fp

    cost_opt  = cost(y_test, y_pred, amounts)
    y_pred_05 = (y_prob >= 0.5).astype(int)
    cost_base = cost(y_test, y_pred_05, amounts)
    cost_delta_pct = (cost_base - cost_opt) / cost_base * 100 if cost_base > 0 else 0.0

    # ── Print summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  FRAUD DETECTION PROJECT - RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Best model       : {best_row['model']}")
    print(f"  PR-AUC  (CV)     : {best_row['PR_AUC_mean']:.4f} +/- {best_row['PR_AUC_std']:.4f}")
    print(f"  PR-AUC  (test)   : {pr_auc:.4f}")
    print(f"  ROC-AUC (test)   : {roc_auc:.4f}")
    print()
    print(f"  Decision threshold (optimal): {opt_t:.4f}")
    print(f"  Precision @ threshold : {prec:.4f}")
    print(f"  Recall    @ threshold : {rec:.4f}")
    print(f"  F1        @ threshold : {f1:.4f}")
    print(f"  F2        @ threshold : {f2:.4f}")
    print()
    print(f"  Expected cost reduction vs t=0.5: {cost_delta_pct:.1f}%")
    print(f"  (ILLUSTRATIVE cost model: FN={FN_MULT}xAmount, FP={FP_COST} units)")
    print()
    print("  -- CV bullet point (copy to resume) -----------------------")
    print(f"  Built XGBoost fraud detector on 285K imbalanced transactions;")
    print(f"  PR-AUC {pr_auc:.3f} on held-out test set (CV: {best_row['PR_AUC_mean']:.3f} +/- {best_row['PR_AUC_std']:.3f});")
    print(f"  recall {rec:.0%} at precision {prec:.0%} via cost-based threshold selection.")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
