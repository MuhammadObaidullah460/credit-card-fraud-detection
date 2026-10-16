"""
src/train.py
------------
Model training, cross-validation, hyperparameter tuning and final model fitting.

Run from project root:
    python src/train.py

Outputs
-------
- reports/results.csv           : model comparison table
- models/final_model.joblib     : fitted final model
"""

import argparse
import pathlib
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

# Suppress harmless convergence / deprecation warnings in tuning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    make_scorer,
)
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    cross_validate,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

# ── project modules ────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_raw, audit, print_audit, drop_duplicates, stratified_split
from features import (
    FEATURE_COLS,
    TARGET_COL,
    get_feature_matrix,
    build_preprocessor,
    build_pipeline,
)

# ── paths ──────────────────────────────────────────────────────────────────────
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
CV_FOLDS = 5
N_ITER_SEARCH = 30           # RandomizedSearchCV iterations per model


# ─── Scorer ───────────────────────────────────────────────────────────────────
ap_scorer = make_scorer(average_precision_score, response_method="predict_proba",
                        pos_label=1)
roc_scorer = make_scorer(roc_auc_score, response_method="predict_proba")
f2_scorer  = make_scorer(fbeta_score, beta=2, pos_label=1, zero_division=0)


# ─── Helper: evaluate a pipeline with CV ──────────────────────────────────────
def cv_evaluate(pipeline, X, y, label: str) -> dict:
    """Run stratified K-fold CV and return mean ± std of key metrics."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    scoring = {
        "avg_precision": ap_scorer,
        "roc_auc": roc_scorer,
        "f2": f2_scorer,
    }
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring,
                            n_jobs=-1, return_train_score=False)
    result = {
        "model": label,
        "PR_AUC_mean": round(float(np.mean(scores["test_avg_precision"])), 4),
        "PR_AUC_std":  round(float(np.std( scores["test_avg_precision"])), 4),
        "ROC_AUC_mean":round(float(np.mean(scores["test_roc_auc"])), 4),
        "ROC_AUC_std": round(float(np.std( scores["test_roc_auc"])), 4),
        "F2_mean":     round(float(np.mean(scores["test_f2"])), 4),
        "F2_std":      round(float(np.std( scores["test_f2"])), 4),
    }
    print(
        f"  {label:50s}  "
        f"PR-AUC: {result['PR_AUC_mean']:.4f} +/- {result['PR_AUC_std']:.4f}  "
        f"ROC-AUC: {result['ROC_AUC_mean']:.4f}"
    )
    return result


# ─── Build all candidate pipelines ────────────────────────────────────────────
def build_candidates(class_ratio: float) -> list[tuple[str, object]]:
    """
    Return a list of (label, pipeline) tuples for all model × strategy combos.

    class_ratio : n_majority / n_minority  (used for scale_pos_weight in XGBoost)
    """
    candidates = []

    # ── 1. Baselines ───────────────────────────────────────────────────────────
    # Majority-class dummy — shows why accuracy alone is misleading
    candidates.append((
        "Dummy (majority class)",
        build_pipeline(DummyClassifier(strategy="most_frequent",
                                       random_state=RANDOM_STATE))
    ))
    # Logistic Regression with balanced weights (simple baseline)
    candidates.append((
        "LogReg (balanced weights)",
        build_pipeline(LogisticRegression(
            class_weight="balanced", max_iter=1000,
            solver="lbfgs", random_state=RANDOM_STATE))
    ))

    # ── 2. Random Forest ───────────────────────────────────────────────────────
    # Using n_jobs=1 inside estimators to avoid OpenMP thread oversubscription
    # when outer cross_validate uses n_jobs=-1
    candidates.append((
        "RandomForest (no resampling)",
        build_pipeline(RandomForestClassifier(
            n_estimators=200, class_weight=None,
            n_jobs=1, random_state=RANDOM_STATE))
    ))
    candidates.append((
        "RandomForest (balanced weights)",
        build_pipeline(RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=1, random_state=RANDOM_STATE))
    ))

    # RF + SMOTE — resampling inside an imbalanced-learn Pipeline
    smote = SMOTE(random_state=RANDOM_STATE)
    candidates.append((
        "RandomForest + SMOTE",
        ImbPipeline(steps=[
            ("preprocessor", build_preprocessor("standard")),
            ("smote", smote),
            ("classifier", RandomForestClassifier(
                n_estimators=200, n_jobs=1, random_state=RANDOM_STATE)),
        ])
    ))

    # RF + Random Under-Sampling
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    candidates.append((
        "RandomForest + RUS",
        ImbPipeline(steps=[
            ("preprocessor", build_preprocessor("standard")),
            ("rus", rus),
            ("classifier", RandomForestClassifier(
                n_estimators=200, n_jobs=1, random_state=RANDOM_STATE)),
        ])
    ))

    # ── 3. XGBoost ────────────────────────────────────────────────────────────
    candidates.append((
        "XGBoost (no resampling)",
        build_pipeline(xgb.XGBClassifier(
            n_estimators=200,
            eval_metric="aucpr", random_state=RANDOM_STATE,
            n_jobs=1, verbosity=0))
    ))
    candidates.append((
        "XGBoost (scale_pos_weight)",
        build_pipeline(xgb.XGBClassifier(
            n_estimators=200, scale_pos_weight=class_ratio,
            eval_metric="aucpr",
            random_state=RANDOM_STATE, n_jobs=1, verbosity=0))
    ))
    candidates.append((
        "XGBoost + SMOTE",
        ImbPipeline(steps=[
            ("preprocessor", build_preprocessor("standard")),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", xgb.XGBClassifier(
                n_estimators=200,
                eval_metric="aucpr", random_state=RANDOM_STATE,
                n_jobs=1, verbosity=0)),
        ])
    ))

    return candidates


# ─── Hyperparameter tuning ────────────────────────────────────────────────────
def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                 class_ratio: float) -> tuple[object, dict]:
    """
    RandomizedSearchCV for XGBoost with scale_pos_weight.
    Optimises average_precision (PR-AUC) with stratified 5-fold CV.

    Returns (best_pipeline, best_params).
    """
    print("\nTuning XGBoost (RandomizedSearchCV, 20 iterations) ...")

    pipeline = build_pipeline(
        xgb.XGBClassifier(
            eval_metric="aucpr",
            n_jobs=1, verbosity=0, random_state=RANDOM_STATE,
        )
    )

    param_dist = {
        "classifier__n_estimators":       [150, 200, 300],
        "classifier__max_depth":          [3, 4, 5, 6],
        "classifier__learning_rate":      [0.02, 0.05, 0.1, 0.15],
        "classifier__subsample":          [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree":   [0.7, 0.8, 0.9, 1.0],
        "classifier__min_child_weight":   [1, 3, 5],
        "classifier__gamma":              [0, 0.1, 0.2],
        "classifier__reg_alpha":          [0, 0.01, 0.1, 1.0],
        "classifier__scale_pos_weight":   [1, class_ratio * 0.5,
                                           class_ratio, class_ratio * 1.5],
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist,
        n_iter=20, scoring="average_precision",
        cv=cv, n_jobs=-1, random_state=RANDOM_STATE,
        refit=True, verbose=1,
    )
    search.fit(X_train, y_train)
    best_params = search.best_params_
    print(f"  Best CV PR-AUC: {search.best_score_:.4f}")
    print(f"  Best params   : {best_params}")
    return search.best_estimator_, best_params


# ─── Main training routine ────────────────────────────────────────────────────
def train(args) -> None:
    # 1. Load and audit
    print("Loading data ...")
    df = load_raw()
    info = audit(df)
    print_audit(info)

    # 2. Drop duplicates
    df = drop_duplicates(df)

    # 3. Stratified split
    train_df, val_df, test_df = stratified_split(
        df, val_size=0.15, test_size=0.15, random_state=RANDOM_STATE
    )

    X_train, y_train = get_feature_matrix(train_df)
    X_val,   y_val   = get_feature_matrix(val_df)
    X_test,  y_test  = get_feature_matrix(test_df)

    # class imbalance ratio for scale_pos_weight
    n0 = (y_train == 0).sum()
    n1 = (y_train == 1).sum()
    class_ratio = n0 / n1
    print(f"\nTrain class ratio (genuine/fraud): {class_ratio:.1f}")

    # 4. Baseline accuracy check (why accuracy is misleading)
    majority_acc = (y_train == 0).mean()
    print(f"\nMajority-class accuracy (train): {majority_acc:.4f} "
          f"-> ~99.8 % accuracy without detecting any fraud!")

    # 5. Cross-validate all candidates
    print("\n--- Cross-validation comparison ---")
    candidates = build_candidates(class_ratio)
    rows = []
    for label, pipe in candidates:
        row = cv_evaluate(pipe, X_train, y_train, label)
        rows.append(row)

    results_df = pd.DataFrame(rows)

    # 6. Hyperparameter tuning (XGBoost)
    best_pipeline, best_params = tune_xgboost(X_train, y_train, class_ratio)

    # Evaluate tuned model with CV
    tuned_row = cv_evaluate(best_pipeline, X_train, y_train,
                            "XGBoost (tuned, RandomizedSearch)")
    results_df = pd.concat([results_df, pd.DataFrame([tuned_row])],
                            ignore_index=True)

    # Save results table
    results_path = REPORTS_DIR / "results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved -> {results_path}")
    print(results_df.to_string(index=False))

    # 7. Fit final model on train split (preserving val_df as an unseen holdout for threshold.py)
    print("\n--- Fitting final model on train split ---")
    best_pipeline.fit(X_train, y_train)

    # 8. Save
    model_path = MODELS_DIR / "final_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"Final model saved -> {model_path}")

    # 9. Persist test set for evaluate.py and threshold.py
    test_df.to_csv(REPORTS_DIR / "test_split.csv", index=False)
    val_df.to_csv(REPORTS_DIR / "val_split.csv", index=False)
    print("Test and validation splits saved to reports/")

    print("\nTraining complete. Run evaluate.py next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    args = parser.parse_args()
    train(args)
