"""
src/explain.py
--------------
SHAP-based explainability for the final tree model.

Run from project root:
    python src/explain.py

Outputs
-------
- reports/figures/shap_summary.png         : global feature importance
- reports/figures/shap_waterfall_*.png     : individual explanations (3 examples)

HONESTY NOTE
------------
V1-V28 are PCA-transformed, anonymized features. SHAP correctly tells us
WHICH latent components matter most, but we cannot interpret *what* V3 or V12
represent in business terms. Individual explanations are therefore limited
to "this feature pushed the prediction up/down by X" without semantic meaning.
"""

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

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import get_feature_matrix, FEATURE_COLS

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH  = ROOT / "models" / "final_model.joblib"
TEST_PATH   = REPORTS_DIR / "test_split.csv"

STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
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


def explain() -> None:
    import shap

    # Load model and test data
    pipeline = joblib.load(MODEL_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    X_test, y_test = get_feature_matrix(test_df)

    print("Transforming test data through preprocessor ...")
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier   = pipeline.named_steps["classifier"]

    X_test_prep = preprocessor.transform(X_test)

    # SHAP wants a DataFrame with column names; reconstruct after transform
    # ColumnTransformer: scaled cols come first, then V1-V28 (passthrough)
    from features import SCALE_COLS, V_COLS
    col_names = SCALE_COLS + V_COLS
    X_test_prep_df = pd.DataFrame(X_test_prep, columns=col_names)

    print("Computing SHAP values (TreeExplainer - fast for tree models) ...")
    explainer   = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_prep_df)

    # XGBoost returns a single array; for binary classification this is the
    # positive-class SHAP values
    if isinstance(shap_values, list):
        sv = shap_values[1]   # positive class
    else:
        sv = shap_values

    # ── 1. Global summary plot (beeswarm) ──────────────────────────────────────
    print("Plotting SHAP summary ...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        sv, X_test_prep_df,
        plot_type="dot",
        max_display=20,
        show=False,
        color_bar=True,
    )
    plt.title("SHAP Summary Plot — Feature Impact on Fraud Probability",
              color="#c9d1d9", pad=12)
    plt.tight_layout()
    _save(plt.gcf(), FIGURES_DIR / "shap_summary.png")

    # ── 2. Three individual waterfall plots ────────────────────────────────────
    # Pick 3 interesting examples: 1 true positive, 1 false negative, 1 FP
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    threshold_path = REPORTS_DIR / "optimal_threshold.txt"
    t = 0.5
    if threshold_path.exists():
        with open(threshold_path) as f:
            t = float(f.read().strip())

    y_pred = (y_prob >= t).astype(int)
    y_true = y_test.values

    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]

    examples = []
    if len(tp_idx): examples.append(("true_positive",  tp_idx[0]))
    if len(fn_idx): examples.append(("false_negative", fn_idx[0]))
    if len(fp_idx): examples.append(("false_positive", fp_idx[0]))
    examples = examples[:3]

    for label, idx in examples:
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv[idx],
                base_values=explainer.expected_value
                    if not isinstance(explainer.expected_value, list)
                    else explainer.expected_value[1],
                data=X_test_prep_df.iloc[idx].values,
                feature_names=col_names,
            ),
            max_display=15,
            show=False,
        )
        true_lbl = "FRAUD" if y_true[idx] == 1 else "Genuine"
        pred_lbl = "FRAUD" if y_pred[idx] == 1 else "Genuine"
        plt.title(
            f"SHAP Waterfall - {label.replace('_', ' ').title()}\n"
            f"True: {true_lbl}  |  Pred: {pred_lbl}  |  "
            f"P(fraud)={y_prob[idx]:.3f}",
            color="#c9d1d9",
        )
        plt.tight_layout()
        _save(plt.gcf(), FIGURES_DIR / f"shap_waterfall_{label}.png")

    print("\nHonesty note:")
    print("  V1-V28 are anonymized PCA features; their SHAP values show")
    print("  statistical importance but cannot be interpreted in business terms.")
    print("Explainability plots complete.")


if __name__ == "__main__":
    explain()
