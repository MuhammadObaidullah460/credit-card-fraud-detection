# Credit Card Fraud Detection

An end-to-end machine learning project demonstrating sound handling of extreme class imbalance, leakage-free preprocessing, business-aware threshold selection, and honest reporting. Built for a data science portfolio.

---

## Overview and Problem Statement

This project trains a binary classifier to flag fraudulent credit card transactions. The core challenge is a **severe class imbalance**: only 492 of 284,807 transactions (0.172%) are fraudulent. A naive model that predicts every transaction as genuine achieves ~99.83% accuracy while catching zero frauds — making accuracy useless as a metric.

Key decisions documented here:
- **Primary metric**: PR-AUC (Average Precision), because it focuses entirely on the minority class and is insensitive to the huge number of true negatives.
- **Imbalance handling**: Compared class weights, SMOTE, and random undersampling inside cross-validation folds to avoid leakage.
- **Threshold**: Selected on the validation set by minimising a configurable cost model (not 0.5 by default).

---

## Dataset

**Source**: [Kaggle — ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Mirror**: [Zenodo record 7395559](https://zenodo.org/records/7395559)

| Property | Value |
|----------|-------|
| Rows | 284,807 |
| Fraud rows | 492 (0.172 %) |
| Columns | Time, V1-V28, Amount, Class |
| Period | Two days, September 2013 |
| Geography | European cardholders |

### Download Instructions

The CSV is ~150 MB and cannot be committed to GitHub. Download it manually:

```powershell
# Option 1 — Kaggle CLI (requires API token at ~/.kaggle/kaggle.json)
.\.venv\Scripts\kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p data\

# Option 2 — Manual
# 1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# 2. Click Download → creditcardfraud.zip
# 3. Extract creditcard.csv into the data\ folder
```

The `data/` folder is in `.gitignore`. Never commit the CSV.

---

## Approach

### Why PR-AUC?

With 0.172% positives, ROC-AUC is over-optimistic because it counts true negatives (which are easy to get right). PR-AUC only measures performance on the positive class, which is what actually matters for fraud detection.

### Why not just use accuracy?

A dummy model predicting "Genuine" for everything scores ~99.83% accuracy and catches **zero frauds**. This project demonstrates this explicitly in the notebook before proceeding to meaningful metrics.

### Split strategy

Stratified 70/15/15 split (train / val / test), random_state=42. A time-ordered split is also available as a robustness check (`data.py:time_ordered_split`). The test set is never used until final evaluation.

### Preprocessing

Only `Amount` and `Time` are scaled (StandardScaler). V1-V28 are already PCA-transformed. All preprocessing is inside sklearn/imblearn Pipelines — the scaler and resampler are fitted only on the training fold inside each CV iteration.

### Imbalance strategies compared

| Strategy | Implementation |
|----------|---------------|
| None | Default XGBoost |
| Class weights | `class_weight="balanced"` / `scale_pos_weight` |
| SMOTE | `imblearn.over_sampling.SMOTE` inside Pipeline |
| Random undersampling | `imblearn.under_sampling.RandomUnderSampler` inside Pipeline |

### Hyperparameter tuning

`RandomizedSearchCV` with 30 iterations, 5-fold stratified CV, optimising `average_precision`. Parameters searched: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, scale_pos_weight.

---

## Results

The full pipeline has been executed end-to-end. Below are the authentic results from 5-fold stratified cross-validation and the final held-out test set evaluation:

### 5-Fold Cross-Validation Comparison

| Model | PR-AUC (mean ± std) | ROC-AUC (mean ± std) | F2 (mean ± std) |
|---|---|---|---|
| **Dummy (majority class)** | 0.0017 ± 0.0000 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 |
| **Logistic Regression (balanced)** | 0.7316 ± 0.0329 | 0.9831 ± 0.0112 | 0.2357 ± 0.0077 |
| **Random Forest (no resampling)** | 0.8419 ± 0.0544 | 0.9538 ± 0.0257 | 0.7976 ± 0.0321 |
| **Random Forest (balanced weights)** | 0.8414 ± 0.0477 | 0.9619 ± 0.0233 | 0.8002 ± 0.0402 |
| **Random Forest + SMOTE** | 0.8434 ± 0.0548 | 0.9779 ± 0.0154 | 0.8186 ± 0.0493 |
| **Random Forest + RUS** | 0.7361 ± 0.0130 | 0.9768 ± 0.0129 | 0.2287 ± 0.0279 |
| **XGBoost (no resampling)** | 0.2817 ± 0.3404 | 0.6733 ± 0.1728 | 0.3286 ± 0.3263 |
| **XGBoost (scale_pos_weight)** | 0.8444 ± 0.0582 | 0.9788 ± 0.0139 | 0.8227 ± 0.0465 |
| **XGBoost + SMOTE** | 0.8463 ± 0.0485 | 0.9813 ± 0.0113 | 0.8090 ± 0.0454 |
| **XGBoost (tuned, RandomizedSearch)** | **0.8527 ± 0.0554** | **0.9829 ± 0.0092** | **0.8297 ± 0.0408** |

### Held-Out Test Set Performance (Final Model)

- **Test PR-AUC**: `0.8161`
- **Test ROC-AUC**: `0.9726`
- **Optimal Decision Threshold**: `0.1200` (tuned on validation set)
- **Precision @ threshold**: `79.45%`
- **Recall @ threshold**: `81.69%` (catches 58 of 71 fraudulent transactions in test set)
- **F1 Score**: `0.8056`
- **F2 Score (recall-oriented)**: `0.8123`

### Key Artifacts and Figures

| Figure | Description |
|--------|-------------|
| `reports/figures/class_imbalance.png` | Bar + pie chart of severe 0.172% class imbalance |
| `reports/figures/pr_curves_validation.png` | Precision-Recall curve on validation set |
| `reports/figures/cost_curve.png` | Total cost curve vs decision threshold |
| `reports/figures/threshold_metrics.png` | Precision, recall, and F1 tradeoffs across thresholds |
| `reports/figures/pr_curves.png` | Final model Precision-Recall curve on test set |
| `reports/figures/roc_curves.png` | Final model ROC curve on test set |
| `reports/figures/confusion_matrix.png` | Confusion matrix at optimal threshold (t=0.120) |
| `reports/figures/shap_summary.png` | SHAP beeswarm feature importance |
| `reports/figures/shap_waterfall_*.png` | Individual transaction waterfall explanations (TP, FN, FP) |

---

## Cost-Based Threshold Analysis

**IMPORTANT: The cost values below are ILLUSTRATIVE ASSUMPTIONS, not real bank figures. Actual fraud investigation costs and loss absorption policies vary by institution and are confidential.**

The model uses a simple cost model:

```
total_cost = Σ(FN_MULTIPLIER × Amount_i for missed frauds)
           + FP_COST × (number of false alarms)
```

Default assumptions:
- `FN_MULTIPLIER = 1.0` — bank absorbs the full fraudulent amount
- `FP_COST = 5.0` — fixed cost per flagged genuine transaction (analyst review time)

These assumptions can be changed via command-line:

```powershell
.\.venv\Scripts\python src\threshold.py --fn_multiplier 1.0 --fp_cost 10.0
```

The optimal threshold is selected on the **validation set only**, then the expected cost difference vs the default t=0.5 is reported on the **test set**. This avoids using the test set for threshold selection.

---

## Explainability

SHAP TreeExplainer is used to explain the final XGBoost model.

- **`shap_summary.png`**: Shows which features push predictions toward fraud (red) or genuine (blue) and by how much, averaged across the test set.
- **`shap_waterfall_*.png`**: Three individual transaction explanations — a true positive, a false negative, and a false positive.

**Honest caveat**: V1-V28 are PCA-transformed, anonymized features. SHAP correctly identifies which latent components matter statistically, but assigning business meaning (e.g. "V14 represents merchant category") is not possible from this dataset alone.

---

## Limitations

1. **Anonymized features**: V1-V28 are PCA components. We cannot interpret them in business terms, which limits actionability of individual predictions.
2. **Two days of data**: The dataset covers only 48 hours in September 2013. Fraud patterns evolve (concept drift), so models trained on this data would need frequent retraining in production.
3. **2013 European cardholders only**: Results may not generalise to other geographies, time periods, or card types.
4. **Only 492 positives**: With so few fraud examples, metric confidence intervals are wide. A 1% change in PR-AUC could be noise.
5. **Cost assumptions are illustrative**: The threshold analysis shows the *method*, not real bank economics.
6. **No additional features**: No merchant category, cardholder history, device fingerprint, or geolocation — all of which real fraud systems use.

---

## How to Run

### Setup (Windows PowerShell)

```powershell
# Clone the repo
git clone <your-repo-url>
cd Credit-Card-Fraud-Detection

# Create and activate virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download the dataset (see Dataset section above)
# Place creditcard.csv in data\
```

### Run the pipeline in order

```powershell
# Step 1: Exploratory Data Analysis
# Open notebooks\01_eda.ipynb in Jupyter and run all cells
.\.venv\Scripts\jupyter notebook notebooks\01_eda.ipynb

# Step 2: Train all models, tune XGBoost, save final model
.\.venv\Scripts\python src\train.py

# Step 3: Select optimal decision threshold via cost model
.\.venv\Scripts\python src\threshold.py

# Step 4: Final evaluation on test set
.\.venv\Scripts\python src\evaluate.py

# Step 5: SHAP explanations
.\.venv\Scripts\python src\explain.py

# Optional: Full modelling notebook
.\.venv\Scripts\jupyter notebook notebooks\02_modeling.ipynb

# Optional: Streamlit demo app
.\.venv\Scripts\streamlit run app\app.py
```

### Expected output files after full run

```
reports/results.csv                     # model comparison table
reports/optimal_threshold.txt           # chosen threshold float
reports/figures/class_imbalance.png
reports/figures/amount_distribution.png
reports/figures/fraud_rate_by_hour.png
reports/figures/top_v_feature_distributions.png
reports/figures/feature_correlation.png
reports/figures/pr_curves_validation.png
reports/figures/cost_curve.png
reports/figures/threshold_metrics.png
reports/figures/pr_curves.png
reports/figures/roc_curves.png
reports/figures/confusion_matrix.png
reports/figures/shap_summary.png
reports/figures/shap_waterfall_true_positive.png
reports/figures/shap_waterfall_false_negative.png
reports/figures/shap_waterfall_false_positive.png
models/final_model.joblib
```

---

## Repo Structure

```
Credit-Card-Fraud-Detection/
├── data/                        # gitignored — place creditcard.csv here
├── notebooks/
│   ├── 01_eda.ipynb             # exploratory data analysis
│   └── 02_modeling.ipynb        # full modelling pipeline
├── src/
│   ├── data.py                  # loading, auditing, splitting
│   ├── features.py              # preprocessing pipelines
│   ├── train.py                 # model training and CV
│   ├── threshold.py             # cost-based threshold selection
│   ├── evaluate.py              # final test-set evaluation
│   └── explain.py               # SHAP explanations
├── reports/
│   ├── results.csv              # generated model comparison table
│   └── figures/                 # all plots (PNG, 150 dpi)
├── models/
│   └── final_model.joblib       # saved final model
├── app/
│   └── app.py                   # Streamlit demo
├── .gitignore
├── LICENSE                      # MIT
├── README.md
└── requirements.txt             # pinned versions
```

---

## License

MIT — see [LICENSE](LICENSE).
