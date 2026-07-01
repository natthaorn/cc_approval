# Credit Card Approval Prediction

![Credit card being held in hand](credit_card.jpg)

A machine learning project that automates credit card approval decisions, the way real commercial banks do. Manually reviewing applications for red flags like high loan balances, low income, or too many credit inquiries is slow and error-prone — this project builds a classifier that does it instead.

## Overview

Commercial banks receive a large volume of credit card applications, and many are rejected for identifiable, data-driven reasons. This project trains a logistic regression model to predict whether an application will be approved, based on applicant attributes such as income, employment status, credit history, and existing debt.

## Dataset

The data (`cc_approvals.data`) is a subset of the [Credit Card Approval dataset](https://archive.ics.uci.edu/dataset/27/credit+approval) from the UCI Machine Learning Repository.

- 690 applications, 14 anonymized attributes (mix of categorical and numeric) plus a target label
- Target column: `+` (approved) / `-` (denied)
- Attribute names are anonymized for confidentiality, but represent typical application fields (age, income, debt, employment, credit history, etc.)
- Missing values are marked with `?` in the raw data

## Approach

The full workflow lives in `notebook.ipynb`:

1. **Load & clean** — read the raw CSV (no header row), replace missing-value placeholders (`?`, `N/A`, `null`) with `NaN`.
2. **Impute missing values** — categorical columns filled with the most frequent value; numeric columns filled with the column mean.
3. **Encode features** — one-hot encode all categorical variables with `pd.get_dummies`, expanding to 389 features.
4. **Train/test split** — 67/33 split (`random_state=42`) on features and target.
5. **Scale** — standardize features with `StandardScaler`, fit on train and applied to test.
6. **Model** — `LogisticRegression`, tuned via `GridSearchCV` (5-fold CV) over `tol` and `max_iter`.
7. **Evaluate** — best CV score and held-out test accuracy, plus a confusion matrix on training predictions.

## Results

- Best cross-validated training score: **~81.4%** (`max_iter=100`, `tol=0.01`)
- Test set accuracy: **~79.4%**

## Files

| File | Description |
|---|---|
| `notebook.ipynb` | End-to-end analysis: preprocessing, modeling, tuning, evaluation |
| `cc_approvals.data` | Raw dataset (CSV, no header) |
| `credit_card.jpg` | Cover image used in the notebook |
| `README.md` | Project documentation |

## Tech Stack

- Python
- pandas, numpy — data loading and preprocessing
- scikit-learn — `StandardScaler`, `LogisticRegression`, `train_test_split`, `GridSearchCV`, `confusion_matrix`

## Getting Started

```bash
pip install pandas numpy scikit-learn jupyter
jupyter notebook notebook.ipynb
```

Run the cells in order — the notebook reads `cc_approvals.data` from the same directory.

## Possible Improvements

- Try alternative encodings (e.g. handle high-cardinality categoricals more carefully than blanket one-hot encoding)
- Compare against other classifiers (random forest, gradient boosting, SVM)
- Address class imbalance if present, and report precision/recall/F1 alongside accuracy
- Use cross-validated test evaluation rather than a single train/test split
