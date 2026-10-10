# Breast Cancer Classification: Machine Learning Model Development and Evaluation

Yuva Intern, Week 4 task. A complete, reproducible machine learning workflow in Python that classifies breast tumours as **malignant** or **benign**, from data preparation through model selection, evaluation and error analysis.

> **Disclaimer:** This is a learning project on a public benchmark dataset. The model is not a medical device and must not be used for real diagnosis.

## Project Overview

- **Problem:** binary classification of tumours using measurements computed from digitised images of a fine-needle aspirate.
- **Positive class:** malignant. A missed cancer (false negative) is the costlier error, so recall and F1 are prioritised alongside accuracy and ROC AUC.
- **Dataset:** Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository, loaded through `sklearn.datasets.load_breast_cancer`. It has 569 samples (357 benign, 212 malignant) and 30 numeric features, with no missing values or duplicates.

## Approach

1. **Data preparation:** data checks, target encoding (malignant = 1), and a stratified 80/20 train/test split (random seed 42).
2. **Leakage-safe pipeline:** `StandardScaler` sits inside a scikit-learn `Pipeline`, so scaling is learned from training folds only.
3. **Model comparison:** logistic regression, K-nearest neighbours, decision tree and random forest, compared with stratified 5-fold cross-validation.
4. **Tuning:** `GridSearchCV` (5-fold, optimising F1) on training data only.
5. **Final evaluation:** a single evaluation of each tuned model on the held-out test set.
6. **Diagnostics:** under/overfitting analysis, learning curve, feature importance, error analysis, threshold analysis and a stability check across 10 random splits.

## Results (test set, 114 samples)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| **Logistic Regression** (selected) | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 |
| K-Nearest Neighbours | 0.939 | 0.973 | 0.857 | 0.911 | 0.982 |
| Decision Tree | 0.930 | 0.925 | 0.881 | 0.902 | 0.972 |
| Random Forest | 0.965 | 1.000 | 0.905 | 0.950 | 0.994 |

Logistic regression was selected on cross-validated F1 before the test set was used. It made 4 errors out of 114 (1 false positive, 3 false negatives). Across 10 different random splits, its accuracy averaged about 97% with a standard deviation of about 1.5 points, so single test scores should be read with that noise in mind.

## Visualisations

Running the script generates nine figures: class distribution, correlation matrix, metric comparison, confusion matrices, ROC curves, decision-tree depth curve (underfitting vs overfitting), learning curve, feature importance and a precision-recall curve.

## Repository Structure

```
.
├── ml_pipeline.py                              # full pipeline: data, models, evaluation, figures
├── breast_cancer_dataset.csv                   # dataset exported by the script
├── Week4_ML_Model_Development_Evaluation.docx  # full written report
└── README.md
```

Running the script also creates the figure files (`m1_eda.png` to `m9_pr.png`) and `ml_results.json` in the working directory.

## How to Run

Requires Python 3.9 or later.

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install pandas numpy scikit-learn matplotlib

# 3. run the pipeline
python ml_pipeline.py
```

All random seeds are fixed (`42`), so the results should be reproducible, though small differences can appear across scikit-learn versions.

## Key Findings

- Simple, scaled logistic regression matched or beat more complex models on this small, well-separated dataset.
- The unrestricted decision tree overfits: it reaches 100% training accuracy while cross-validated accuracy plateaus at about 92%.
- Most errors are borderline cases near the decision boundary. Lowering the decision threshold would trade some precision for higher recall; this was explored on the test set for illustration only, and a real project would choose the threshold on validation data.

## Limitations and Future Work

- Small, single-source dataset and no external validation.
- A 114-sample test set gives noisy estimates.
- Highly collinear features make individual coefficients hard to interpret.
- Possible improvements: nested cross-validation, probability calibration, cost-sensitive learning, SVM or gradient boosting, SHAP explanations, and testing on external data.

## Author

Lokesh, Yuva Intern programme.
