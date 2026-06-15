# Predictive Analytics for Health Risk Stratification (MSc Dissertation)

**ML applications in non-hospital COVID-19 treatment eligibility**

MSc Data Science dissertation, Teesside University (2025). Distinction.
Author: Sriharsha Surannagari (W9607526). Supervisor: Dr Mansha Nawaz.

This repository holds the cleaned code, the full written report and the Power BI
dashboard from my dissertation. The full thesis (with all figures and tables) is
in [`report/Dissertation_HealthRisk_ML_Report.pdf`](report/Dissertation_HealthRisk_ML_Report.pdf);
the dashboard is in [`report/HealthRisk_PowerBI_Dashboard.pdf`](report/HealthRisk_PowerBI_Dashboard.pdf).

## Abstract

This dissertation investigates machine learning for identifying high-risk
individuals eligible for non-hospital COVID-19 treatment. It uses publicly
available NHS data describing demographic and clinical information for a
population of 1.49 million digitally identified high-risk individuals, grouped by
clinical condition. After cleaning, imputation, feature engineering and class
balancing, six classifiers and a set of regressors are trained and compared to
predict treatment-eligibility risk, with the aim of supporting timely
intervention and more efficient resource allocation.

## Data

NHS-published counts of high-risk patients broken down by clinical condition
group and by demographic band (sex, age band, ethnicity). The headline
population is 1.49M individuals; the modelling table is the aggregated
condition-group breakdown. A binary **High_Risk** target is derived as
above-median total count per group, and SMOTE is used to balance the classes.

## Methodology

1. **Preprocessing** — column renaming, missing-value checks, summary statistics, correlation analysis.
2. **Feature engineering** — derive the High_Risk target; select ethnicity-based features, drop raw demographic-band columns to avoid target leakage.
3. **Class balancing** — SMOTE (the published counts are small and imbalanced).
4. **Modelling** — six classifiers: Logistic Regression, Random Forest, Gradient Boosting, Decision Tree, AdaBoost, SVM; plus a regression view of the count target.
5. **Tuning** — GridSearchCV over Random Forest and Gradient Boosting.
6. **Evaluation** — classification: Accuracy, Precision, Recall, F1, AUC-ROC and confusion matrices; regression: MSE, RMSE, R². Train-vs-test comparison to check for under/over-fitting.

## Headline results (as reported in the dissertation)

- **Best classifier — AdaBoost:** Accuracy 1.00, Precision 1.00, Recall 1.00, F1 1.00 on the balanced test set; AdaBoost outperformed all other classifiers, with Logistic Regression the weakest.
- **Best regressor — Gradient Boosting:** R² = 0.2454, MSE = 0.1657, RMSE = 0.4071.
- A generalised model was selected to balance training and test performance and avoid over/under-fitting.

Full per-model tables, metric bar charts, confusion matrices and train-vs-test
comparisons are in the report PDF.

> **Honest note on scope.** The 1.49M figure is the represented population; the
> aggregated modelling table is small, so the perfect classification scores
> reflect a small, balanced sample rather than a large held-out cohort. The value
> of the project is the end-to-end pipeline (preprocessing, balancing, six-model
> comparison, tuning and evaluation) and the Power BI reporting layer, not the
> absolute accuracy number.

## Run it

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
python src/health_risk_pipeline.py     # expects DataSet.csv in the working dir
```

`src/SourceCode_original_notebook.pdf` is the original exported Jupyter notebook.

## Contents

- `src/health_risk_pipeline.py` — cleaned, runnable pipeline (six classifiers + regression).
- `src/SourceCode_original_notebook.pdf` — original notebook export.
- `report/Dissertation_HealthRisk_ML_Report.pdf` — full thesis.
- `report/HealthRisk_PowerBI_Dashboard.pdf` — Power BI dashboard.

## Tools

Python, pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), matplotlib, seaborn, Power BI.
