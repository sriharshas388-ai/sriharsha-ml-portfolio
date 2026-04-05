# healthcare-ml-benchmark

Compare six supervised learning models on a diabetes hospital readmission task.

Inspired by my MSc dissertation approach — run multiple models on the same preprocessed dataset, compare metrics fairly, export a results table for analysis.

## Models compared

1. Logistic Regression  
2. Random Forest  
3. Gradient Boosting  
4. Support Vector Machine (RBF)  
5. k-Nearest Neighbours  
6. Gaussian Naive Bayes  

## Metrics

- Accuracy, Precision, Recall, F1, ROC-AUC  
- 5-fold stratified cross-validation  
- Results saved to `outputs/benchmark_results.csv`

## Run

```bash
pip install -r requirements.txt
python benchmark_models.py
```

## Data

Uses the **Diabetes 130-US hospitals** dataset via `ucimlrepo` (UCI id 296), or a reproducible synthetic fallback if download fails.

## Why this exists

Useful for understanding which algorithms work best on imbalanced clinical tabular data before investing in explainability layers. Public data only.
