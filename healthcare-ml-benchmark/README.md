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

## References (APA)

- Strack, B., DeShazo, J. P., Gennings, C., et al. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International, 2014*, 781670. https://doi.org/10.1155/2014/781670
- Caruana, R., Lou, Y., Gehrke, J., et al. (2015). Intelligible models for healthcare. *KDD '15*, 1721–1730. https://doi.org/10.1145/2783258.2788613
- Rudin, C. (2019). Stop explaining black box ML models for high-stakes decisions and use interpretable models instead. *Nature Machine Intelligence, 1*, 206–215. https://doi.org/10.1038/s42256-019-0048-x

*The Diabetes 130-US hospitals dataset is described by Strack et al. (2014). Caruana et al. (2015) and Rudin (2019) frame why interpretable baselines belong in a clinical benchmark.*
