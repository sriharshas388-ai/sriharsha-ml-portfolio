# Pima Diabetes — Risk Benchmark + Explainability (XAI)

A small, fully reproducible study on the **Pima Indians Diabetes** dataset (real,
public): benchmark six classifiers for diabetes-onset risk and explain the
predictions. Everything below is produced by running `pipeline.py` (seed = 42,
pure NumPy/pandas — no scikit-learn required), so the numbers are reproducible on
any machine.

This is a methods-focused companion to my MSc dissertation, where I compared six
ML models on 1.49M de-identified NHS records and applied SHAP/LIME; here the same
workflow is shown end-to-end on a small open dataset.

## Data & preprocessing

768 patients, 8 features, 34.9% positive (diabetes onset within 5 years). A
well-known quirk of this dataset is that zeros in `glucose`, `blood_pressure`,
`skin_thickness`, `insulin` and `bmi` are physiologically impossible and actually
encode **missing values** — the pipeline sets these to `NaN` and median-imputes
before modelling. Features are standardised within each training fold to avoid
leakage.

## Results — discrimination & calibration (5-fold stratified CV)

| Model | AUROC (mean ± sd) | Accuracy | Brier |
|---|---|---|---|
| Linear Discriminant Analysis | 0.838 ± 0.036 | 0.754 | 0.164 |
| Logistic Regression | 0.837 ± 0.037 | 0.770 | 0.157 |
| Random Forest (30×d4) | 0.831 ± 0.024 | 0.755 | 0.159 |
| k-NN (k=15) | 0.830 ± 0.028 | 0.775 | 0.159 |
| Gaussian Naive Bayes | 0.817 ± 0.029 | 0.746 | 0.185 |
| Decision Tree (d=4) | 0.792 ± 0.016 | 0.745 | 0.174 |

The two simplest, fully interpretable models (LDA, logistic regression) top the
table on AUROC and give the best Brier (calibration) scores — a concrete instance
of Rudin's (2019) argument that interpretable models are often competitive for
high-stakes tabular prediction. AUROC ≈ 0.84 is consistent with published
benchmarks on this dataset.

## Results — explainability

**Global permutation importance** (drop in AUROC when a feature is shuffled;
logistic model, base AUROC 0.845):

| Feature | AUROC drop |
|---|---|
| glucose | 0.2051 |
| bmi | 0.0540 |
| pregnancies | 0.0252 |
| dpf (diabetes pedigree) | 0.0137 |
| age | 0.0034 |
| blood_pressure | 0.0014 |
| insulin | 0.0001 |
| skin_thickness | 0.0001 |

Glucose dominates, with BMI a distant second — clinically sensible and stable
across runs.

**Local LIME-style explanation** for the highest-risk patient (predicted risk
0.99) recovers glucose and BMI as the main local drivers, agreeing with the
global ranking.

## Run it

```bash
python3 pipeline.py
```

Only needs NumPy + pandas (`pip install -r requirements.txt`). On first run it
downloads the dataset and caches it to `pima.csv`; the cached copy is committed so
it also runs fully offline. To reproduce with the published SHAP/LIME libraries,
swap the logistic model for any classifier and call `shap.Explainer` /
`lime.lime_tabular` — the methods cited below.

## References (APA)

- Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S.
  (1988). Using the ADAP learning algorithm to forecast the onset of diabetes
  mellitus. *Proceedings of the Annual Symposium on Computer Application in
  Medical Care*, 261–265. (Pima Indians Diabetes dataset.)
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model
  predictions. *NeurIPS, 30*, 4765–4774. https://doi.org/10.48550/arXiv.1705.07874
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?":
  Explaining the predictions of any classifier. *KDD '16*, 1135–1144.
  https://doi.org/10.1145/2939672.2939778
- Rudin, C. (2019). Stop explaining black box ML models for high-stakes decisions
  and use interpretable models instead. *Nature Machine Intelligence, 1*, 206–215.
  https://doi.org/10.1038/s42256-019-0048-x

## Note on data

The Pima Indians Diabetes dataset is publicly available and widely used for ML
research and teaching. No private patient records are used.
