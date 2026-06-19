# health-data-quality-kit

Small Python utility for tabular data quality checks — the kind I used to run in industry before loading analytics tables.

Useful when working with healthcare-style datasets: missing values, duplicate rows, unexpected categories, basic distribution drift between train and test splits.

## Features

- Column-level null rate report  
- Duplicate row detection  
- Numeric range / outlier flags (IQR method)  
- Categorical cardinality warnings  
- Train vs test drift check reporting both **significance** (KS test, numeric) and **magnitude** (Population Stability Index, numeric *and* categorical)

## Usage

```bash
pip install -r requirements.txt
python -m health_dq.cli --input sample_data.csv --report outputs/dq_report.json
```

Or in Python:

```python
from health_dq.checks import run_quality_report
import pandas as pd

df = pd.read_csv("your_file.csv")
report = run_quality_report(df)
print(report["summary"])
```

## Sample data

`sample_data/patients_sample.csv` is fully synthetic — safe for demos and GitHub.

## Context

Data quality issues directly affect ML fairness and explanation trust. Connecting rigorous data-quality checks with explainable AI is a research thread I keep returning to.

## References (APA)

- Schelter, S., Lange, D., Schmidt, P., et al. (2018). Automating large-scale data quality verification. *Proceedings of the VLDB Endowment, 11*(12), 1781–1794. https://doi.org/10.14778/3229863.3229867
- Polyzotis, N., Roy, S., Whang, S. E., & Zinkevich, M. (2017). Data management challenges in production machine learning. *ACM SIGMOD*, 1723–1726. https://doi.org/10.1145/3035918.3054782
- Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit. *Journal of the American Statistical Association, 46*(253), 68–78. https://doi.org/10.1080/01621459.1951.10500769
- Webb, G. I., Hyde, R., Cao, H., Nguyen, H.-L., & Petitjean, F. (2016). Characterizing concept drift. *Data Mining and Knowledge Discovery, 30*(4), 964–994. https://doi.org/10.1007/s10618-015-0448-4
- Gardner, J., Popović, Z., & Schmidt, L. (2023). Benchmarking distribution shift in tabular data with TableShift. *Advances in Neural Information Processing Systems (NeurIPS), 36.* https://doi.org/10.48550/arXiv.2312.07577
- Silva, G. F. D. S., Barcellos Filho, F. N., Wichmann, R. M., da Silva Junior, F. C., & Chiavegatto Filho, A. D. P. (2025). Strategies for detecting and mitigating dataset shift in machine learning for health predictions: A systematic review. *Journal of Biomedical Informatics, 170,* 104902. https://doi.org/10.1016/j.jbi.2025.104902

*The constraint-as-unit-test design follows Deequ (Schelter et al., 2018); the train/test drift check pairs the Kolmogorov–Smirnov test (Massey, 1951) for significance with the Population Stability Index for drift magnitude — quantifying how much a distribution moved, not just whether it moved, in the spirit of Webb et al. (2016). Statistical tests remain the most common shift-detection strategy in clinical tabular ML (Silva et al., 2025; Gardner et al., 2023).*

## Results (reproducible — run `python make_results.py`)

Quality report on the real **UCI Heart Disease** dataset (303 rows, 14 columns):
**1 duplicate row** and **75 rows** with IQR outliers were flagged. To exercise the
drift detector, the cohort is split into a younger 'reference' (n=152) and an older
'current' batch (n=151), which induces a genuine shift. The check now reports two
complementary signals per feature — the KS test for *significance* and the
Population Stability Index (PSI) for *magnitude*.

![Drift by feature](figures/drift_by_feature.png)

The two panels show why magnitude matters. KS flags age, ca, thalach, trestbps,
oldpeak, cp and slope as significant (p < 0.05), but PSI separates the genuinely
large shifts (thalach PSI=0.66, ca=0.42, trestbps=0.36, oldpeak=0.28 — all "significant",
≥0.25) from changes that are statistically detectable yet small in size: **slope**
is KS-significant (p=0.005) but its PSI is just **0.006** ("insignificant"), and
**cp** sits in the "moderate" band (PSI=0.11). `age` — the split variable — is off-scale
(PSI≈11) as expected. In monitoring terms, a tiny p-value alone can over-trigger
retraining; PSI tells you whether the move is large enough to act on. PSI also covers
categorical features, which the KS-only check skipped entirely.
