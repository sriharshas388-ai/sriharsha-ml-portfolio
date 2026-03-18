# health-data-quality-kit

Small Python utility for tabular data quality checks — the kind I used to run in industry before loading analytics tables.

Useful when working with healthcare-style datasets: missing values, duplicate rows, unexpected categories, basic distribution drift between train and test splits.

## Features

- Column-level null rate report  
- Duplicate row detection  
- Numeric range / outlier flags (IQR method)  
- Categorical cardinality warnings  
- Simple train vs test drift check (KS test for numeric columns)

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

Data quality issues directly affect ML fairness and explanation trust. I want to connect DQ checks with explainable AI in my PhD work.
