# health-risk-xai

Explainable machine learning for cardiovascular risk prediction.

Built while preparing PhD applications in explainable AI for healthcare. The idea is simple: a model might predict "high risk" but a clinician or analyst needs to know **why**.

## What it does

- Trains a Random Forest on the UCI Heart Disease dataset (Cleveland subset)
- Generates **SHAP** summary and force plots
- Runs **LIME** for individual prediction explanations
- Saves plots to `outputs/` for reports and presentations

## Quick start

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_and_explain.py
```

## Dataset

[UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) — publicly available, widely used for ML teaching and research. Loaded automatically via `ucimlrepo` with a local CSV fallback.

## Sample output

After running, check `outputs/`:

- `shap_summary.png` — global feature importance
- `shap_bar.png` — mean absolute SHAP values
- `lime_example.html` — single-patient explanation

## Notes

This is a learning/portfolio project. Explanation UI for non-CS audiences (dashboards, plain-language summaries) is the direction I want to take in PhD research.

## License

MIT
