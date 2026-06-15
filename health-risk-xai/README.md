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

## References (APA)

- Detrano, R., Janosi, A., Steinbrunn, W., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *The American Journal of Cardiology, 64*(5), 304–310. https://doi.org/10.1016/0002-9149(89)90524-9
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS, 30*, 4765–4774. https://doi.org/10.48550/arXiv.1705.07874
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD '16*, 1135–1144. https://doi.org/10.1145/2939672.2939778
- Rudin, C. (2019). Stop explaining black box ML models for high-stakes decisions and use interpretable models instead. *Nature Machine Intelligence, 1*, 206–215. https://doi.org/10.1038/s42256-019-0048-x

*SHAP and LIME are the explanation methods implemented here; the UCI Heart Disease (Cleveland) data originates from Detrano et al. (1989). Rudin (2019) motivates comparing against interpretable baselines.*
