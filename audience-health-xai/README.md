# Audience Health XAI

Plain-language explanations for healthcare risk predictions — aligned with **audience-centred explainable AI** research (Newcastle COMP2179 theme).

## What it does

- Trains a random forest on structured cardiovascular-style tabular data (synthetic/public fallback)
- Computes SHAP contributions for an individual prediction
- Generates **two narrative formats**:
  - **Clinical** — feature-level contributions with cautious clinical framing
  - **Operational** — simplified language for care coordinators

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python narrative_explainer.py
```

Output: `outputs/audience_narratives.txt`

## Data

Public/synthetic data only. No real patient records.
