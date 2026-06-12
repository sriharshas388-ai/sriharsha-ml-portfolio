"""
Audience-centred narrative explanations for healthcare risk predictions.

Maps SHAP-style feature contributions into plain-language summaries for
clinical and operational stakeholders. Uses public UCI heart disease data only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).parent / "outputs"
RANDOM_STATE = 42

FEATURE_LABELS = {
    "age": "patient age",
    "sex": "sex",
    "cp": "chest pain type",
    "trestbps": "resting blood pressure",
    "chol": "cholesterol",
    "fbs": "fasting blood sugar",
    "restecg": "resting ECG result",
    "thalach": "maximum heart rate achieved",
    "exang": "exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "ST segment slope",
    "ca": "number of major vessels",
    "thal": "thalassemia test result",
}


@dataclass
class AudienceProfile:
    name: str
    detail_level: str  # "clinical" | "operational"


AUDIENCES = {
    "clinical": AudienceProfile("Clinical reviewer", "clinical"),
    "coordinator": AudienceProfile("Care coordinator", "operational"),
}


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(RANDOM_STATE)
    n = 400
    X = pd.DataFrame(
        {
            "age": rng.integers(29, 80, n),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(0, 4, n),
            "trestbps": rng.integers(94, 200, n),
            "chol": rng.integers(126, 564, n),
            "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n),
            "thalach": rng.integers(71, 202, n),
            "exang": rng.integers(0, 2, n),
            "oldpeak": rng.uniform(0, 6.2, n).round(1),
            "slope": rng.integers(0, 3, n),
            "ca": rng.integers(0, 4, n),
            "thal": rng.integers(1, 4, n),
        }
    )
    risk = (
        0.02 * (X["age"] - 50)
        + 0.4 * X["cp"]
        + 0.3 * X["exang"]
        + 0.15 * (X["chol"] > 240).astype(int)
        + 0.1 * (X["thalach"] < 120).astype(int)
    )
    y = (risk + rng.normal(0, 0.3, n) > 0.8).astype(int)
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[RandomForestClassifier, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE)
    model.fit(X_scaled, y)
    return model, scaler


def shap_for_instance(
    model: RandomForestClassifier, scaler: StandardScaler, X: pd.DataFrame, idx: int
) -> pd.Series:
    explainer = shap.TreeExplainer(model)
    row = scaler.transform(X.iloc[[idx]])
    values = explainer.shap_values(row)
    if isinstance(values, list):
        values = values[1]
    return pd.Series(values[0], index=X.columns).sort_values(key=abs, ascending=False)


def narrative_for_audience(
    prediction: int,
    probability: float,
    contributions: pd.Series,
    audience: AudienceProfile,
    top_n: int = 3,
) -> str:
    label = "elevated cardiovascular risk" if prediction == 1 else "lower cardiovascular risk"
    prob_pct = round(probability * 100, 1)
    drivers = contributions.head(top_n)

    if audience.detail_level == "clinical":
        driver_text = "; ".join(
            f"{FEATURE_LABELS.get(f, f)} {'increased' if v > 0 else 'decreased'} risk "
            f"(contribution {v:+.3f})"
            for f, v in drivers.items()
        )
        return (
            f"Model assessment: {label} ({prob_pct}% estimated probability). "
            f"Primary contributing factors: {driver_text}. "
            "Review alongside clinical history; explanation reflects model logic, not a diagnosis."
        )

    plain = ", ".join(FEATURE_LABELS.get(f, f) for f in drivers.index)
    action = (
        "Consider prioritising follow-up and risk-factor review."
        if prediction == 1
        else "Routine monitoring may be appropriate based on local pathway."
    )
    return (
        f"The model suggests {label} ({prob_pct}% chance). "
        f"The main factors in this case relate to {plain}. {action}"
    )


def run() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model, scaler = train_model(X_train, y_train)

    # Pick a high-risk example from test set
    proba = model.predict_proba(scaler.transform(X_test))[:, 1]
    idx = int(np.argmax(proba))
    row = X_test.iloc[[idx]]
    pred = int(model.predict(scaler.transform(row))[0])
    p = float(proba[idx])
    contribs = shap_for_instance(model, scaler, X_test, idx)

    lines = ["Audience-centred explanation examples", "=" * 40, ""]
    for key, profile in AUDIENCES.items():
        lines.append(f"[{profile.name}]")
        lines.append(narrative_for_audience(pred, p, contribs, profile))
        lines.append("")

    out = OUTPUT_DIR / "audience_narratives.txt"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    run()
