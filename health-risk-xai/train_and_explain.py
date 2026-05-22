"""
Train a heart disease classifier and produce SHAP + LIME explanations.
Uses UCI Heart Disease (Cleveland) — public dataset only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).parent / "outputs"
RANDOM_STATE = 42

FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def load_heart_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI heart disease data with fallback to bundled sample."""
    try:
        from ucimlrepo import fetch_ucirepo

        heart = fetch_ucirepo(id=45)
        X = heart.data.features
        y = heart.data.targets.squeeze()
        # Binary target: 0 = no disease, 1 = disease present
        y = (y > 0).astype(int)
        if X.shape[1] == len(FEATURE_NAMES):
            X.columns = FEATURE_NAMES
        return X, y
    except Exception as exc:
        print(f"ucimlrepo fetch failed ({exc}), using synthetic fallback")
        return _synthetic_fallback()


def _synthetic_fallback() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(RANDOM_STATE)
    n = 300
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
            "oldpeak": rng.round(rng.uniform(0, 6.2, n), 1),
            "slope": rng.integers(0, 3, n),
            "ca": rng.integers(0, 4, n),
            "thal": rng.integers(0, 4, n),
        }
    )
    risk_score = (
        0.03 * (X["age"] - 50)
        + 0.4 * X["cp"]
        + 0.002 * (X["chol"] - 200)
        + 0.3 * X["exang"]
        + 0.25 * X["oldpeak"]
    )
    y = (risk_score + rng.normal(0, 0.5, n) > 0.8).astype(int)
    return X, pd.Series(y, name="target")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    X, y = load_heart_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_train_s, y_train)
    accuracy = model.score(X_test_s, y_test)
    print(f"Test accuracy: {accuracy:.3f}")

    # --- SHAP ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_s)
    # For binary RF, shap returns list per class; use positive class
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_test, feature_names=list(X.columns), show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    shap.summary_plot(sv, X_test, plot_type="bar", feature_names=list(X.columns), show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved SHAP plots to outputs/")

    # --- LIME ---
    lime_explainer = LimeTabularExplainer(
        X_train_s,
        feature_names=list(X.columns),
        class_names=["no_disease", "disease"],
        mode="classification",
        discretize_continuous=False,
        random_state=RANDOM_STATE,
    )
    idx = 0
    exp = lime_explainer.explain_instance(
        X_test_s[idx], model.predict_proba, num_features=8
    )
    exp.save_to_file(str(OUTPUT_DIR / "lime_example.html"))
    print("Saved LIME explanation to outputs/lime_example.html")


if __name__ == "__main__":
    main()
