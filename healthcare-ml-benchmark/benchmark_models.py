"""
Benchmark six ML classifiers on diabetes readmission-style data.
Public / synthetic data only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUTPUT_DIR = Path(__file__).parent / "outputs"
RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    try:
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=296)
        X = ds.data.features.select_dtypes(include=[np.number]).copy()
        y_raw = ds.data.targets.squeeze()
        # readmitted within 30 days as binary positive class
        if y_raw.dtype == object:
            y = y_raw.astype(str).str.contains("<30", case=False, na=False).astype(int)
        else:
            y = (y_raw > 0).astype(int)
        X = X.dropna(axis=1, how="all").fillna(X.median(numeric_only=True))
        return X, y
    except Exception as exc:
        print(f"Dataset download failed ({exc}); using synthetic data")
        return _synthetic_data()


def _synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(RANDOM_STATE)
    n = 5000
    X = pd.DataFrame(
        {
            "time_in_hospital": rng.integers(1, 15, n),
            "num_medications": rng.integers(1, 40, n),
            "num_lab_procedures": rng.integers(1, 100, n),
            "num_procedures": rng.integers(0, 7, n),
            "number_emergency": rng.integers(0, 5, n),
            "number_inpatient": rng.integers(0, 5, n),
            "number_outpatient": rng.integers(0, 5, n),
            "age_midpoint": rng.integers(30, 90, n),
        }
    )
    logit = (
        -2.0
        + 0.12 * X["time_in_hospital"]
        + 0.04 * X["num_medications"]
        + 0.35 * X["number_emergency"]
        + 0.02 * (X["age_midpoint"] - 60)
    )
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)
    return X, pd.Series(y, name="readmit_30d")


def get_models() -> dict:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15)),
            ]
        ),
        "gaussian_nb": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", GaussianNB()),
            ]
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    X, y = load_data()

    # Keep runtime practical on large UCI extracts (full run optional later)
    max_samples = 12_000
    if len(y) > max_samples:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_samples, random_state=RANDOM_STATE, stratify=y
        )
        print(f"Subsampled to {max_samples} rows for benchmark runtime")

    print(f"Samples: {len(y)}, features: {X.shape[1]}, positive rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    rows = []
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for name, model in get_models().items():
        cv = cross_validate(
            model, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        row = {
            "model": name,
            "cv_accuracy_mean": cv["test_accuracy"].mean(),
            "cv_f1_mean": cv["test_f1"].mean(),
            "cv_roc_auc_mean": cv["test_roc_auc"].mean(),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_pred, zero_division=0),
            "test_f1": f1_score(y_test, y_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
        }
        rows.append(row)
        print(
            f"{name:22s}  test_f1={row['test_f1']:.3f}  cv_f1={row['cv_f1_mean']:.3f}"
        )

    results = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    out_path = OUTPUT_DIR / "benchmark_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults written to {out_path}")
    print("\nTop model by test F1:")
    print(results.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
