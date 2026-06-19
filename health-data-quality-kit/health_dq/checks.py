"""Core data quality check functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def null_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Return null counts and percentages per column."""
    nulls = df.isnull().sum()
    pct = (nulls / len(df) * 100).round(2)
    return (
        pd.DataFrame({"null_count": nulls, "null_pct": pct})
        .sort_values("null_pct", ascending=False)
        .reset_index()
        .rename(columns={"index": "column"})
    )


def duplicate_summary(df: pd.DataFrame) -> dict[str, int]:
    dup_count = int(df.duplicated().sum())
    return {
        "total_rows": len(df),
        "duplicate_rows": dup_count,
        "duplicate_pct": round(dup_count / max(len(df), 1) * 100, 2),
    }


def numeric_outlier_flags(df: pd.DataFrame, iqr_multiplier: float = 1.5) -> dict[str, int]:
    """Flag rows with outliers in any numeric column (IQR rule)."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return {"outlier_rows": 0}

    mask = pd.Series(False, index=df.index)
    for col in numeric.columns:
        q1, q3 = numeric[col].quantile(0.25), numeric[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        mask |= (numeric[col] < lower) | (numeric[col] > upper)

    return {"outlier_rows": int(mask.sum())}


def categorical_warnings(df: pd.DataFrame, max_cardinality: int = 50) -> list[dict[str, Any]]:
    warnings = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        nunique = df[col].nunique(dropna=True)
        if nunique > max_cardinality:
            warnings.append(
                {
                    "column": col,
                    "unique_values": int(nunique),
                    "message": f"High cardinality ({nunique} unique values)",
                }
            )
    return warnings


def population_stability_index(
    expected: pd.Series, actual: pd.Series, bins: int = 10, eps: float = 1e-6
) -> float:
    """Population Stability Index (PSI) between a reference and a current sample.

    PSI = sum( (a_i - e_i) * ln(a_i / e_i) ) over bins, where e_i / a_i are the
    proportion of the expected / actual sample falling in bin i. It quantifies the
    *magnitude* of a distribution shift, complementing the KS test's binary verdict
    (Webb et al., 2016). Numeric columns are binned on quantiles of the reference;
    categorical columns use category frequencies. A small epsilon avoids ln(0).

    Conventional bands: <0.10 insignificant, 0.10-0.25 moderate, >0.25 significant.
    """
    expected = expected.dropna()
    actual = actual.dropna()
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    if pd.api.types.is_numeric_dtype(expected):
        # Quantile edges from the reference; dedupe to handle low-variance columns.
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:  # not enough distinct values to bin meaningfully
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        e_counts = np.histogram(expected, bins=edges)[0].astype(float)
        a_counts = np.histogram(actual, bins=edges)[0].astype(float)
    else:
        categories = pd.Index(expected.unique()).union(actual.unique())
        e_counts = expected.value_counts().reindex(categories, fill_value=0).to_numpy(float)
        a_counts = actual.value_counts().reindex(categories, fill_value=0).to_numpy(float)

    e_prop = e_counts / e_counts.sum()
    a_prop = a_counts / a_counts.sum()
    e_prop = np.clip(e_prop, eps, None)
    a_prop = np.clip(a_prop, eps, None)
    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))


def _psi_band(psi: float) -> str:
    if psi < 0.10:
        return "insignificant"
    if psi < 0.25:
        return "moderate"
    return "significant"


def drift_check(
    train: pd.DataFrame,
    test: pd.DataFrame,
    alpha: float = 0.05,
    psi_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    """Detect distribution drift between a reference (train) and current (test) frame.

    Numeric columns get a KS two-sample test (significance) *and* a PSI value
    (magnitude). Categorical columns -- previously ignored -- are now covered by PSI
    on category frequencies. A column is flagged if KS is significant (p < alpha)
    or PSI exceeds ``psi_threshold``.
    """
    results = []
    shared = set(train.columns) & set(test.columns)
    for col in sorted(shared):
        tr = train[col].dropna()
        te = test[col].dropna()
        if len(tr) < 20 or len(te) < 20:
            continue

        numeric = pd.api.types.is_numeric_dtype(train[col])
        psi = round(population_stability_index(tr, te), 4)
        entry: dict[str, Any] = {
            "column": col,
            "type": "numeric" if numeric else "categorical",
            "psi": psi,
            "psi_band": _psi_band(psi),
        }

        flagged = psi >= psi_threshold
        if numeric:
            stat, pval = stats.ks_2samp(tr, te)
            entry["ks_statistic"] = round(float(stat), 4)
            entry["p_value"] = round(float(pval), 6)
            flagged = flagged or (pval < alpha)

        if flagged:
            entry["drift_detected"] = True
            results.append(entry)
    return results


def run_quality_report(
    df: pd.DataFrame,
    reference: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run all checks and return a structured report dict."""
    report: dict[str, Any] = {
        "summary": {
            "rows": len(df),
            "columns": df.shape[1],
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        },
        "nulls": null_profile(df).to_dict(orient="records"),
        "duplicates": duplicate_summary(df),
        "outliers": numeric_outlier_flags(df),
        "categorical_warnings": categorical_warnings(df),
    }
    if reference is not None:
        report["drift_vs_reference"] = drift_check(reference, df)
    return report
