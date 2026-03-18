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


def drift_check(
    train: pd.DataFrame, test: pd.DataFrame, alpha: float = 0.05
) -> list[dict[str, Any]]:
    """KS test for numeric columns between train and test."""
    results = []
    shared = set(train.columns) & set(test.columns)
    for col in shared:
        if not pd.api.types.is_numeric_dtype(train[col]):
            continue
        tr = train[col].dropna()
        te = test[col].dropna()
        if len(tr) < 20 or len(te) < 20:
            continue
        stat, pval = stats.ks_2samp(tr, te)
        if pval < alpha:
            results.append(
                {
                    "column": col,
                    "ks_statistic": round(float(stat), 4),
                    "p_value": round(float(pval), 6),
                    "drift_detected": True,
                }
            )
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
