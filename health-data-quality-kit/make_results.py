"""Reproducible results + figure for health-data-quality-kit.

Runs the quality report on the real UCI Heart Disease dataset and demonstrates
drift detection: we split the cohort into a 'reference' (younger patients) and a
'current' batch (older patients), which induces a genuine distribution shift, and
the kit's KS-based drift check flags the affected features. Seed = 42.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from health_dq.checks import run_quality_report

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
BLUE, RED = "#1f4e79", "#c0392b"


def main():
    df = pd.read_csv(os.path.join(HERE, "heart.csv"))
    rep = run_quality_report(df)
    print(f"Rows: {rep['summary']['rows']}, cols: {rep['summary']['columns']}, "
          f"duplicates: {rep['duplicates']['duplicate_rows']}, "
          f"outlier rows (IQR): {rep['outliers']['outlier_rows']}")

    # induce a real shift: reference = younger half, current = older half
    med = df["age"].median()
    ref, cur = df[df["age"] <= med], df[df["age"] > med]
    report = run_quality_report(cur, reference=ref)
    drifted = {d["column"]: d for d in report.get("drift_vs_reference", [])}
    print(f"\nDrift check (reference n={len(ref)} younger vs current n={len(cur)} older):")
    for c, d in drifted.items():
        print(f"  DRIFT  {c}: KS={d['ks_statistic']}, p={d['p_value']}")

    # KS statistic for every numeric column (for the figure)
    cols = [c for c in df.columns if c != "target"]
    ks = {}
    for c in cols:
        stat, p = stats.ks_2samp(ref[c].dropna(), cur[c].dropna())
        ks[c] = (float(stat), float(p))
    order = sorted(cols, key=lambda c: ks[c][0])
    vals = [ks[c][0] for c in order]
    colors = [RED if ks[c][1] < 0.05 else BLUE for c in order]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(order, vals, color=colors)
    ax.set_xlabel("Kolmogorov–Smirnov statistic (reference vs current)")
    ax.set_title("Data drift by feature (red = flagged, p < 0.05)")
    fig.tight_layout(); fig.savefig(f"{FIG}/drift_by_feature.png", dpi=150); plt.close(fig)
    print("\nfigure ->", os.listdir(FIG))


if __name__ == "__main__":
    main()
