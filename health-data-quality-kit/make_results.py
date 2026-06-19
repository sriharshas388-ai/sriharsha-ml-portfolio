"""Reproducible results + figure for health-data-quality-kit.

Runs the quality report on the real UCI Heart Disease dataset and demonstrates
drift detection: we split the cohort into a 'reference' (younger patients) and a
'current' batch (older patients), which induces a genuine distribution shift. The
kit's drift check reports both the KS test (significance) and the Population
Stability Index (PSI, magnitude) per feature. Seed = 42.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from health_dq.checks import run_quality_report, population_stability_index

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
        ks_txt = f"KS={d['ks_statistic']}, p={d['p_value']}, " if "ks_statistic" in d else ""
        print(f"  DRIFT  {c}: {ks_txt}PSI={d['psi']} ({d['psi_band']})")

    # KS statistic + PSI for every numeric column (for the figure)
    cols = [c for c in df.columns if c != "target"]
    ks, psi = {}, {}
    for c in cols:
        stat, p = stats.ks_2samp(ref[c].dropna(), cur[c].dropna())
        ks[c] = (float(stat), float(p))
        psi[c] = population_stability_index(ref[c], cur[c])
    order = sorted(cols, key=lambda c: ks[c][0])

    fig, (axk, axp) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    axk.barh(order, [ks[c][0] for c in order],
             color=[RED if ks[c][1] < 0.05 else BLUE for c in order])
    axk.set_xlabel("Kolmogorov–Smirnov statistic")
    axk.set_title("Significance: KS (red = p < 0.05)")

    # age is the split variable, so its PSI is off-scale; cap the axis for
    # readability and annotate the true value.
    cap = 1.0
    axp.barh(order, [min(psi[c], cap) for c in order],
             color=[RED if psi[c] >= 0.25 else BLUE for c in order])
    for thr, ls in ((0.10, ":"), (0.25, "--")):
        axp.axvline(thr, color="grey", linestyle=ls, linewidth=1)
    for c in order:
        if psi[c] > cap:
            axp.text(cap, order.index(c), f" {psi[c]:.1f}→", va="center", fontsize=8)
    axp.set_xlim(0, cap * 1.05)
    axp.set_xlabel("Population Stability Index (PSI, capped at 1.0)")
    axp.set_title("Magnitude: PSI (--- 0.25 significant)")

    fig.suptitle("Data drift by feature — reference (younger) vs current (older)")
    fig.tight_layout(); fig.savefig(f"{FIG}/drift_by_feature.png", dpi=150); plt.close(fig)
    print("\nfigure ->", os.listdir(FIG))


if __name__ == "__main__":
    main()
