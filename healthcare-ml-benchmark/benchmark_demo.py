"""Reproducible 6-model benchmark on the real UCI Heart Disease dataset
(303 patients). Demonstrates the comparison harness end-to-end with no sklearn
dependency. Seed = 42.  (The Diabetes-130 path in benchmark_models.py uses
scikit-learn + ucimlrepo and is run locally.)"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlkit import load_heart, cv_benchmark

FIG = os.path.join(os.path.dirname(__file__), "figures"); os.makedirs(FIG, exist_ok=True)
BLUE, GREY = "#1f4e79", "#9bb7d4"


def main():
    X, y, cols = load_heart()
    rows = cv_benchmark(X, y)
    print(f"UCI Heart Disease: n={len(X)}, positives={y.mean():.1%}\n")
    print("| Model | AUROC±sd | Accuracy | Brier |\n|---|---|---|---|")
    for m, a, s, ac, b in rows:
        print(f"| {m} | {a:.3f}±{s:.3f} | {ac:.3f} | {b:.3f} |")

    names = [r[0] for r in rows][::-1]; means = [r[1] for r in rows][::-1]; sds = [r[2] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(names, means, xerr=sds, color=BLUE, ecolor=GREY, capsize=4)
    ax.set_xlim(0.82, 0.93); ax.set_xlabel("AUROC (5-fold stratified CV, mean ± sd)")
    ax.set_title("Six-model benchmark — UCI Heart Disease")
    for i, m in enumerate(means): ax.text(m+0.002, i, f"{m:.3f}", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/benchmark_auroc.png", dpi=150); plt.close(fig)
    print("\nfigure ->", os.listdir(FIG))


if __name__ == "__main__":
    main()
