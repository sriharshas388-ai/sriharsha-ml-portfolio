"""
Generate result figures for the Pima diabetes study (saved to figures/).
All values come from the same reproducible pipeline (seed = 42).
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import (load, cv_benchmark, logreg, permutation_importance,
                      auroc, COLS, stratified_folds, MODELS)

FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)
BLUE, GREY, RED = "#1f4e79", "#9bb7d4", "#c0392b"


def fig_auroc(X, y):
    rows = cv_benchmark(X, y)
    names = [r[0] for r in rows][::-1]
    means = [r[1] for r in rows][::-1]
    sds = [r[2] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(names, means, xerr=sds, color=BLUE, ecolor=GREY, capsize=4)
    ax.set_xlim(0.7, 0.88)
    ax.set_xlabel("AUROC (5-fold stratified CV, mean ± sd)")
    ax.set_title("Diabetes-onset prediction: model discrimination")
    for i, m in enumerate(means):
        ax.text(m + 0.004, i, f"{m:.3f}", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/auroc_by_model.png", dpi=150); plt.close(fig)


def fig_importance(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    _, pr = logreg(Xs, y, Xs)
    w, b = pr["coef"], pr["intercept"]
    predict = lambda Z: 1 / (1 + np.exp(-(Z @ w + b)))
    base, imp = permutation_importance(predict, Xs, y)
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh([COLS[i] for i in order], imp[order], color=RED)
    ax.set_xlabel("Importance = drop in AUROC when feature is shuffled")
    ax.set_title(f"Global feature importance (logistic, base AUROC={base:.3f})")
    fig.tight_layout(); fig.savefig(f"{FIG}/feature_importance.png", dpi=150); plt.close(fig)
    return predict


def fig_roc_and_calibration(X, y, predict):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    p = predict((X - mu) / sd)
    # ROC
    thr = np.linspace(0, 1, 200)
    tpr = [np.mean(p[y == 1] >= t) for t in thr]
    fpr = [np.mean(p[y == 0] >= t) for t in thr]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, color=BLUE, lw=2, label=f"Logistic (AUROC={auroc(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color=GREY)
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC curve"); ax[0].legend(loc="lower right")
    # Calibration (reliability)
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p, bins) - 1
    xs, ys = [], []
    for b_ in range(10):
        m = idx == b_
        if m.sum() > 5:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    ax[1].plot([0, 1], [0, 1], "--", color=GREY, label="Perfect")
    ax[1].plot(xs, ys, "o-", color=BLUE, label="Logistic")
    ax[1].set_xlabel("Predicted risk"); ax[1].set_ylabel("Observed frequency")
    ax[1].set_title("Calibration (reliability diagram)"); ax[1].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/roc_and_calibration.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    X, y = load()
    fig_auroc(X, y)
    predict = fig_importance(X, y)
    fig_roc_and_calibration(X, y, predict)
    print("Saved figures to", FIG)
    print(os.listdir(FIG))
