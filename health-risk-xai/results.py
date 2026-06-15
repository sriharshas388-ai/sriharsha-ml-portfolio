"""Reproducible results + figures for health-risk-xai on the real UCI Heart
Disease dataset (303 patients). Pure NumPy/pandas + matplotlib. Seed = 42."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlkit import (load_heart, cv_benchmark, logreg, permutation_importance,
                   lime_local, auroc)

FIG = os.path.join(os.path.dirname(__file__), "figures"); os.makedirs(FIG, exist_ok=True)
BLUE, GREY, RED = "#1f4e79", "#9bb7d4", "#c0392b"


def main():
    X, y, cols = load_heart()
    print(f"UCI Heart Disease: n={len(X)}, positives={y.mean():.1%}, features={len(cols)}\n")

    rows = cv_benchmark(X, y)
    print("5-fold CV (sorted by AUROC):")
    print("| Model | AUROC±sd | Acc | Brier |\n|---|---|---|---|")
    for m, a, s, ac, b in rows:
        print(f"| {m} | {a:.3f}±{s:.3f} | {ac:.3f} | {b:.3f} |")

    mu, sd = X.mean(0), X.std(0)+1e-9; Xs = (X-mu)/sd
    _, pr = logreg(Xs, y, Xs); w, b = pr["coef"], pr["intercept"]
    predict = lambda Z: 1/(1+np.exp(-(Z@w+b)))
    base, imp = permutation_importance(predict, Xs, y)
    print(f"\nGlobal importance (logistic, AUROC={base:.3f}):")
    for i in np.argsort(-imp): print(f"  {cols[i]}: {imp[i]:.4f}")

    # Figure 1: feature importance
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh([cols[i] for i in order], imp[order], color=RED)
    ax.set_xlabel("Importance = drop in AUROC when feature is shuffled")
    ax.set_title(f"Heart disease — global feature importance (AUROC={base:.3f})")
    fig.tight_layout(); fig.savefig(f"{FIG}/feature_importance.png", dpi=150); plt.close(fig)

    # Figure 2: ROC + calibration
    p = predict(Xs)
    thr = np.linspace(0, 1, 200)
    tpr = [np.mean(p[y == 1] >= t) for t in thr]; fpr = [np.mean(p[y == 0] >= t) for t in thr]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, color=BLUE, lw=2, label=f"Logistic (AUROC={auroc(y,p):.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color=GREY)
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC curve"); ax[0].legend(loc="lower right")
    bins = np.linspace(0, 1, 11); idx = np.digitize(p, bins)-1; xs, ys = [], []
    for bb in range(10):
        m = idx == bb
        if m.sum() > 3: xs.append(p[m].mean()); ys.append(y[m].mean())
    ax[1].plot([0, 1], [0, 1], "--", color=GREY, label="Perfect")
    ax[1].plot(xs, ys, "o-", color=BLUE, label="Logistic")
    ax[1].set_xlabel("Predicted risk"); ax[1].set_ylabel("Observed frequency")
    ax[1].set_title("Calibration"); ax[1].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/roc_and_calibration.png", dpi=150); plt.close(fig)

    # Local explanation
    hi = int(np.argmax(p)); contrib = lime_local(predict, Xs[hi])
    print(f"\nLocal explanation, patient #{hi} (risk={p[hi]:.2f}):")
    for i in np.argsort(-np.abs(contrib))[:5]: print(f"  {cols[i]}: {contrib[i]:+.4f}")
    print("\nfigures ->", os.listdir(FIG))


if __name__ == "__main__":
    main()
