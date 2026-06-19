"""Compact NumPy ML kit (models, metrics, CV, explainability) — no sklearn.
Shared by the result scripts so every figure is reproducible (seed = 42).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def load_heart(path=None):
    path = path or os.path.join(os.path.dirname(__file__), "heart.csv")
    df = pd.read_csv(path)
    y = df["target"].astype(float).values
    X = df.drop(columns=["target"]).astype(float)
    return X.values, y, list(X.columns)


def auroc(y, s):
    order = np.argsort(s); r = np.empty_like(order, float); r[order] = np.arange(1, len(s)+1)
    pos = y == 1; npos, nneg = pos.sum(), (~pos).sum()
    return float("nan") if npos == 0 or nneg == 0 else (r[pos].sum()-npos*(npos+1)/2)/(npos*nneg)


def brier(y, p): return float(np.mean((p-y)**2))
def acc(y, p):  return float(np.mean((p >= 0.5) == y))


def auprc(y, s):
    """Area under the precision-recall curve (average precision).
    For imbalanced clinical data AUPRC is more informative than AUROC because
    precision reacts to false positives under low prevalence
    (Saito & Rehmsmeier, 2015; McDermott et al., 2024)."""
    order = np.argsort(-s); yt = y[order]
    tp = np.cumsum(yt); fp = np.cumsum(1-yt); P = yt.sum()
    if P == 0: return float("nan")
    precision = tp/(tp+fp); recall = tp/P
    recall = np.concatenate([[0.0], recall]); precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:]-recall[:-1])*precision[1:]))  # step-wise AP


def ece(y, p, bins=10):
    """Expected Calibration Error: weighted gap between confidence and accuracy
    across equal-width probability bins (Huang et al., 2020). Lower is better."""
    edges = np.linspace(0, 1, bins+1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0: continue
        e += m.mean()*abs(y[m].mean()-p[m].mean())
    return float(e)


def logreg(Xtr, ytr, Xte, lr=0.1, epochs=600, l2=1e-3):
    w = np.zeros(Xtr.shape[1]); b = 0.0
    for _ in range(epochs):
        p = 1/(1+np.exp(-(Xtr@w+b))); g = p-ytr
        w -= lr*(Xtr.T@g/len(ytr)+l2*w); b -= lr*g.mean()
    return 1/(1+np.exp(-(Xte@w+b))), {"coef": w, "intercept": b}


def gnb(Xtr, ytr, Xte):
    lp = {}
    for c in (0, 1):
        Xc = Xtr[ytr == c]; m, v = Xc.mean(0), Xc.var(0)+1e-6
        lp[c] = (-0.5*np.log(2*np.pi*v)-(Xte-m)**2/(2*v)).sum(1)+np.log(len(Xc)/len(Xtr))
    mx = np.maximum(lp[0], lp[1]); return np.exp(lp[1]-mx)/(np.exp(lp[0]-mx)+np.exp(lp[1]-mx)), {}


def knn(Xtr, ytr, Xte, k=15):
    out = np.empty(len(Xte))
    for i, x in enumerate(Xte):
        out[i] = ytr[np.argpartition(((Xtr-x)**2).sum(1), k)[:k]].mean()
    return out, {}


def lda(Xtr, ytr, Xte):
    mu = {c: Xtr[ytr == c].mean(0) for c in (0, 1)}; Sw = np.zeros((Xtr.shape[1],)*2)
    for c in (0, 1):
        d = Xtr[ytr == c]-mu[c]; Sw += d.T@d
    w = np.linalg.pinv(Sw/len(Xtr))@(mu[1]-mu[0]); s = Xtr@w
    thr = (s[ytr == 1].mean()+s[ytr == 0].mean())/2
    return 1/(1+np.exp(-((Xte@w)-thr))), {"coef": w}


def _gini(y):
    if len(y) == 0: return 0.0
    p = y.mean(); return 1-p*p-(1-p)**2


def tree(Xtr, ytr, Xte, depth=4):
    def build(idx, d):
        yv = ytr[idx]
        if d == 0 or len(idx) < 20 or yv.mean() in (0, 1):
            return {"leaf": float(yv.mean()) if len(yv) else 0.5}
        best = None
        for f in range(Xtr.shape[1]):
            for t in np.quantile(Xtr[idx, f], np.linspace(.2, .8, 7)):
                l = idx[Xtr[idx, f] <= t]; r = idx[Xtr[idx, f] > t]
                if len(l) < 10 or len(r) < 10: continue
                g = (len(l)*_gini(ytr[l])+len(r)*_gini(ytr[r]))/len(idx)
                if best is None or g < best[0]: best = (g, f, t, l, r)
        if best is None: return {"leaf": float(yv.mean())}
        _, f, t, l, r = best
        return {"f": f, "t": t, "L": build(l, d-1), "R": build(r, d-1)}
    def pr(n, x):
        while "leaf" not in n: n = n["L"] if x[n["f"]] <= n["t"] else n["R"]
        return n["leaf"]
    root = build(np.arange(len(Xtr)), depth)
    return np.array([pr(root, x) for x in Xte]), {}


def forest(Xtr, ytr, Xte, n=30, depth=4):
    a = np.zeros(len(Xte))
    for _ in range(n):
        bs = RNG.integers(0, len(Xtr), len(Xtr)); a += tree(Xtr[bs], ytr[bs], Xte, depth)[0]
    return a/n, {}


MODELS = {"Logistic Regression": logreg, "Gaussian Naive Bayes": gnb, "k-NN (k=15)": knn,
          "Linear Discriminant Analysis": lda, "Decision Tree (d=4)": tree,
          "Random Forest (30xd4)": forest}


def stratified_folds(y, k=5):
    folds = [[] for _ in range(k)]
    for c in (0, 1):
        idx = RNG.permutation(np.where(y == c)[0])
        for i, j in enumerate(idx): folds[i % k].append(j)
    return [np.array(sorted(f)) for f in folds]


def cv_benchmark(X, y, k=5):
    folds = stratified_folds(y, k)
    res = {m: {"a": [], "ac": [], "b": [], "ap": [], " e": []} for m in MODELS}
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
        Xtr, Xte = (X[tr]-mu)/sd, (X[te]-mu)/sd
        for m, fn in MODELS.items():
            p = np.clip(fn(Xtr, y[tr], Xte)[0], 1e-6, 1-1e-6)
            res[m]["a"].append(auroc(y[te], p)); res[m]["ac"].append(acc(y[te], p)); res[m]["b"].append(brier(y[te], p))
            res[m]["ap"].append(auprc(y[te], p)); res[m][" e"].append(ece(y[te], p))
    rows = [(m, np.mean(v["a"]), np.std(v["a"]), np.mean(v["ac"]), np.mean(v["b"]),
             np.mean(v["ap"]), np.mean(v[" e"])) for m, v in res.items()]
    rows.sort(key=lambda r: -r[1]); return rows


def permutation_importance(predict, X, y, reps=10):
    base = auroc(y, predict(X)); imp = np.zeros(X.shape[1])
    for f in range(X.shape[1]):
        d = []
        for _ in range(reps):
            Xp = X.copy(); Xp[:, f] = RNG.permutation(Xp[:, f]); d.append(base-auroc(y, predict(Xp)))
        imp[f] = np.mean(d)
    return base, imp


def lime_local(predict, x, n=2000, kernel=0.75):
    S = x + RNG.normal(0, 1.0, (n, len(x))); yp = predict(S)
    dist = np.sqrt(((S-x)**2).sum(1)); w = np.exp(-(dist**2)/(2*kernel**2))+1e-9
    A = np.column_stack([np.ones(n), S-x]); W = np.diag(w)
    return (np.linalg.pinv(A.T@W@A)@(A.T@W@yp))[1:]
