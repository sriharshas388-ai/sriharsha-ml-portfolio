"""
Reproducible diabetes-risk benchmark + explainability on the Pima Indians
Diabetes dataset (real public data).

Pipeline:
  1. Load data (downloads on first run; caches to pima.csv).
  2. Preprocess: zeros in Glucose/BloodPressure/SkinThickness/Insulin/BMI are
     physiologically impossible and encode MISSING values (a well-known quirk of
     this dataset) -> set to NaN and median-impute.
  3. Benchmark six classifiers with stratified 5-fold cross-validation,
     reporting AUROC, accuracy and Brier (calibration).
  4. Explain the logistic model with permutation importance (global) and a
     LIME-style local linear surrogate for one high-risk patient.

Pure NumPy/pandas: no scikit-learn needed, so every number is reproducible
anywhere. For library SHAP/LIME see the README.

Dataset: Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., &
Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset
of diabetes mellitus. Proc. Annu. Symp. Comput. Appl. Med. Care, 261-265.
"""
from __future__ import annotations
import io, os, urllib.request
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "dpf", "age"]
ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
CACHE = os.path.join(os.path.dirname(__file__), "pima.csv")


def load():
    if os.path.exists(CACHE):
        raw = pd.read_csv(CACHE, header=None)
    else:
        txt = urllib.request.urlopen(URL, timeout=30).read().decode()
        raw = pd.read_csv(io.StringIO(txt), header=None)
        raw.to_csv(CACHE, header=False, index=False)
    raw.columns = COLS + ["outcome"]
    X = raw[COLS].astype(float).copy()
    for c in ZERO_AS_MISSING:
        X.loc[X[c] == 0, c] = np.nan
    X = X.fillna(X.median())                      # median imputation
    return X.values, raw["outcome"].astype(float).values


# ---------- metrics ----------
def auroc(y, s):
    order = np.argsort(s); ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(s)+1)
    pos = y == 1; npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[pos].sum() - npos*(npos+1)/2) / (npos*nneg)


def brier(y, p): return float(np.mean((p-y)**2))
def acc(y, p):  return float(np.mean((p >= 0.5) == y))


# ---------- models (NumPy) ----------
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
    mx = np.maximum(lp[0], lp[1])
    return np.exp(lp[1]-mx)/(np.exp(lp[0]-mx)+np.exp(lp[1]-mx)), {}

def knn(Xtr, ytr, Xte, k=15):
    out = np.empty(len(Xte))
    for i, x in enumerate(Xte):
        idx = np.argpartition(((Xtr-x)**2).sum(1), k)[:k]; out[i] = ytr[idx].mean()
    return out, {}

def lda(Xtr, ytr, Xte):
    mu = {c: Xtr[ytr == c].mean(0) for c in (0, 1)}
    Sw = np.zeros((Xtr.shape[1],)*2)
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
        y = ytr[idx]
        if d == 0 or len(idx) < 20 or y.mean() in (0, 1):
            return {"leaf": float(y.mean()) if len(y) else 0.5}
        best = None
        for f in range(Xtr.shape[1]):
            for t in np.quantile(Xtr[idx, f], np.linspace(.2, .8, 7)):
                l = idx[Xtr[idx, f] <= t]; r = idx[Xtr[idx, f] > t]
                if len(l) < 10 or len(r) < 10: continue
                g = (len(l)*_gini(ytr[l])+len(r)*_gini(ytr[r]))/len(idx)
                if best is None or g < best[0]: best = (g, f, t, l, r)
        if best is None: return {"leaf": float(y.mean())}
        _, f, t, l, r = best
        return {"f": f, "t": t, "L": build(l, d-1), "R": build(r, d-1)}
    def pr(n, x):
        while "leaf" not in n: n = n["L"] if x[n["f"]] <= n["t"] else n["R"]
        return n["leaf"]
    root = build(np.arange(len(Xtr)), depth)
    return np.array([pr(root, x) for x in Xte]), {}

def forest(Xtr, ytr, Xte, n=30, depth=4):
    acc_ = np.zeros(len(Xte))
    for _ in range(n):
        bs = RNG.integers(0, len(Xtr), len(Xtr))
        acc_ += tree(Xtr[bs], ytr[bs], Xte, depth)[0]
    return acc_/n, {}

MODELS = {"Logistic Regression": logreg, "Gaussian Naive Bayes": gnb,
          "k-NN (k=15)": knn, "Linear Discriminant Analysis": lda,
          "Decision Tree (d=4)": tree, "Random Forest (30xd4)": forest}


def stratified_folds(y, k=5):
    folds = [[] for _ in range(k)]
    for c in (0, 1):
        idx = RNG.permutation(np.where(y == c)[0])
        for i, j in enumerate(idx): folds[i % k].append(j)
    return [np.array(sorted(f)) for f in folds]


def cv_benchmark(X, y, k=5):
    folds = stratified_folds(y, k)
    res = {m: {"auroc": [], "acc": [], "brier": []} for m in MODELS}
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
        Xtr, Xte = (X[tr]-mu)/sd, (X[te]-mu)/sd
        for m, fn in MODELS.items():
            p = np.clip(fn(Xtr, y[tr], Xte)[0], 1e-6, 1-1e-6)
            res[m]["auroc"].append(auroc(y[te], p))
            res[m]["acc"].append(acc(y[te], p))
            res[m]["brier"].append(brier(y[te], p))
    rows = [(m, np.mean(v["auroc"]), np.std(v["auroc"]), np.mean(v["acc"]), np.mean(v["brier"]))
            for m, v in res.items()]
    rows.sort(key=lambda r: -r[1])
    return rows


def permutation_importance(predict, X, y, reps=10):
    base = auroc(y, predict(X)); imp = np.zeros(X.shape[1])
    for f in range(X.shape[1]):
        d = []
        for _ in range(reps):
            Xp = X.copy(); Xp[:, f] = RNG.permutation(Xp[:, f])
            d.append(base-auroc(y, predict(Xp)))
        imp[f] = np.mean(d)
    return base, imp


def lime_local(predict, x, n=2000, kernel=0.75):
    S = x + RNG.normal(0, 1.0, (n, len(x))); yp = predict(S)
    dist = np.sqrt(((S-x)**2).sum(1)); wts = np.exp(-(dist**2)/(2*kernel**2))+1e-9
    A = np.column_stack([np.ones(n), S-x]); W = np.diag(wts)
    return (np.linalg.pinv(A.T@W@A)@(A.T@W@yp))[1:]


if __name__ == "__main__":
    X, y = load()
    print(f"Pima Indians Diabetes: n={len(X)}, positives={y.mean():.1%}, features={len(COLS)}\n")
    print("5-fold stratified CV:\n")
    print("| Model | AUROC (mean±sd) | Accuracy | Brier |")
    print("|---|---|---|---|")
    for m, a, asd, ac, br in cv_benchmark(X, y):
        print(f"| {m} | {a:.3f} ± {asd:.3f} | {ac:.3f} | {br:.3f} |")

    # explain logistic on a standardized full-data fit
    mu, sd = X.mean(0), X.std(0)+1e-9; Xs = (X-mu)/sd
    _, pr = logreg(Xs, y, Xs); w, b = pr["coef"], pr["intercept"]
    predict = lambda Z: 1/(1+np.exp(-(Z@w+b)))
    base, imp = permutation_importance(predict, Xs, y)
    print(f"\nGlobal permutation importance (logistic, AUROC={base:.3f}):")
    print("| Feature | AUROC drop |\n|---|---|")
    for i in np.argsort(-imp): print(f"| {COLS[i]} | {imp[i]:.4f} |")
    hi = int(np.argmax(predict(Xs))); contrib = lime_local(predict, Xs[hi])
    print(f"\nLocal LIME-style explanation, patient #{hi} (risk={predict(Xs)[hi]:.2f}):")
    print("| Feature | Contribution |\n|---|---|")
    for i in np.argsort(-np.abs(contrib)): print(f"| {COLS[i]} | {contrib[i]:+.4f} |")
