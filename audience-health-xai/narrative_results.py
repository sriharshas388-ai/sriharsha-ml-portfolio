"""Audience-centred explanation demo on the real UCI Heart Disease dataset.

Trains a logistic model, computes per-feature local contributions for one
patient, then renders the SAME explanation two ways — for a clinician and for a
care coordinator — and a contribution figure. Seed = 42. Pure NumPy + matplotlib.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlkit import load_heart, logreg, lime_local

FIG = os.path.join(os.path.dirname(__file__), "figures"); os.makedirs(FIG, exist_ok=True)
GREEN, RED = "#2e7d52", "#c0392b"

PLAIN = {  # plain-language labels for non-CS audiences
    "cp": "chest-pain type", "ca": "number of major vessels seen on imaging",
    "thal": "thalassemia stress-test result", "oldpeak": "ST depression in exercise",
    "thalach": "maximum heart rate achieved", "exang": "exercise-induced angina",
    "sex": "sex", "slope": "slope of peak exercise ST", "age": "age",
    "trestbps": "resting blood pressure", "chol": "cholesterol", "fbs": "fasting blood sugar",
    "restecg": "resting ECG result",
}


def narrate(cols, contrib, risk):
    order = np.argsort(-np.abs(contrib))
    top = [(cols[i], contrib[i]) for i in order[:4]]
    raises = [PLAIN.get(c, c) for c, v in top if v > 0]
    lowers = [PLAIN.get(c, c) for c, v in top if v < 0]
    clinical = (f"Model risk = {risk:.0%}. Leading positive contributors: "
                + ", ".join(f"{c} ({v:+.3f})" for c, v in top if v > 0)
                + ". Mitigating: " + (", ".join(f"{c} ({v:+.3f})" for c, v in top if v < 0) or "none") + ".")
    operational = (f"This patient is flagged HIGH risk ({risk:.0%}). The biggest reasons are "
                   + (", ".join(raises) if raises else "several factors")
                   + (f"; partly offset by {', '.join(lowers)}" if lowers else "")
                   + ". Suggest clinical review.")
    return clinical, operational


def main():
    X, y, cols = load_heart()
    mu, sd = X.mean(0), X.std(0)+1e-9; Xs = (X-mu)/sd
    _, pr = logreg(Xs, y, Xs); w, b = pr["coef"], pr["intercept"]
    predict = lambda Z: 1/(1+np.exp(-(Z@w+b)))
    p = predict(Xs); hi = int(np.argmax(p))
    contrib = lime_local(predict, Xs[hi])

    clinical, operational = narrate(cols, contrib, p[hi])
    print(f"Patient #{hi}\n\n[CLINICAL]\n{clinical}\n\n[OPERATIONAL]\n{operational}\n")
    with open(os.path.join(os.path.dirname(__file__), "outputs_example.md"), "w") as f:
        f.write(f"# Example audience-specific explanation (patient #{hi})\n\n"
                f"**Clinical view**\n\n{clinical}\n\n**Operational view**\n\n{operational}\n")

    order = np.argsort(np.abs(contrib))
    colors = [GREEN if contrib[i] < 0 else RED for i in order]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh([PLAIN.get(cols[i], cols[i]) for i in order], contrib[order], color=colors)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("Local contribution to predicted risk (red = raises, green = lowers)")
    ax.set_title(f"Why this patient is high-risk ({p[hi]:.0%}) — local explanation")
    fig.tight_layout(); fig.savefig(f"{FIG}/local_explanation.png", dpi=150); plt.close(fig)
    print("figure ->", os.listdir(FIG))


if __name__ == "__main__":
    main()
