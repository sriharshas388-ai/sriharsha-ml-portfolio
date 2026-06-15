"""
Predictive Analytics for Health Risk Stratification
ML applications in non-hospital COVID-19 treatment eligibility.

Cleaned, runnable version of the MSc dissertation notebook
(Sriharsha Surannagari, W9607526, Teesside University, 2025). The original
exported notebook is in SourceCode_original_notebook.pdf.

The dataset (DataSet.csv) is an NHS-published table of high-risk patient counts
by clinical condition group and demographic band. Place it next to this script.

Requires: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, RandomForestRegressor,
                              GradientBoostingClassifier as GBC)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, mean_squared_error, r2_score)
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42


# ----------------------------- data -----------------------------
def load_and_prepare(path="DataSet.csv"):
    data = pd.read_csv(path)
    data.rename(columns={
        "Asian/Asian British ": "Asian",
        "Black/African/Caribbean/Black British ": "Black",
        "Mixed/Multiple ethnic groups ": "Mixed",
        "Not Stated ": "Not_Stated",
        "Other ethnic group ": "Other",
        "Clinical condition group ": "Condition_Group",
    }, inplace=True)

    # binary target: above-median total count = high risk
    data["High_Risk"] = (data["Total"] > data["Total"].median()).astype(int)

    drop_cols = ["Condition_Group", "Total", "High_Risk", "Female", "Male",
                 "12 to 16 ", "17 to 19 ", "20 to 29 ", "30 to 39 ", "40 to 49 ",
                 "50 to 59 ", "60 to 69 ", "70 to 79 ", "80 to 89 ", "90+"]
    X = data.drop(columns=[c for c in drop_cols if c in data.columns])
    y = data["High_Risk"]
    return data, X, y


# --------------------------- modelling ---------------------------
CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
}

PARAM_GRIDS = {
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [10, 20]},
    "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
}


def run_classification(X, y):
    # class balancing (the published counts are small / imbalanced)
    X_bal, y_bal = SMOTE(random_state=RANDOM_STATE, k_neighbors=1).fit_resample(X, y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=RANDOM_STATE)

    rows, cms = [], {}
    for name, model in CLASSIFIERS.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        rep = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
        rows.append({
            "Model": name,
            "Accuracy": rep["accuracy"],
            "Precision": rep.get("1", {}).get("precision", 0.0),
            "Recall": rep.get("1", {}).get("recall", 0.0),
            "F1-Score": rep.get("1", {}).get("f1-score", 0.0),
            "AUC-ROC": roc_auc_score(y_te, y_prob),
        })
        cms[name] = confusion_matrix(y_te, y_pred)

    # light hyperparameter tuning (RF, GB)
    for name, grid in PARAM_GRIDS.items():
        gs = GridSearchCV(CLASSIFIERS[name], grid, cv=3)
        gs.fit(X_tr, y_tr)
        print(f"Best params {name}: {gs.best_params_} (CV {gs.best_score_:.3f})")

    return pd.DataFrame(rows), cms


def run_regression(data, X):
    """Regression view: predict the (log) total count per group."""
    y = np.log1p(data["Total"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    regressors = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting": __import__("sklearn.ensemble", fromlist=["GradientBoostingRegressor"]).GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    rows = []
    for name, model in regressors.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        mse = mean_squared_error(y_te, pred)
        rows.append({"Model": name, "MSE": mse, "RMSE": np.sqrt(mse),
                     "R2": r2_score(y_te, pred)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data, X, y = load_and_prepare()
    clf_df, _ = run_classification(X, y)
    print("\nClassification results:\n", clf_df.to_string(index=False))
    clf_df.to_csv("classification_results.csv", index=False)
    reg_df = run_regression(data, X)
    print("\nRegression results:\n", reg_df.to_string(index=False))
