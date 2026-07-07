import pandas as pd, numpy as np, joblib, os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

VITALS = ["hr", "sbp", "spo2", "rr"]

def features(vitals_df):
    """Turn each patient's hourly vitals into summary numbers the model can learn from."""
    rows = []
    for pid, g in vitals_df.groupby("patient_id"):
        g = g.sort_values("hour")
        feat = {"patient_id": pid}
        for v in VITALS:
            feat[f"{v}_mean"] = g[v].mean()
            feat[f"{v}_std"] = g[v].std()
            feat[f"{v}_min"] = g[v].min()
            feat[f"{v}_max"] = g[v].max()
            feat[f"{v}_trend"] = g[v].iloc[-1] - g[v].iloc[0]
        rows.append(feat)
    out = pd.DataFrame(rows)
    return out.fillna(out.mean(numeric_only=True))

def train():
    vitals = pd.read_csv("data/vitals.csv")
    labels = pd.read_csv("data/labels.csv")
    X = features(vitals).merge(labels, on="patient_id")
    y = X.pop("label")
    X.pop("patient_id")

    print("patients:", len(y), "| positives:", int(y.sum()), "| negatives:", int((y==0).sum()))

    clf = GradientBoostingClassifier(random_state=1)

    # cross-validated predictions: every patient gets a held-out prediction
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    print("Cross-validated AUROC:", round(roc_auc_score(y, p), 3),
          "| AUPRC:", round(average_precision_score(y, p), 3))

    # fit final model on all data and save it (for the API later)
    clf.fit(X, y)
    os.makedirs("results", exist_ok=True)
    joblib.dump(clf, "results/ts_model.pkl")
    print("saved results/ts_model.pkl")
    return clf

if __name__ == "__main__":
    train()