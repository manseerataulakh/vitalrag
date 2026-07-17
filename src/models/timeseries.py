import pandas as pd, numpy as np, joblib, os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

VITALS = ["hr", "sbp", "spo2", "rr"]

def _news2_rr(rr):
    if pd.isna(rr): return np.nan
    if rr <= 8: return 3
    if rr <= 11: return 1
    if rr <= 20: return 0
    if rr <= 24: return 2
    return 3

def _news2_spo2(spo2):
    if pd.isna(spo2): return np.nan
    if spo2 >= 96: return 0
    if spo2 >= 94: return 1
    if spo2 >= 92: return 2
    return 3

def _news2_sbp(sbp):
    if pd.isna(sbp): return np.nan
    if sbp <= 90: return 3
    if sbp <= 100: return 2
    if sbp <= 110: return 1
    if sbp <= 219: return 0
    return 3

def _news2_hr(hr):
    if pd.isna(hr): return np.nan
    if hr <= 40: return 3
    if hr <= 50: return 1
    if hr <= 90: return 0
    if hr <= 110: return 1
    if hr <= 130: return 2
    return 3

def _news2_temp(temp):
    if pd.isna(temp): return np.nan
    if temp <= 35.0: return 3
    if temp <= 36.0: return 1
    if temp <= 38.0: return 0
    if temp <= 39.0: return 1
    return 2

def _news2_row(row):
    scores = [
        _news2_rr(row.get("rr")),
        _news2_spo2(row.get("spo2")),
        _news2_sbp(row.get("sbp")),
        _news2_hr(row.get("hr")),
        _news2_temp(row.get("temp")),
    ]
    valid = [s for s in scores if not np.isnan(s)]
    return sum(valid) if valid else np.nan

def features(vitals_df):
    rows = []
    for pid, g in vitals_df.groupby("patient_id"):
        g = g.sort_values("hour").reset_index(drop=True)
        feat = {"patient_id": pid}

        # basic stats per vital
        for v in VITALS:
            feat[f"{v}_mean"] = g[v].mean()
            feat[f"{v}_std"]  = g[v].std()
            feat[f"{v}_min"]  = g[v].min()
            feat[f"{v}_max"]  = g[v].max()
            feat[f"{v}_trend"] = g[v].iloc[-1] - g[v].iloc[0]

        # NEWS2 score per hour, then aggregate
        news = g.apply(_news2_row, axis=1)
        feat["news2_mean"] = news.mean()
        feat["news2_max"]  = news.max()
        feat["news2_trend"] = news.iloc[-1] - news.iloc[0]

        # shock index (HR / SBP) — elevated when > 0.7
        si = g["hr"] / g["sbp"].replace(0, np.nan)
        feat["shock_index_mean"] = si.mean()
        feat["shock_index_max"]  = si.max()

        # vital crossing counts (abnormal hours)
        n = max(len(g), 1)
        feat["hr_high_frac"]   = (g["hr"] > 100).sum() / n
        feat["spo2_low_frac"]  = (g["spo2"] < 94).sum() / n
        feat["rr_high_frac"]   = (g["rr"] > 20).sum() / n
        feat["sbp_low_frac"]   = (g["sbp"] < 90).sum() / n

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    print("Cross-validated AUROC:", round(roc_auc_score(y, p), 3),
          "| AUPRC:", round(average_precision_score(y, p), 3))

    clf.fit(X, y)
    os.makedirs("results", exist_ok=True)
    joblib.dump(clf, "results/ts_model.pkl")
    print("saved results/ts_model.pkl")
    return clf

if __name__ == "__main__":
    train()
