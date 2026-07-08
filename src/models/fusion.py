import os, joblib, pandas as pd

def predict(patient_id, alpha=0.65):
    """Combine time-series risk and notes/LLM risk into one prediction."""
    # local imports so this works when run as a module
    from src.models.timeseries import features, VITALS
    from src.models.rag import retrieve
    from src.models.llm import summarize

    # --- time-series branch ---
    clf = joblib.load("results/ts_model.pkl")
    vitals = pd.read_csv("data/vitals.csv")
    pv = vitals[vitals.patient_id == patient_id]
    if len(pv) == 0:
        raise ValueError(f"No vitals for patient {patient_id}")
    X = features(pv).drop(columns=["patient_id"])
    ts_risk = float(clf.predict_proba(X)[:, 1][0])

    # --- notes / LLM branch ---
    snippets = retrieve(patient_id)
    llm_out = summarize(snippets)
    text_risk = float(llm_out["risk_signal"])

    # --- fusion: weighted blend ---
    fused = alpha * ts_risk + (1 - alpha) * text_risk

    return {
        "patient_id": int(patient_id),
        "ts_risk": round(ts_risk, 3),
        "text_risk": round(text_risk, 3),
        "fused_risk": round(fused, 3),
        "summary": llm_out["summary"],
        "evidence": snippets,
    }

if __name__ == "__main__":
    pid = pd.read_csv("data/notes.csv").patient_id.iloc[0]
    result = predict(pid)
    import json
    print(json.dumps(result, indent=2))