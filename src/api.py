import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from src.models.rag import get_model
from src.models.llm import answer_question

app = FastAPI(title="VitalRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load precomputed data once at startup
with open("results/predictions.json") as f:
    PRED = json.load(f)
with open("results/vitals_demo.json") as f:
    VITALS = json.load(f)

NOTES = pd.read_parquet("results/notes.parquet")

# assign stable ICU-NNN codes, highest risk first
_sorted_pids = sorted(PRED.keys(), key=lambda pid: PRED[pid]["fused_risk"], reverse=True)
ICU_CODE = {pid: f"ICU-{i+1:03d}" for i, pid in enumerate(_sorted_pids)}

@app.get("/patients")
def patients():
    """List all patients with their fused risk (for the dropdown)."""
    return [
        {"id": int(pid), "code": ICU_CODE[pid], "fused_risk": p["fused_risk"]}
        for pid, p in PRED.items()
    ]

@app.get("/patient/{pid}")
def patient(pid: int):
    """Full prediction + vitals for one patient."""
    p = PRED.get(str(pid))
    vit = [v for v in VITALS if v["patient_id"] == pid]
    return {"prediction": {**p, "code": ICU_CODE.get(str(pid), "")}, "vitals": vit}

@app.get("/ask")
def ask(pid: int, q: str = Query(..., min_length=1)):
    """Answer a free-text question about a patient using RAG over their notes."""
    pnotes = NOTES[NOTES.patient_id == pid].reset_index(drop=True)
    if len(pnotes) == 0:
        return {"answer": "No notes available for this patient.", "evidence": []}

    model = get_model()
    q_emb = model.encode([q], normalize_embeddings=True).astype("float32")
    n_emb = model.encode(pnotes["text"].tolist(), normalize_embeddings=True).astype("float32")
    sims = (n_emb @ q_emb.T).ravel()
    order = np.argsort(-sims)[:3]
    evidence = [{"text": str(pnotes.text.iloc[i]), "score": float(sims[i])} for i in order]

    answer = answer_question(q, evidence)
    return {"answer": answer, "evidence": evidence}

@app.get("/health")
def health():
    return {"ok": True}
