import pandas as pd, numpy as np, faiss, os

MODEL_NAME = "all-MiniLM-L6-v2"
QUERY = "signs of clinical deterioration, distress, or worsening condition"

# load torch/sentence-transformers lazily, only when a request actually
# needs the model - importing them eagerly at module load time (which
# happens on every app startup via api.py) kept the process near the
# 512MB Render free-tier ceiling even for requests that never touch /ask.
_model = None
def get_model():
    global _model
    if _model is None:
        import torch
        torch.set_num_threads(1)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def build_index():
    notes = pd.read_csv("data/notes.csv")
    model = get_model()
    emb = model.encode(notes["text"].tolist(), normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    os.makedirs("results", exist_ok=True)
    faiss.write_index(index, "results/notes.index")
    notes.to_parquet("results/notes.parquet")
    print("indexed", len(notes), "notes")

def retrieve(patient_id, k=3):
    notes = pd.read_parquet("results/notes.parquet")
    pnotes = notes[notes.patient_id == patient_id].reset_index(drop=True)
    if len(pnotes) == 0:
        return []
    model = get_model()
    q = model.encode([QUERY], normalize_embeddings=True).astype("float32")
    emb = model.encode(pnotes["text"].tolist(), normalize_embeddings=True).astype("float32")
    sims = (emb @ q.T).ravel()
    order = np.argsort(-sims)[:k]
    return [{"text": str(pnotes.text.iloc[i]), "score": float(sims[i])} for i in order]

if __name__ == "__main__":
    build_index()
    # test on the first patient in the notes file
    first_pid = pd.read_csv("data/notes.csv").patient_id.iloc[0]
    print("Sample retrieval for patient", first_pid)
    for r in retrieve(first_pid):
        print(round(r["score"], 3), "-", r["text"])