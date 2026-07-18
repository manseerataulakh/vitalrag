import pandas as pd, numpy as np, faiss, os
from google.genai import types
from src.models.llm import client

EMBED_MODEL = "gemini-embedding-001"
QUERY = "signs of clinical deterioration, distress, or worsening condition"

def embed_texts(texts: list[str], task_type: str) -> np.ndarray:
    """Embed via the Gemini API (task_type: RETRIEVAL_QUERY or RETRIEVAL_DOCUMENT)."""
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return np.array([e.values for e in resp.embeddings], dtype="float32")

def build_index():
    notes = pd.read_csv("data/notes.csv")
    emb = embed_texts(notes["text"].tolist(), "RETRIEVAL_DOCUMENT")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    os.makedirs("results", exist_ok=True)
    faiss.write_index(index, "results/notes.index")
    notes.to_parquet("results/notes.parquet")
    print("indexed", len(notes), "notes")

def retrieve(patient_id, k=3):
    notes = pd.read_parquet("results/notes.parquet")
    pnotes = notes[notes.patient_id == patient_id].reset_index(drop=True)
    if len(pnotes) == 0:
        return []
    q = embed_texts([QUERY], "RETRIEVAL_QUERY")
    emb = embed_texts(pnotes["text"].tolist(), "RETRIEVAL_DOCUMENT")
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
