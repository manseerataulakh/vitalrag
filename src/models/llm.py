import os, json
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-2.5-flash-lite"

def summarize(snippets):
    if not snippets:
        return {"summary": "No notes available.", "risk_signal": 0.5}
    joined = "\n".join(f"- {s['text']}" for s in snippets)
    prompt = (
        "You are assisting ICU clinicians. Given these retrieved nursing notes, "
        "return STRICT JSON only, no other text, no markdown fences: "
        '{"summary": "<=25 words", "risk_signal": <number between 0 and 1>}. '
        "risk_signal reflects how concerning these notes are for patient deterioration. "
        "Notes:\n" + joined
    )
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    txt = (resp.text or "").strip()
    txt = txt.replace("```json", "").replace("```", "").strip()
    try:
        out = json.loads(txt)
        out["risk_signal"] = float(out["risk_signal"])
        return out
    except Exception:
        return {"summary": "Summary unavailable.", "risk_signal": 0.5}

def answer_question(question: str, snippets: list) -> str:
    if not snippets:
        return "No relevant notes found for this patient."
    joined = "\n".join(f"- {s['text']}" for s in snippets)
    prompt = (
        "You are assisting ICU clinicians. Answer the following question using ONLY "
        "the provided nursing note excerpts. Be concise (2-3 sentences). "
        "If the notes don't contain enough information, say so.\n\n"
        f"Question: {question}\n\nNotes:\n{joined}"
    )
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    return (resp.text or "").strip()

if __name__ == "__main__":
    import sys, pandas as pd
    sys.path.append(os.path.dirname(__file__))
    from rag import retrieve
    pid = pd.read_csv("data/notes.csv").patient_id.iloc[0]
    snippets = retrieve(pid)
    print("Retrieved for patient", pid)
    for s in snippets:
        print("  ", s["text"])
    print("LLM output:", summarize(snippets))