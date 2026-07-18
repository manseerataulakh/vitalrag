# VitalRAG — ICU Deterioration Dashboard

A clinical decision-support prototype that combines time-series vital signs analysis with retrieval-augmented generation (RAG) over real MIMIC-IV clinical notes to predict ICU patient deterioration risk.

> **Not for clinical use.** Built on the [MIMIC-IV Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/2.2/) (de-identified).

---

## Overview

VitalRAG fuses two independent risk signals:

| Signal | Source | Model |
|--------|--------|-------|
| **Time-series risk** | Hourly ICU vitals (HR, SBP, SpO₂, RR) | Gradient Boosting Classifier with NEWS2 score, shock index, and vital crossing features |
| **Notes risk** | Real MIMIC-IV ICD diagnoses + demographics | RAG retrieval (FAISS + sentence-transformers) + Gemini summarisation |

The two scores are fused as `0.65 × ts_risk + 0.35 × text_risk`.

---

## Features

- **ICU patient sidebar** — all patients ranked by fused risk with colour-coded pills (critical / elevated / stable)
- **Three separate vitals charts** — Heart Rate, SpO₂, and Respiratory Rate over the first 24h
- **Clinical evidence panel** — top retrieved note snippets with relevance scores and an LLM-generated summary
- **Live Q&A** — ask any free-text question about the selected patient; RAG retrieves relevant note passages and Gemini answers using only those sources
- **15/15 correct classification** on the dashboard patient cohort

---

## Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19 + TypeScript + Vite + Recharts |
| Backend | FastAPI + Uvicorn |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS (flat inner-product) |
| LLM | Gemini 2.5 Flash Lite |
| ML model | scikit-learn GradientBoostingClassifier |
| Data | MIMIC-IV Demo v2.2 |

---

## Project Structure

```
vitalrag/
├── data/
│   ├── vitals.csv          # hourly ICU vitals (stay_id keyed)
│   ├── notes.csv           # real MIMIC diagnoses + demographics per stay
│   └── labels.csv          # ground truth deterioration labels
├── src/
│   ├── api.py              # FastAPI endpoints (/patients, /patient, /ask)
│   ├── precompute.py       # batch prediction pipeline
│   └── models/
│       ├── timeseries.py   # feature engineering + GBM training
│       ├── rag.py          # FAISS index build + retrieval
│       ├── llm.py          # Gemini summarise + Q&A
│       └── fusion.py       # weighted score fusion
├── frontend/
│   └── src/
│       ├── App.tsx         # main dashboard component
│       └── dashboard.css   # design tokens + styles
└── results/
    ├── predictions.json    # precomputed risk scores + evidence
    ├── notes.parquet       # indexed notes
    └── notes.index         # FAISS index
```

---

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- [MIMIC-IV Demo access](https://physionet.org/content/mimic-iv-demo/2.2/) (free PhysioNet account)
- Gemini API key ([ai.google.dev](https://ai.google.dev))

### Backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# add your Gemini key
echo "GEMINI_API_KEY=your_key_here" > .env

# build FAISS index + run precompute (calls Gemini — free tier: 20 req/day)
python -m src.models.rag
python -m src.precompute

# start API
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd frontend
echo "VITE_API_URL=http://localhost:8000" > .env
npm install
npm run dev
```

Open `http://localhost:5173`.

---

## Model Performance

Trained on 136 MIMIC-IV demo ICU stays (64 positive, 72 negative):

| Metric | Score |
|--------|-------|
| Cross-validated AUROC | 0.672 |
| Dashboard accuracy (15 patients) | 15/15 |

Features include: per-vital mean/std/min/max/trend, NEWS2 score (mean/max/trend), shock index (HR/SBP), and vital crossing fractions (hours with HR > 100, SpO₂ < 94, RR > 20, SBP < 90).
