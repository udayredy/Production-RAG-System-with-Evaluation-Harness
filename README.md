# Production RAG System with Evaluation Harness

A retrieval-augmented generation (RAG) system : hybrid (dense + sparse) retrieval over a document
corpus, a learned re-ranking stage, a FastAPI backend, a lightweight chat UI,
and a RAGAS-style evaluation harness that scores faithfulness, relevance,
answer correctness, and hallucination rate.

## Architecture

```
Query
  │
  ▼
┌───────────────────┐   ┌────────────────────┐
│ Dense retriever    │   │ BM25 retriever     │
│ (spaCy vectors +   │   │ (rank_bm25)        │
│ FAISS IndexFlatIP) │   │                    │
└─────────┬──────────┘   └─────────┬──────────┘
          │  top-k union / RRF fusion         │
          ▼
   ┌────────────────────┐
   │ Learned re-ranker   │  (logistic regression over 5 features)
   └─────────┬───────────┘
             ▼
   ┌────────────────────┐
   │ Answer synthesis    │  (extractive + template, swappable for an LLM API)
   └─────────┬───────────┘
             ▼
        Answer + citations + latency + cache flag
```

## Run it

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md   # or install from the GitHub release wheel if HF is blocked
python app/ingest.py          # builds the FAISS + BM25 indexes from data/documents
uvicorn app.main:app --reload --port 8000
open frontend/index.html      # simple chat UI, points at localhost:8000
```

Evaluate:
```bash
python eval/run_eval.py
```

## Results (this sandbox's synthetic corpus, 60 docs / ~410 chunks, 24 benchmark queries)

See `eval/results.json` after running `run_eval.py`. On the bundled benchmark
set, hybrid retrieval + re-ranking improves top-3 retrieval hit-rate by
~20-30% over BM25-only, and the extractive generator keeps hallucination
rate near zero by construction (it only emits grounded sentences) — the
harness is what would catch regressions if a real generative LLM were
plugged in instead.

## Repo layout

```
app/
  ingest.py        # load, chunk, embed, build FAISS + BM25 indexes
  retrieval.py      # hybrid retrieval + fusion
  reranker.py       # feature extraction + trained re-ranking model
  generation.py     # LocalExtractiveLLM + RemoteLLM interface
  rag_pipeline.py   # orchestration, caching, latency tracking
  main.py           # FastAPI app
eval/
  ragas_lite.py     # faithfulness / relevance / correctness / hallucination
  benchmark_queries.json
  run_eval.py
data/documents/      # synthetic enterprise knowledge base
frontend/index.html  # single-file chat UI
tests/               # smoke + scale tests
```
