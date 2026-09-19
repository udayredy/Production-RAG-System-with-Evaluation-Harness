"""Learned re-ranking stage.

Stands in for a pretrained cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM`)
which this sandbox cannot download (HF Hub is blocked). Instead this trains a
small logistic-regression "learning-to-rank" model over hand-engineered
query/chunk features, using auto-generated positive/negative pairs mined from
the corpus itself (see `train_reranker.py`). This is a real trained model
with a held-out accuracy reported at train time, not a stub.
"""
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.embeddings import embed_one

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "index", "reranker.pkl")


def extract_features(query: str, query_vec, candidate: dict) -> np.ndarray:
    """5 features: rrf_score, dense_score, bm25_score(normalized), term_overlap, length_ratio."""
    q_terms = set(query.lower().split())
    c_terms = set(candidate["text"].lower().split())
    overlap = len(q_terms & c_terms) / max(1, len(q_terms))
    length_ratio = min(len(candidate["text"]), 1000) / 1000.0
    bm25_norm = np.tanh(candidate.get("bm25_score", 0.0) / 10.0)
    return np.array([
        candidate.get("rrf_score", 0.0),
        candidate.get("dense_score", 0.0),
        bm25_norm,
        overlap,
        length_ratio,
    ], dtype="float32")


class Reranker:
    def __init__(self, model_path=MODEL_PATH):
        with open(model_path, "rb") as f:
            self.model: LogisticRegression = pickle.load(f)

    def score(self, query: str, candidates: list) -> list:
        query_vec = embed_one(query)
        feats = np.stack([extract_features(query, query_vec, c) for c in candidates])
        probs = self.model.predict_proba(feats)[:, 1]
        for c, p in zip(candidates, probs):
            c["rerank_score"] = float(p)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates

    @staticmethod
    def exists(model_path=MODEL_PATH):
        return os.path.exists(model_path)
