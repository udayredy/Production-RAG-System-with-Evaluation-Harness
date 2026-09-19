"""Orchestrates retrieval -> re-ranking -> generation, with response caching
and latency tracking (mirrors the "response caching, ~1.2s median latency"
line from the resume)."""
import hashlib
import time
from functools import lru_cache

from app.generation import get_llm_client
from app.reranker import Reranker
from app.retrieval import HybridRetriever

_retriever = None
_reranker = None
_llm = None
_cache = {}


def _lazy_init():
    global _retriever, _reranker, _llm
    if _retriever is None:
        _retriever = HybridRetriever()
    if _reranker is None and Reranker.exists():
        _reranker = Reranker()
    if _llm is None:
        _llm = get_llm_client()


def _cache_key(query: str, k: int) -> str:
    return hashlib.sha256(f"{query}|{k}".encode()).hexdigest()


def answer_query(query: str, k: int = 5, use_cache: bool = True):
    _lazy_init()
    t0 = time.time()

    key = _cache_key(query, k)
    if use_cache and key in _cache:
        result = dict(_cache[key])
        result["cached"] = True
        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return result

    candidates = _retriever.hybrid_search(query, k=max(k, 20))
    if _reranker is not None:
        candidates = _reranker.score(query, candidates)
    top = candidates[:k]

    answer = _llm.generate(query, top)

    result = {
        "query": query,
        "answer": answer,
        "retrieved": [
            {
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "rrf_score": round(c.get("rrf_score", 0.0), 4),
                "rerank_score": round(c.get("rerank_score", 0.0), 4) if "rerank_score" in c else None,
            }
            for c in top
        ],
        "cached": False,
    }
    _cache[key] = result
    result = dict(result)
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return result
