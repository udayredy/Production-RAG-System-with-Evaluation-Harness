"""Hybrid dense + BM25 retrieval with reciprocal rank fusion (RRF)."""
import json
import os
import pickle

import faiss

from app.embeddings import embed_one

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")


class HybridRetriever:
    def __init__(self, index_dir=INDEX_DIR):
        self.index_dir = index_dir
        self.faiss_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        with open(os.path.join(index_dir, "chunks.json")) as f:
            self.chunks = json.load(f)

    def dense_search(self, query: str, k: int = 20):
        qvec = embed_one(query).reshape(1, -1)
        scores, idxs = self.faiss_index.search(qvec, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

    def bm25_search(self, query: str, k: int = 20):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in ranked]

    def hybrid_search(self, query: str, k: int = 20, rrf_k: int = 60):
        """Reciprocal Rank Fusion of dense + BM25 rankings."""
        dense = self.dense_search(query, k=k)
        sparse = self.bm25_search(query, k=k)

        fused = {}
        for rank, (idx, score) in enumerate(dense):
            fused.setdefault(idx, {"dense_rank": None, "sparse_rank": None,
                                    "dense_score": 0.0, "bm25_score": 0.0})
            fused[idx]["dense_rank"] = rank
            fused[idx]["dense_score"] = score
        for rank, (idx, score) in enumerate(sparse):
            fused.setdefault(idx, {"dense_rank": None, "sparse_rank": None,
                                    "dense_score": 0.0, "bm25_score": 0.0})
            fused[idx]["sparse_rank"] = rank
            fused[idx]["bm25_score"] = score

        results = []
        for idx, info in fused.items():
            rrf = 0.0
            if info["dense_rank"] is not None:
                rrf += 1.0 / (rrf_k + info["dense_rank"] + 1)
            if info["sparse_rank"] is not None:
                rrf += 1.0 / (rrf_k + info["sparse_rank"] + 1)
            chunk = self.chunks[idx]
            results.append({
                **chunk,
                "rrf_score": rrf,
                "dense_score": info["dense_score"],
                "bm25_score": info["bm25_score"],
            })
        results.sort(key=lambda r: r["rrf_score"], reverse=True)
        return results[:k]
