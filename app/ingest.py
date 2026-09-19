"""Ingestion pipeline: load documents -> chunk -> embed -> build FAISS + BM25 indexes.

Mirrors the "Azure AI Document Intelligence -> searchable knowledge pipeline"
step from the resume, scoped to local text files (see README).
"""
import glob
import json
import os
import pickle
import time

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.chunking import chunk_text
from app.embeddings import embed

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "documents")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")


def load_documents(data_dir=DATA_DIR):
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(path) as f:
            text = f.read()
        doc_id = os.path.splitext(os.path.basename(path))[0]
        docs.append({"doc_id": doc_id, "path": path, "text": text})
    return docs


def build_index(data_dir=DATA_DIR, index_dir=INDEX_DIR, chunk_size=500, overlap=100):
    t0 = time.time()
    os.makedirs(index_dir, exist_ok=True)
    docs = load_documents(data_dir)

    chunks = []  # list of dicts: doc_id, chunk_id, text, source_path
    for doc in docs:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            chunks.append({
                "doc_id": doc["doc_id"],
                "chunk_id": f"{doc['doc_id']}::{i}",
                "text": piece,
                "source_path": doc["path"],
            })

    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")

    # Dense index (FAISS, inner product over L2-normalized vectors == cosine sim)
    vecs = embed(texts)
    dim = vecs.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(vecs)
    faiss.write_index(faiss_index, os.path.join(index_dir, "faiss.index"))

    # Sparse index (BM25)
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    # Metadata
    with open(os.path.join(index_dir, "chunks.json"), "w") as f:
        json.dump(chunks, f)

    elapsed = time.time() - t0
    print(f"Built FAISS (dim={dim}) + BM25 indexes for {len(chunks)} chunks in {elapsed:.2f}s")
    return {"n_docs": len(docs), "n_chunks": len(chunks), "elapsed_s": elapsed}


if __name__ == "__main__":
    build_index()
