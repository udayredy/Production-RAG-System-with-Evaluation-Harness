"""Trains the learned re-ranker on auto-mined positive/negative pairs.

Positive pairs: (a gold question, a chunk from the document whose title
answers that question). Negative pairs: (that same question, a random
chunk from an unrelated document). This produces a supervised
learning-to-rank training set without needing any external labeled data
or LLM judge.
"""
import json
import os
import pickle
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from app.reranker import extract_features, MODEL_PATH
from app.retrieval import HybridRetriever

random.seed(11)

GOLD_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gold_qa.json")


def build_training_set(retriever: HybridRetriever, gold):
    X, y = [], []
    all_chunk_idxs = list(range(len(retriever.chunks)))
    for item in gold:
        query = item["question"]
        domain = item["domain"]
        query_vec = None  # computed inside extract_features via embeddings module

        candidates = retriever.hybrid_search(query, k=20)
        for c in candidates:
            label = 1 if c["doc_id"].startswith(domain) and item["gold_answer"][:30].lower() in c["text"].lower() else 0
            feats = extract_features(query, query_vec, c)
            X.append(feats)
            y.append(label)

        # also add a few explicit hard negatives from other domains
        other_domain_chunks = [c for c in retriever.chunks if not c["doc_id"].startswith(domain)]
        for neg_chunk in random.sample(other_domain_chunks, k=min(5, len(other_domain_chunks))):
            feats = extract_features(query, query_vec, {**neg_chunk, "rrf_score": 0.0, "dense_score": 0.0, "bm25_score": 0.0})
            X.append(feats)
            y.append(0)
    return np.array(X), np.array(y)


def main():
    retriever = HybridRetriever()
    with open(GOLD_PATH) as f:
        gold = json.load(f)

    X, y = build_training_set(retriever, gold)
    print(f"Training set: {X.shape[0]} examples, positive rate={y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = float("nan")
    print(f"Held-out accuracy: {acc:.3f}  AUC: {auc:.3f}")
    print(f"Learned feature weights: {dict(zip(['rrf','dense','bm25','overlap','len_ratio'], model.coef_[0].round(3)))}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved reranker to {MODEL_PATH}")


if __name__ == "__main__":
    main()
