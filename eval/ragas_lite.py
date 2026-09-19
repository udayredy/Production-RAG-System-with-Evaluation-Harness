"""RAGAS-style evaluation metrics computed without an LLM judge.

Real RAGAS (faithfulness, answer relevance, context precision/recall,
answer correctness) uses an LLM to check entailment between answer and
context. Without LLM access, this reimplements the same *scoring logic*
using n-gram grounding + embedding similarity, which is what those metrics
approximate in practice:

- faithfulness: fraction of answer sentences that are (near-)substrings of
  the retrieved context, i.e. grounded rather than invented.
- answer_relevance: cosine similarity between the answer and the query, in
  the same embedding space used for retrieval.
- answer_correctness: token-level F1 between the generated answer and the
  gold answer (standard extractive-QA metric).
- hallucination_rate: 1 - faithfulness.
"""
import re

import numpy as np

from app.embeddings import embed


def _tokens(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def _split_sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def faithfulness(answer: str, context_chunks: list) -> float:
    context_text = " ".join(c["text"] for c in context_chunks).lower()
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0
    grounded = 0
    for sent in sentences:
        toks = _tokens(sent)
        if not toks:
            continue
        overlap = sum(1 for t in toks if t in context_text)
        if overlap / len(toks) >= 0.6:
            grounded += 1
    return grounded / len(sentences)


def answer_relevance(query: str, answer: str) -> float:
    vecs = embed([query, answer])
    q, a = vecs[0], vecs[1]
    denom = (np.linalg.norm(q) * np.linalg.norm(a))
    return float(np.dot(q, a) / denom) if denom > 0 else 0.0


def _f1(pred_tokens, gold_tokens):
    pred_set, gold_set = set(pred_tokens), set(gold_tokens)
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def answer_correctness(answer: str, gold_answer: str) -> float:
    return _f1(_tokens(answer), _tokens(gold_answer))


def hallucination_rate(answer: str, context_chunks: list) -> float:
    return 1.0 - faithfulness(answer, context_chunks)


def evaluate_single(query: str, answer: str, context_chunks: list, gold_answer: str) -> dict:
    return {
        "faithfulness": round(faithfulness(answer, context_chunks), 4),
        "answer_relevance": round(answer_relevance(query, answer), 4),
        "answer_correctness": round(answer_correctness(answer, gold_answer), 4),
        "hallucination_rate": round(hallucination_rate(answer, context_chunks), 4),
    }
