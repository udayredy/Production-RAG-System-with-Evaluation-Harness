"""Embedding backend.

Uses spaCy's `en_core_web_md` pretrained word vectors (real 300-d GloVe-style
vectors trained on Common Crawl), mean-pooled across a chunk's tokens. This
stands in for an Azure OpenAI / sentence-transformers embedding call, which
this sandbox cannot reach (see README). The interface is deliberately the
same shape (`embed(texts) -> np.ndarray [n, dim]`) as a hosted embeddings API
so swapping the backend later is a one-class change.
"""
import numpy as np
import spacy

_NLP = None
DIM = 300


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_md", disable=["parser", "tagger", "ner", "lemmatizer"])
    return _NLP


def embed(texts):
    nlp = _get_nlp()
    vecs = np.zeros((len(texts), DIM), dtype="float32")
    for i, doc in enumerate(nlp.pipe(texts, batch_size=64)):
        v = doc.vector
        norm = np.linalg.norm(v)
        vecs[i] = v / norm if norm > 0 else v
    return vecs


def embed_one(text: str):
    return embed([text])[0]
