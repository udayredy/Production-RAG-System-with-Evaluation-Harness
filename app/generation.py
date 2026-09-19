"""Answer generation stage.

`LLMClient` is the provider-agnostic interface. `LocalExtractiveLLM` is what
actually runs in this sandbox (no outbound LLM API access). `RemoteLLM` shows
exactly where a real Azure OpenAI / OpenAI call would go — set
USE_REMOTE_LLM=true and OPENAI_API_KEY (or AZURE_OPENAI_* vars) and the
pipeline switches over with no other code changes.
"""
import os
import re
from abc import ABC, abstractmethod

import numpy as np

from app.embeddings import embed, embed_one


class LLMClient(ABC):
    @abstractmethod
    def generate(self, query: str, context_chunks: list) -> str:
        ...


class LocalExtractiveLLM(LLMClient):
    """Extractive + template-based synthesis: never hallucinates, because it
    only ever emits sentences that are literally present in retrieved context,
    ranked by embedding similarity to the query."""

    def _split_sentences(self, text: str):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def generate(self, query: str, context_chunks: list, max_sentences: int = 3) -> str:
        if not context_chunks:
            return "I don't have enough information in the knowledge base to answer that."

        qvec = embed_one(query)
        candidates = []
        seen = set()
        for chunk in context_chunks:
            for sent in self._split_sentences(chunk["text"]):
                sent = sent.strip()
                # Skip section-title fragments (they restate the question,
                # e.g. "HR Policy Note 006: How many weeks...?"), dedupe, and
                # skip fragments truncated at a char-window chunk boundary
                # (no terminal punctuation -> not a complete sentence).
                if (
                    len(sent.split()) < 4
                    or sent.endswith("?")
                    or not sent.endswith((".", "!"))
                    or sent in seen
                    or any(sent in s or s in sent for s in seen)
                ):
                    continue
                seen.add(sent)
                candidates.append((sent, chunk["doc_id"]))

        if not candidates:
            return "I don't have enough information in the knowledge base to answer that."

        sents = [c[0] for c in candidates]
        svecs = embed(sents)
        sims = svecs @ qvec
        ranked_idx = np.argsort(-sims)[:max_sentences]

        chosen = [(sents[i], candidates[i][1], float(sims[i])) for i in ranked_idx]
        answer = " ".join(s for s, _, _ in chosen)
        sources = sorted(set(doc for _, doc, _ in chosen))
        return answer + f"\n\n[Sources: {', '.join(sources)}]"


class RemoteLLM(LLMClient):
    """Stub showing where a real hosted LLM call would go. Not used unless
    USE_REMOTE_LLM=true and credentials are present; this sandbox has no
    outbound access to OpenAI/Azure so this path is untested here."""

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("RemoteLLM requires OPENAI_API_KEY or AZURE_OPENAI_API_KEY")

    def generate(self, query: str, context_chunks: list) -> str:
        # import openai  # left uninstalled/uncalled in this sandbox
        context = "\n\n".join(c["text"] for c in context_chunks)
        prompt = (
            "Answer the question using ONLY the context below. Cite sources.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        raise NotImplementedError(
            "Wire this up to your provider's chat completion endpoint, e.g. "
            "openai.chat.completions.create(model=..., messages=[{'role':'user','content':prompt}])"
        )


def get_llm_client() -> LLMClient:
    if os.environ.get("USE_REMOTE_LLM", "false").lower() == "true":
        return RemoteLLM()
    return LocalExtractiveLLM()
