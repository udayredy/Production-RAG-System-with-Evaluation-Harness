"""Simple, dependency-free document chunker."""
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    source_path: str


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Word-based sliding-window chunking (chunk_size/overlap in characters)."""
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks
