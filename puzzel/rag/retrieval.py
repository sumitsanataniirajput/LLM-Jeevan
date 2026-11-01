"""Retrieval utilities for RAG.

Provides keyword-based and hybrid retrieval over document chunks.
No external embedding models or vector databases are used; retrieval
is based on deterministic text matching (BM25-like keyword scoring).

All processing is local. No external AI APIs are used.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    """A single retrieval result.

    Attributes:
        document_id: Source document identifier.
        chunk_index: Chunk index within the document.
        text: The retrieved chunk text.
        score: Relevance score (higher = more relevant).
        metadata: Additional metadata about the source.
    """

    document_id: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace/punctuation tokenizer."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency for a corpus of token lists.

    Uses the standard BM25 IDF formula: log((N - n + 0.5) / (n + 0.5)) + 1.
    This is always non-negative and works correctly even with small corpora.
    """
    num_docs = len(documents)
    doc_freq: dict[str, int] = {}
    for doc in documents:
        seen = set(doc)
        for token in seen:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    return {
        token: math.log((num_docs - freq + 0.5) / (freq + 0.5)) + 1.0
        for token, freq in doc_freq.items()
    }


class Retriever:
    """Keyword-based retriever for document chunks.

    Uses a BM25-like scoring function over tokenized text. All indexing
    and retrieval happen in memory; no external services are used.

    Attributes:
        chunks: Indexed chunks as list of dicts with document_id,
            chunk_index, text, and optional metadata.
        k1: BM25 term saturation parameter.
        b: BM25 length normalization parameter.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[dict[str, Any]] = []
        self._tokenized_chunks: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._indexed: bool = False

    def add_chunk(
        self,
        document_id: str,
        chunk_index: int,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a single chunk to the index."""
        self._chunks.append(
            {
                "document_id": document_id,
                "chunk_index": chunk_index,
                "text": text,
                "metadata": metadata or {},
            }
        )
        self._indexed = False

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Add multiple chunks to the index."""
        for c in chunks:
            self.add_chunk(
                document_id=c["document_id"],
                chunk_index=c["chunk_index"],
                text=c["text"],
                metadata=c.get("metadata"),
            )

    def _build_index(self) -> None:
        """Build the inverted index and IDF table."""
        self._tokenized_chunks = [_tokenize(c["text"]) for c in self._chunks]
        self._idf = _compute_idf(self._tokenized_chunks)
        total_len = sum(len(tokens) for tokens in self._tokenized_chunks)
        self._avgdl = total_len / len(self._tokenized_chunks) if self._tokenized_chunks else 0.0
        self._indexed = True

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant chunks for a query.

        Args:
            query: Search query text.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievalResult objects sorted by score (descending).
        """
        if not self._indexed:
            self._build_index()
        if not self._chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: list[float] = []
        for i, doc_tokens in enumerate(self._tokenized_chunks):
            score = 0.0
            doc_len = len(doc_tokens)
            for qt in query_tokens:
                if qt not in self._idf:
                    continue
                tf = doc_tokens.count(qt)
                if tf == 0:
                    continue
                idf = self._idf[qt]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
                score += idf * (tf * (self.k1 + 1)) / denom
            scores.append(score)

        # Rank and return top-k.
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[RetrievalResult] = []
        for idx in ranked[:top_k]:
            if scores[idx] <= 0:
                continue
            c = self._chunks[idx]
            results.append(
                RetrievalResult(
                    document_id=c["document_id"],
                    chunk_index=c["chunk_index"],
                    text=c["text"],
                    score=scores[idx],
                    metadata=c["metadata"],
                )
            )
        return results

    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 5,
        keyword_weight: float = 0.7,
    ) -> list[RetrievalResult]:
        """Hybrid retrieval combining keyword scoring with simple
        semantic overlap (token overlap ratio).

        Args:
            query: Search query text.
            top_k: Maximum number of results.
            keyword_weight: Weight for keyword score (1-keyword_weight
                is weight for overlap score).

        Returns:
            List of RetrievalResult objects sorted by combined score.
        """
        if not self._indexed:
            self._build_index()
        if not self._chunks:
            return []

        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        keyword_results = self.retrieve(query, top_k=len(self._chunks))
        keyword_scores = {r.document_id + ":" + str(r.chunk_index): r.score for r in keyword_results}

        combined: list[tuple[str, float, dict[str, Any]]] = []
        for i, c in enumerate(self._chunks):
            chunk_tokens = set(_tokenize(c["text"]))
            overlap = len(query_tokens & chunk_tokens) / len(query_tokens | chunk_tokens) if query_tokens | chunk_tokens else 0.0
            kw_score = keyword_scores.get(c["document_id"] + ":" + str(c["chunk_index"]), 0.0)
            combined_score = keyword_weight * kw_score + (1 - keyword_weight) * overlap
            combined.append(
                (c["document_id"] + ":" + str(c["chunk_index"]), combined_score, c)
            )

        combined.sort(key=lambda x: x[1], reverse=True)
        results: list[RetrievalResult] = []
        for key, score, c in combined[:top_k]:
            if score <= 0:
                continue
            results.append(
                RetrievalResult(
                    document_id=c["document_id"],
                    chunk_index=c["chunk_index"],
                    text=c["text"],
                    score=score,
                    metadata=c["metadata"],
                )
            )
        return results