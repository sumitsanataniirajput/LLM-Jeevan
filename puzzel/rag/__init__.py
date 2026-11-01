"""RAG (Retrieval-Augmented Generation) package for Puzzel.

Provides document ingestion, chunking, retrieval, and citation support
for augmenting the language model with external knowledge.

No external AI APIs or pretrained models are used. All processing is
local and deterministic.
"""

from puzzel.rag.chunking import chunk_text, chunk_document
from puzzel.rag.citations import Citation, format_citations
from puzzel.rag.extraction import extract_text, extract_from_file
from puzzel.rag.retrieval import Retriever, RetrievalResult

__all__ = [
    "chunk_text",
    "chunk_document",
    "Citation",
    "format_citations",
    "extract_text",
    "extract_from_file",
    "Retriever",
    "RetrievalResult",
]