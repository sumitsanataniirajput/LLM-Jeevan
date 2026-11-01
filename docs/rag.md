# Puzzel RAG Documentation

Puzzel includes a Retrieval-Augmented Generation (RAG) system for
augmenting the language model with external knowledge.

## Overview

The RAG pipeline consists of:

1. **Extraction** — Read text from TXT, Markdown, CSV, JSON, PDF, DOCX files
2. **Chunking** — Split documents into overlapping chunks
3. **Retrieval** — Keyword-based (BM25-like) search over chunks
4. **Citations** — Track and format source references

No external embedding models or vector databases are used. All processing
is local and deterministic.

## Extraction

```python
from puzzel.rag import extract_text, extract_from_file

# From raw content
text = extract_text("Hello # World", content_type="text/markdown")

# From file
text = extract_from_file("data/raw/en/chunk_000.txt")
```

Supported formats: `.txt`, `.md`, `.csv`, `.json`, `.pdf` (requires pypdf),
`.docx` (requires python-docx).

## Chunking

```python
from puzzel.rag import chunk_text, chunk_document

chunks = chunk_text(
    document_text,
    chunk_size=512,
    overlap=64,
    separator="\n"
)

# For storage/indexing
chunk_dicts = chunk_document(
    document_id="doc1",
    text=document_text,
    chunk_size=512,
    overlap=64
)
```

## Retrieval

```python
from puzzel.rag import Retriever

retriever = Retriever()
retriever.add_chunk("doc1", 0, "The quick brown fox", {"source": "test.txt"})
retriever.add_chunk("doc1", 1, "Python programming", {"source": "test.py"})

results = retriever.retrieve("python programming", top_k=5)
for r in results:
    print(f"{r.document_id}:{r.chunk_index} score={r.score:.3f}")
```

Hybrid retrieval (keyword + overlap):

```python
results = retriever.hybrid_retrieve("python", top_k=5, keyword_weight=0.7)
```

## Citations

```python
from puzzel.rag import Citation, format_citations

citation = Citation(
    document_id="doc1",
    chunk_index=0,
    text="The quick brown fox...",
    start_char=0,
    end_char=20,
    score=0.95,
    metadata={"source": "test.txt"}
)

print(format_citations([citation], style="numbered"))
```

## Integration with Generation

RAG retrieval results can be used as context for model generation:

```python
from puzzel.rag import Retriever
from puzzel.inference import InferenceEngine

# Retrieve relevant chunks
results = retriever.retrieve(user_query, top_k=3)
context = "\n".join(r.text for r in results)

# Generate with context
result = engine.generate(f"Context: {context}\n\nQuestion: {user_query}")
```

## Limitations

- Keyword-based retrieval (no semantic embeddings).
- All data held in memory; no persistent vector store.
- Suitable for small document collections only.