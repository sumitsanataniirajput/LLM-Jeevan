"""Document chunking for RAG.

Splits text into overlapping chunks suitable for retrieval and
embedding. All splitting is deterministic and local.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextChunk:
    """A single chunk of text with metadata.

    Attributes:
        text: The chunk text.
        start_char: Character offset of the chunk start in the source.
        end_char: Character offset of the chunk end in the source.
        chunk_index: Sequential index of this chunk within the document.
    """

    text: str
    start_char: int
    end_char: int
    chunk_index: int


def chunk_text(
    text: str,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
    separator: str = "\n",
) -> list[TextChunk]:
    """Split text into overlapping chunks.

    Chunks are formed by splitting on ``separator`` and greedily filling
    each chunk up to ``chunk_size`` characters, with ``overlap`` characters
    carried over to the next chunk.

    Args:
        text: Source text to chunk.
        chunk_size: Maximum characters per chunk (must be > overlap).
        overlap: Number of characters to overlap between chunks.
        separator: Separator used for splitting (e.g., "\\n", " ", "").

    Returns:
        List of TextChunk objects in order.

    Raises:
        ValueError: If chunk_size <= overlap or parameters are invalid.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(overlap, int) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    if not text:
        return []

    parts = text.split(separator)
    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_size = 0
    start_char = 0
    chunk_index = 0

    for part in parts:
        part_with_sep = part + separator
        part_size = len(part_with_sep)

        if current_size + part_size > chunk_size and current_parts:
            # Emit current chunk.
            chunk_text_str = separator.join(current_parts)
            chunks.append(
                TextChunk(
                    text=chunk_text_str,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text_str),
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

            # Carry overlap: keep the last part(s) that fit within overlap.
            carry_parts: list[str] = []
            carry_size = 0
            for p in reversed(current_parts):
                p_with_sep = p + separator
                if carry_size + len(p_with_sep) <= overlap:
                    carry_parts.insert(0, p)
                    carry_size += len(p_with_sep)
                else:
                    break
            current_parts = carry_parts
            current_size = carry_size
            start_char = start_char + len(chunk_text_str) - carry_size

        current_parts.append(part)
        current_size += part_size

    # Emit final chunk.
    if current_parts:
        chunk_text_str = separator.join(current_parts)
        chunks.append(
            TextChunk(
                text=chunk_text_str,
                start_char=start_char,
                end_char=start_char + len(chunk_text_str),
                chunk_index=chunk_index,
            )
        )

    return chunks


def chunk_document(
    document_id: str,
    text: str,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
    separator: str = "\n",
) -> list[dict[str, Any]]:
    """Chunk a document and return dicts suitable for storage/indexing.

    Args:
        document_id: Unique identifier for the source document.
        text: Source text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between chunks.
        separator: Separator used for splitting.

    Returns:
        List of dicts with keys: document_id, chunk_index, text,
        start_char, end_char.
    """
    from typing import Any

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap, separator=separator)
    return [
        {
            "document_id": document_id,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "start_char": c.start_char,
            "end_char": c.end_char,
        }
        for c in chunks
    ]