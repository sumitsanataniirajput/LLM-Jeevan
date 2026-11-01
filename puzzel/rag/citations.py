"""Source citations for RAG responses.

Provides citation tracking and formatting for retrieved document chunks
used in generation. Citations are purely metadata; no external services
are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Citation:
    """A reference to a source document chunk.

    Attributes:
        document_id: Unique identifier of the source document.
        chunk_index: Index of the chunk within the document.
        text: The chunk text (truncated for display).
        start_char: Character offset of the chunk in the source.
        end_char: Character offset of the chunk end in the source.
        score: Optional relevance score from retrieval.
        metadata: Optional additional metadata (source URL, title, etc.).
    """

    document_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def display_text(self, max_len: int = 120) -> str:
        """Return a truncated display version of the chunk text."""
        if len(self.text) <= max_len:
            return self.text
        return self.text[: max_len - 3] + "..."

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "score": self.score,
            "metadata": self.metadata,
        }


def format_citations(
    citations: list[Citation],
    style: str = "numbered",
) -> str:
    """Format a list of citations for display.

    Args:
        citations: List of Citation objects.
        style: Formatting style: "numbered", "bullet", or "inline".

    Returns:
        Formatted citation string.
    """
    if not citations:
        return ""

    if style == "numbered":
        lines = []
        for i, c in enumerate(citations, 1):
            source = c.metadata.get("source", c.document_id)
            lines.append(f"[{i}] {source} (chunk {c.chunk_index}): {c.display_text()}")
        return "\n".join(lines)
    if style == "bullet":
        lines = []
        for c in citations:
            source = c.metadata.get("source", c.document_id)
            lines.append(f"- {source} (chunk {c.chunk_index}): {c.display_text()}")
        return "\n".join(lines)
    if style == "inline":
        parts = []
        for c in citations:
            source = c.metadata.get("source", c.document_id)
            parts.append(f"[{source}:{c.chunk_index}]")
        return " ".join(parts)
    raise ValueError(f"unsupported citation style: {style!r}")