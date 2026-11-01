"""Document extraction utilities for RAG.

Extracts plain text from common file formats: TXT, Markdown, CSV, JSON.
PDF and DOCX extraction are supported when optional dependencies are
available; otherwise a clear error is raised.

No external AI services are used. All extraction is local and deterministic.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def extract_text(content: str, content_type: str = "text/plain") -> str:
    """Extract plain text from raw content.

    Args:
        content: Raw file content as a string.
        content_type: MIME type or format hint ("text/plain", "text/markdown",
            "text/csv", "application/json").

    Returns:
        Extracted plain text.

    Raises:
        ValueError: If the content type is not supported.
    """
    if content_type in ("text/plain", "text/x-markdown", "text/markdown"):
        return content
    if content_type == "text/csv":
        return _extract_csv_text(content)
    if content_type == "application/json":
        return _extract_json_text(content)
    raise ValueError(f"unsupported content type: {content_type!r}")


def extract_from_file(path: str | Path) -> str:
    """Extract plain text from a file.

    The file format is inferred from its extension:
    .txt, .md -> plain text; .csv -> CSV text extraction;
    .json -> JSON text extraction.

    PDF (.pdf) and DOCX (.docx) are supported only if optional
    dependencies (pypdf, python-docx) are installed.

    Args:
        path: Path to the file.

    Returns:
        Extracted plain text.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
        ImportError: If PDF/DOCX support is requested but dependencies
            are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    if not path.is_file():
        raise ValueError(f"not a file: {path}")

    suffix = path.suffix.lower()
    content = path.read_text(encoding="utf-8", errors="replace")

    if suffix in {".txt", ".text"}:
        return content
    if suffix in {".md", ".markdown"}:
        return content
    if suffix == ".csv":
        return _extract_csv_text(content)
    if suffix == ".json":
        return _extract_json_text(content)
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    raise ValueError(f"unsupported file format: {suffix}")


def _extract_csv_text(content: str) -> str:
    """Extract readable text from CSV content."""
    lines: list[str] = []
    reader = csv.reader(content.splitlines())
    for row in reader:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def _extract_json_text(content: str) -> str:
    """Extract readable text from JSON content."""
    data: Any = json.loads(content)
    return _json_to_text(data)


def _json_to_text(data: Any, indent: int = 0) -> str:
    """Convert JSON data to plain text recursively."""
    prefix = "  " * indent
    if isinstance(data, dict):
        parts = []
        for key, value in data.items():
            parts.append(f"{prefix}{key}: {_json_to_text(value, indent + 1)}")
        return "\n".join(parts)
    if isinstance(data, list):
        parts = []
        for i, item in enumerate(data):
            parts.append(f"{prefix}[{i}]: {_json_to_text(item, indent + 1)}")
        return "\n".join(parts)
    return str(data)


def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF file using pypdf if available."""
    try:
        import pypdf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PDF extraction requires the 'pypdf' package. "
            "Install it with: pip install pypdf"
        ) from exc

    reader = pypdf.PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


def _extract_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx if available."""
    try:
        from docx import Document  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "DOCX extraction requires the 'python-docx' package. "
            "Install it with: pip install python-docx"
        ) from exc

    doc = Document(str(path))
    parts: list[str] = []
    for para in doc.paragraphs:
        if para.text:
            parts.append(para.text)
    return "\n".join(parts)