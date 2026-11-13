"""Deterministically recommend a chunking strategy without chunking a document.

This module is intentionally limited to strategy selection.  It does not create
chunks, embeddings, vector stores, retrievals, or LLM calls.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from typing import Any


STRATEGIES = {
    "fixed_size",
    "recursive",
    "semantic",
    "structure_based",
    "agentic",
}

TEXT_EXTENSIONS = {
    "txt",
    "md",
    "markdown",
    "html",
    "htm",
    "xml",
    "py",
    "js",
    "ts",
    "java",
    "go",
    "rs",
    "c",
    "cpp",
    "h",
    "hpp",
    "json",
    "yaml",
    "yml",
    "csv",
}
DOCUMENT_EXTENSIONS = TEXT_EXTENSIONS | {"pdf", "docx"}

_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S+", re.MULTILINE)
_CODE_DEFINITION = re.compile(
    r"^\s*(?:async\s+def|def|class|function|interface|module)\s+\w+", re.MULTILINE
)
_HTML_SECTION = re.compile(r"<\s*(?:article|section|main|h[1-6])\b", re.IGNORECASE)
_WORDS = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


class DocumentReadError(RuntimeError):
    """Raised when a supported document cannot be extracted as text."""


def read_document(path: str | Path) -> tuple[str, dict[str, Any]]:
    """Extract text and parser metadata from a real document on disk.

    Text-like files are decoded as UTF-8. PDF and DOCX extraction is optional
    at import time and uses the packages listed in the repository requirements.
    The function deliberately extracts text/metadata only; it never chunks or
    calls an LLM.
    """
    document_path = Path(path)
    try:
        payload = document_path.read_bytes()
    except OSError as error:
        raise DocumentReadError(f"Could not read {document_path}: {error}") from error
    return read_document_bytes(payload, document_path.name)


def read_document_bytes(payload: bytes, file_name: str) -> tuple[str, dict[str, Any]]:
    """Extract text and metadata from uploaded document bytes."""
    suffix = Path(file_name).suffix.lower().lstrip(".")
    if suffix == "pdf":
        return _read_pdf_bytes(payload, file_name)
    if suffix == "docx":
        return _read_docx_bytes(payload, file_name)
    if suffix not in TEXT_EXTENSIONS and suffix:
        raise DocumentReadError(
            f"Unsupported file type '.{suffix}'. Supported types: {', '.join(sorted(DOCUMENT_EXTENSIONS))}."
        )
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise DocumentReadError(f"{file_name} is not valid UTF-8 text") from error
    return text, _text_parser_metadata(text, suffix)


def _read_pdf_bytes(payload: bytes, file_name: str) -> tuple[str, dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise DocumentReadError("PDF support requires 'pypdf'. Install the project requirements first.") from error
    try:
        reader = PdfReader(io.BytesIO(payload))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as error:
        raise DocumentReadError(f"Could not extract text from {file_name}: {error}") from error
    text = "\n\n".join(page for page in pages if page.strip())
    metadata = _text_parser_metadata(text, "pdf")
    metadata["page_count"] = len(reader.pages)
    metadata["has_extractable_text"] = bool(text.strip())
    return text, metadata


def _read_docx_bytes(payload: bytes, file_name: str) -> tuple[str, dict[str, Any]]:
    try:
        import docx2txt
    except ImportError as error:
        raise DocumentReadError("DOCX support requires 'docx2txt'. Install the project requirements first.") from error
    try:
        # docx2txt expects a path; use a short-lived temporary file for uploads.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".docx") as temporary:
            temporary.write(payload)
            temporary.flush()
            text = docx2txt.process(temporary.name) or ""
    except Exception as error:
        raise DocumentReadError(f"Could not extract text from {file_name}: {error}") from error
    metadata = _text_parser_metadata(text, "docx")
    metadata["has_extractable_text"] = bool(text.strip())
    return text, metadata


def _text_parser_metadata(text: str, file_type: str) -> dict[str, Any]:
    """Create conservative metadata from extracted text without an LLM."""
    heading_count = len(_MARKDOWN_HEADING.findall(text))
    html_section_count = len(_HTML_SECTION.findall(text))
    nested_marker_count = len(re.findall(r"^\s{4,}[-*]|^\s*\d+(?:\.\d+){2,}", text, re.MULTILINE))
    reference_marker_count = len(
        re.findall(r"\b(?:see|refer to)\s+(?:section|appendix)|\[\[[^]]+\]\]", text, re.IGNORECASE)
    )
    boundaries = re.findall(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", text, re.MULTILINE)
    if file_type in {"html", "htm", "xml"}:
        boundaries.extend(re.findall(r"<\s*(?:article|section|main|h[1-6])\b", text, re.IGNORECASE))
    return {
        "file_type": file_type or "unknown",
        "heading_count": heading_count,
        "section_boundaries": boundaries,
        "has_structure": heading_count >= 2 or html_section_count >= 2,
        "html_section_count": html_section_count,
        "nested_marker_count": nested_marker_count,
        "reference_marker_count": reference_marker_count,
        # Conservative deterministic signal for fragmented, cross-referenced text.
        "irregular_content": nested_marker_count >= 3 and reference_marker_count >= 2,
    }


def select_chunking_strategy(
    *,
    file_name: str | None = None,
    file_type: str | None = None,
    content: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one explainable chunking recommendation for a document.

    ``metadata`` may include parser evidence such as ``headings`` (a list),
    ``heading_count``, ``has_structure``, ``complex_layout``,
    ``cross_references``, or ``requires_agentic``.  Metadata is treated as
    supplied parser evidence; no external parser or model is invoked here.
    """
    metadata = metadata or {}
    text = content or ""
    detected_type = _normalise_file_type(file_name, file_type)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    heading_count = _heading_count(text, metadata)
    code_definition_count = len(_CODE_DEFINITION.findall(text))
    html_section_count = len(_HTML_SECTION.findall(text))
    signals = {
        "file_type": detected_type or "unknown",
        "content_characters": len(text),
        "paragraph_count": len(paragraphs),
        "has_headings": heading_count >= 2,
        "heading_count": heading_count,
        "has_code_blocks": "```" in text,
        "code_definition_count": code_definition_count,
        "html_section_count": html_section_count,
        "topic_complexity": "unknown",
        "parser_structure_detected": bool(
            metadata.get("has_structure") or metadata.get("headings") or metadata.get("heading_count")
        ),
    }

    if _has_reliable_structure(
        detected_type, heading_count, code_definition_count, html_section_count, metadata
    ):
        signals["topic_complexity"] = "structure_preservable"
        return _result(
            "structure_based",
            "Reliable native structural boundaries were detected and should be preserved.",
            0.92,
            signals,
        )

    if _needs_agentic_recommendation(text, metadata):
        signals["topic_complexity"] = "highly_irregular"
        return _result(
            "agentic",
            "The content has irregular nested structure and interdependent references; this recommends, but does not run, agentic segmentation.",
            0.74,
            signals,
        )

    if _has_topic_shifts(paragraphs, metadata):
        signals["topic_complexity"] = "multiple_distinct_topics"
        return _result(
            "semantic",
            "Distinct topic shifts were detected without reliable native structure.",
            0.76,
            signals,
        )

    if _is_flat_text(text, paragraphs, metadata):
        signals["topic_complexity"] = "low"
        return _result(
            "fixed_size",
            "The input is effectively a flat text stream with no useful boundaries.",
            0.81,
            signals,
        )

    return _result(
        "recursive",
        "No stronger structural or complexity signal was found, so recursive splitting is the safe general-purpose recommendation.",
        0.70 if text else 0.45,
        signals,
    )


def _normalise_file_type(file_name: str | None, file_type: str | None) -> str:
    if file_type:
        return file_type.lower().lstrip(".")
    return Path(file_name).suffix.lower().lstrip(".") if file_name else ""


def _heading_count(text: str, metadata: dict[str, Any]) -> int:
    headings = metadata.get("headings", [])
    supplied_count = metadata.get("heading_count", 0)
    if isinstance(headings, (list, tuple, set)):
        supplied_count = max(supplied_count, len(headings))
    return max(int(supplied_count or 0), len(_MARKDOWN_HEADING.findall(text)))


def _has_reliable_structure(
    file_type: str,
    heading_count: int,
    code_definition_count: int,
    html_section_count: int,
    metadata: dict[str, Any],
) -> bool:
    section_boundaries = metadata.get("section_boundaries", [])
    boundary_count = (
        len(section_boundaries)
        if isinstance(section_boundaries, (list, tuple, set))
        else int(bool(section_boundaries))
    )
    if heading_count >= 2 and metadata.get("headings"):
        return True
    if boundary_count >= 2:
        return True
    if metadata.get("has_structure") and (heading_count >= 2 or boundary_count):
        return True
    if file_type in {"md", "markdown"} and heading_count >= 2:
        return True
    if file_type in {"html", "htm", "xml"} and html_section_count >= 2:
        return True
    return code_definition_count >= 2


def _needs_agentic_recommendation(text: str, metadata: dict[str, Any]) -> bool:
    if metadata.get("requires_agentic"):
        return True
    complex_layout = bool(metadata.get("complex_layout") or metadata.get("irregular_layout"))
    cross_references = bool(metadata.get("cross_references") or metadata.get("interdependent_sections"))
    if complex_layout and cross_references:
        return True
    nested_markers = len(re.findall(r"^\s{4,}[-*]|^\s*\d+(?:\.\d+){2,}", text, re.MULTILINE))
    reference_markers = len(re.findall(r"\b(?:see|refer to)\s+(?:section|appendix)|\[\[[^]]+\]\]", text, re.I))
    return nested_markers >= 3 and reference_markers >= 2 and bool(metadata.get("irregular_content"))


def _has_topic_shifts(paragraphs: list[str], metadata: dict[str, Any]) -> bool:
    if metadata.get("topic_complexity") in {"high", "multiple_topics"}:
        return True
    if len(paragraphs) < 3:
        return False
    word_sets = [set(word.lower() for word in _WORDS.findall(paragraph)) for paragraph in paragraphs]
    disjoint_transitions = 0
    for first, second in zip(word_sets, word_sets[1:]):
        if len(first) >= 5 and len(second) >= 5 and len(first & second) / len(first | second) < 0.08:
            disjoint_transitions += 1
    explicit_topic_cues = sum(
        bool(re.match(r"(?:topic|subject|theme)\s*:\s*", paragraph, re.I)) for paragraph in paragraphs
    )
    return explicit_topic_cues >= 2 or disjoint_transitions >= 2


def _is_flat_text(text: str, paragraphs: list[str], metadata: dict[str, Any]) -> bool:
    if metadata.get("flat_text"):
        return True
    normalised = text.strip()
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", normalised))
    return bool(normalised) and len(paragraphs) <= 1 and "\n" not in normalised and sentence_count <= 1


def _result(strategy: str, reason: str, confidence: float, signals: dict[str, Any]) -> dict[str, Any]:
    return {"strategy": strategy, "reason": reason, "confidence": confidence, "signals": signals}


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend a RAG chunking strategy without chunking.")
    parser.add_argument("file", nargs="?", help="Optional UTF-8 text file to inspect.")
    parser.add_argument("--content", help="Document content, used instead of reading FILE.")
    parser.add_argument("--file-type", help="Type such as markdown, html, or python.")
    parser.add_argument("--metadata", default="{}", help="Parser metadata as a JSON object.")
    args = parser.parse_args()

    try:
        metadata = json.loads(args.metadata)
    except json.JSONDecodeError as error:
        parser.error(f"--metadata must be valid JSON: {error.msg}")
    if not isinstance(metadata, dict):
        parser.error("--metadata must be a JSON object")

    content = args.content
    extracted_metadata: dict[str, Any] = {}
    if content is None and args.file:
        try:
            content, extracted_metadata = read_document(args.file)
        except DocumentReadError as error:
            parser.error(str(error))
    merged_metadata = {**extracted_metadata, **metadata}
    result = select_chunking_strategy(
        file_name=args.file, file_type=args.file_type, content=content, metadata=merged_metadata
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
