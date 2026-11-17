"""File ingestion and retrieval for the local FAISS knowledge base."""
from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
# `tools.py` imports this module before `agent.py`. Load the repository .env
# here so the embedding configuration is available during application startup.
load_dotenv(BASE_DIR.parent / ".env")
VECTOR_STORE_PATH = BASE_DIR / "vector_store"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".py", ".csv"}

class LocalHashEmbeddings(Embeddings):
    """Small dependency-free embeddings for local FAISS indexing.

    Groq provides chat/completions, not an embeddings endpoint. Hashing word
    and character n-grams keeps RAG local and avoids requiring another hosted
    provider or a multi-hundred-megabyte model download.
    """

    dimensions = 384

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        normalized = " ".join(text.lower().split())
        tokens = normalized.split()
        features = tokens + [normalized[index:index + 3] for index in range(max(0, len(normalized) - 2))]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


embeddings = LocalHashEmbeddings()
logger = logging.getLogger("uvicorn.error.rag")

SearchMode = Literal["vector", "keyword", "hybrid"]
_TOKEN_PATTERN = re.compile(r"[\w]+(?:[._:/-][\w]+)*", re.UNICODE)
_EXACT_IDENTIFIER_PATTERN = re.compile(
    r"(?:\b(?:err(?:or)?|http|status|code|exception|errno)[-_:/ ]?[a-z]?\d{2,}\b"
    r"|\b[a-z]{1,32}[-_][a-z0-9][a-z0-9_.-]{2,}\b"
    r"|\b0x[0-9a-f]+\b)",
    re.IGNORECASE,
)
_ERROR_CODE_PATTERN = re.compile(
    r"\b(?:err(?:or)?|http)[-_:/ ]?[a-z0-9][a-z0-9_.-]*\b", re.IGNORECASE
)
_TECHNICAL_MARKERS = (
    "tài liệu kỹ thuật", "tai lieu ky thuat", "mã lỗi", "ma loi", "error code",
    "stack trace", "exception", "api", "sdk", "cli", "config", "configuration",
    "thuật ngữ", "thuat ngu", "technical documentation", "documentation", "endpoint",
    "function", "class", "parameter", "schema", "regex", "log lỗi", "log loi",
)


def _loader_for(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path))
    if suffix == ".docx":
        return Docx2txtLoader(str(path))
    if suffix == ".csv":
        return CSVLoader(str(path), encoding="utf-8")
    return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)


_METADATA_FIELDS = ("category", "date", "author", "department")
_METADATA_LINE = re.compile(r"^\s*(category|date|author|department)\s*:\s*(.*?)\s*$", re.I | re.M)


def _document_metadata(text: str) -> dict[str, str]:
    match = re.match(r"\A\s*---\s*\n(.*?)\n---(?:\s*\n|\s*$)", text, re.S)
    if not match:
        return {}
    return {key.lower(): value.strip().strip("\"'") for key, value in _METADATA_LINE.findall(match.group(1)) if value.strip()}


def _chunk_documents(documents: list[Document], suffix: str) -> tuple[str, list[Document]]:
    text = "\n\n".join(doc.page_content for doc in documents)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    if suffix == ".csv":
        return "row", splitter.split_documents(documents)
    boundary = None
    if suffix == ".md" and len(re.findall(r"(?m)^#{1,6}\s+", text)) >= 2:
        boundary = r"(?=^#{1,6}\s+)"
    elif suffix == ".py" and len(re.findall(r"(?m)^(?:async\s+def|def|class)\s+", text)) >= 2:
        boundary = r"(?=^(?:async\s+def|def|class)\s+)"
    if boundary:
        sections = [Document(page_content=part, metadata=dict(doc.metadata)) for doc in documents
                    for part in re.split(boundary, doc.page_content, flags=re.M) if part.strip()]
        return "structure_based", splitter.split_documents(sections)
    if "\n" not in text and not re.search(r"[.!?]\s", text):
        return "fixed_size", RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators=[""]).split_documents(documents)
    return "recursive", splitter.split_documents(documents)


def read_file_text(file_path: str | Path, source_name: str | None = None) -> int:
    """Read an allowed upload and add its text to FAISS. Returns chunk count."""
    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    documents: list[Document] = [doc for doc in _loader_for(path).load() if doc.page_content.strip()]
    if not documents:
        raise ValueError("No extractable text was found in this file.")
    common_metadata = _document_metadata(documents[0].page_content)
    for doc in documents:
        doc.metadata.update(common_metadata)
        if path.suffix.lower() == ".csv":
            doc.metadata.update({key.lower(): value.strip() for key, value in _METADATA_LINE.findall(doc.page_content) if value.strip()})
        doc.metadata["source"] = source_name or path.name
    strategy, chunks = _chunk_documents(documents, path.suffix.lower())
    for index, chunk in enumerate(chunks):
        chunk.metadata.update(chunk_strategy=strategy, chunk_index=index)
    logger.info("RAG chunking source=%s strategy=%s metadata=%s chunks=%d", source_name or path.name, strategy, common_metadata, len(chunks))
    if not chunks:
        raise ValueError("The file did not produce any searchable text chunks.")

    index_file = VECTOR_STORE_PATH / "index.faiss"
    if index_file.exists():
        store = FAISS.load_local(str(VECTOR_STORE_PATH), embeddings, allow_dangerous_deserialization=True)
        expected_dimension = len(embeddings.embed_query("dimension check"))
        if store.index.d != expected_dimension:
            # Rebuild indexes created with a previous embedding provider
            # (for example OpenAI's text-embedding-3-small).
            store = FAISS.from_documents(chunks, embeddings)
        else:
            store.add_documents(chunks)
    else:
        store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(VECTOR_STORE_PATH))
    logger.info("RAG index updated source=%s chunks=%d", source_name or path.name, len(chunks))
    return len(chunks)


def select_search_mode(query: str) -> SearchMode:
    """Choose retrieval strategy from the query without exposing it to the UI.

    General questions benefit from semantic matching. Error codes and broader
    technical/document queries combine lexical and semantic signals, while
    other exact identifiers can use lexical matching alone.
    """
    normalized = " ".join(query.lower().split())
    if _ERROR_CODE_PATTERN.search(normalized) or any(marker in normalized for marker in _TECHNICAL_MARKERS):
        return "hybrid"
    if _EXACT_IDENTIFIER_PATTERN.search(normalized):
        return "keyword"
    return "vector"


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def _store_documents(store: FAISS) -> list[Document]:
    # FAISS's in-memory docstore is persisted alongside index.faiss by
    # ``save_local``. Keep only valid, non-empty documents and preserve order.
    documents = getattr(getattr(store, "docstore", None), "_dict", {})
    return [doc for doc in documents.values() if isinstance(doc, Document) and doc.page_content.strip()]


def _keyword_search(documents: list[Document], query: str, k: int = 4) -> list[Document]:
    """Return lexical matches using a small BM25-style scorer."""
    query_terms = _tokens(query)
    if not query_terms:
        return []
    tokenized = [_tokens(doc.page_content) for doc in documents]
    document_frequency = Counter(term for terms in tokenized for term in set(terms))
    average_length = sum(map(len, tokenized)) / max(len(tokenized), 1)
    query_frequency = Counter(query_terms)
    scored: list[tuple[float, int, Document]] = []
    total_documents = len(documents)
    for index, (doc, terms) in enumerate(zip(documents, tokenized)):
        if not terms:
            continue
        frequencies = Counter(terms)
        length_norm = len(terms) / max(average_length, 1.0)
        score = 0.0
        for term, query_count in query_frequency.items():
            term_frequency = frequencies.get(term, 0)
            if not term_frequency:
                continue
            idf = math.log(1 + (total_documents - document_frequency.get(term, 0) + 0.5) /
                           (document_frequency.get(term, 0) + 0.5))
            score += idf * ((term_frequency * 2.2) / (term_frequency + 1.2 * (0.8 + 0.2 * length_norm))) * query_count
        if score:
            scored.append((score, index, doc))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc for _, _, doc in scored[:k]]


_FILTER_LABELS = {
    "department": r"(?:phòng(?:\s+ban)?|bộ\s+phận|department)",
    "author": r"(?:tác\s+giả|author)",
    "category": r"(?:category|loại\s+tài\s+liệu)",
}


def validate_metadata_filters(query: str, proposed: dict[str, str], local: dict[str, str]) -> dict[str, str]:
    """Accept model filters only when their field and value are explicit in the query."""
    validated = {}
    for field, value in proposed.items():
        if field == "date":
            if local.get(field) == value:
                validated[field] = value
        elif field in _FILTER_LABELS and value.strip():
            pattern = rf"(?<!\w){_FILTER_LABELS[field]}\s+(?:của\s+)?{re.escape(value.strip())}(?!\w)"
            if re.search(pattern, query, re.IGNORECASE):
                validated[field] = value.strip()
    validated.update(local)
    return validated


def extract_local_metadata_filters(query: str) -> dict[str, str]:
    """Recognize explicit metadata labels using values present in the index."""
    if not (VECTOR_STORE_PATH / "index.faiss").exists():
        return {}
    store = FAISS.load_local(str(VECTOR_STORE_PATH), embeddings, allow_dangerous_deserialization=True)
    documents = _store_documents(store)
    filters = {}
    for field, label in _FILTER_LABELS.items():
        values = {str(doc.metadata[field]).strip() for doc in documents if doc.metadata.get(field)}
        for value in sorted(values, key=len, reverse=True):
            pattern = rf"(?<!\w){label}\s+(?:của\s+)?{re.escape(value)}(?!\w)"
            if re.search(pattern, query, re.IGNORECASE):
                filters[field] = value
                break
    # An explicitly named unknown department must not silently become an
    # unfiltered search; missing metadata remains eligible as usual.
    if "department" not in filters:
        match = re.search(r"(?i)(?<!\w)(?:phòng(?:\s+ban)?|bộ\s+phận|department)\s+(?:của\s+)?([\w-]+)", query)
        if match and match.group(1).casefold() not in {"ban", "nào", "này", "đó", "khác", "theo", "của"}:
            filters["department"] = match.group(1)
    match = re.search(r"(?i)(?:ngày|date)\s+(\d{1,4})[-/](\d{1,2})[-/](\d{1,4})", query)
    if match:
        first, month, last = (int(part) for part in match.groups())
        year, day = (first, last) if len(match.group(1)) == 4 else (last, first)
        try:
            filters["date"] = date(year, month, day).isoformat()
        except ValueError:
            logger.warning("RAG metadata extraction invalid_date=%r", match.group(0))
    return filters


def _matches_metadata(doc: Document, filters: dict[str, str]) -> bool:
    """Keep unknown metadata eligible; exclude explicit mismatches only."""
    for key, value in filters.items():
        actual = doc.metadata.get(key)
        if actual is not None and str(actual).strip():
            if str(actual).strip().casefold() != value.strip().casefold():
                return False
    return True


def _vector_search(store: FAISS, query: str, k: int = 4, filters: dict[str, str] | None = None) -> list[Document]:
    filters = filters or {}
    fetch_k = store.index.ntotal if filters else k
    return store.similarity_search(query, k=k, fetch_k=fetch_k,
                                   filter=lambda metadata: _matches_metadata(Document(page_content="", metadata=metadata), filters))


def _hybrid_search(store: FAISS, query: str, k: int = 4, filters: dict[str, str] | None = None) -> list[Document]:
    filters = filters or {}
    eligible = [doc for doc in _store_documents(store) if _matches_metadata(doc, filters)]
    rankings = (_vector_search(store, query, k=max(k, 8), filters=filters), _keyword_search(eligible, query, k=max(k, 8)))
    fused: dict[str, tuple[float, Document]] = {}
    for results in rankings:
        for rank, doc in enumerate(results, 1):
            identifier = f"{doc.metadata.get('source', '')}\n{doc.page_content}"
            previous = fused.get(identifier)
            fused[identifier] = ((previous[0] if previous else 0.0) + 1.0 / (60 + rank), doc)
    return [item[1] for item in sorted(fused.values(), key=lambda item: -item[0])[:k]]


def _translate_query(query: str) -> str:
    """Translate Vietnamese to English offline when the model is installed."""
    try:
        from langdetect import DetectorFactory, LangDetectException, detect
        import argostranslate.translate

        DetectorFactory.seed = 0
        try:
            language = detect(query)
        except LangDetectException:
            language = "unknown"
        if language != "vi":
            logger.info("RAG translation status=skipped detected_language=%s", language)
            return ""
        languages = argostranslate.translate.get_installed_languages()
        source = next((item for item in languages if item.code == "vi"), None)
        target = next((item for item in languages if item.code == "en"), None)
        if not source or not target:
            logger.warning("RAG translation status=unavailable reason=vi_en_model_missing")
            return ""
        translation = source.get_translation(target)
        if translation is None:
            logger.warning("RAG translation status=unavailable reason=vi_en_model_missing")
            return ""
        result = translation.translate(query).strip()
        logger.info("RAG translation status=translated provider=argos from=vi to=en")
        return result
    except Exception:
        logger.exception("RAG translation status=failed; searching original query")
        return ""


def _rewrite_query(query: str) -> str:
    original = " ".join(query.split())
    translated = _translate_query(original) if original else ""
    return " ".join(part for part in (original, translated) if part)


def _rerank(documents: list[Document], query: str) -> list[Document]:
    terms = set(_tokens(query))
    phrases = [phrase.casefold() for phrase in re.findall(r'"([^"]+)"', query)]
    scored = []
    for rank, doc in enumerate(documents):
        text = doc.page_content.casefold()
        coverage = len(terms & set(_tokens(text))) / max(len(terms), 1)
        scored.append((coverage + sum(0.3 for phrase in phrases if phrase in text) + 0.15 / (rank + 1), rank, doc))
    return [doc for _, _, doc in sorted(scored, key=lambda item: (-item[0], item[1]))]


def retrieve_documents(query: str, metadata_filter: dict[str, str] | None = None) -> list[Document]:
    """Retrieve ranked chunks for one question, preserving source metadata."""
    rewritten = _rewrite_query(query)
    filters: dict[str, str] = {}
    if metadata_filter:
        unknown = set(metadata_filter) - set(_METADATA_FIELDS)
        if unknown:
            raise ValueError(f"Unsupported metadata filters: {', '.join(sorted(unknown))}")
        filters.update(metadata_filter)
    selected_mode = select_search_mode(query)
    # A metadata-constrained question often contains exact terms in the source
    # language; combine lexical and vector retrieval to avoid missing them.
    mode = "hybrid" if filters and selected_mode == "vector" else selected_mode
    logger.info("RAG query rewriting original=%r rewritten=%r", query, rewritten)
    logger.info("RAG search mode=%s metadata_filter=%s", mode, filters)
    if not (VECTOR_STORE_PATH / "index.faiss").exists():
        logger.warning("RAG index is unavailable")
        raise FileNotFoundError("No uploaded documents are available yet. Ask the user to upload a supported file.")
    store = FAISS.load_local(str(VECTOR_STORE_PATH), embeddings, allow_dangerous_deserialization=True)
    documents = _store_documents(store)
    eligible = [doc for doc in documents if _matches_metadata(doc, filters)]
    if mode == "vector":
        candidates = _vector_search(store, rewritten, k=12, filters=filters)
    elif mode == "keyword":
        candidates = _keyword_search(eligible, rewritten, k=12)
    else:
        candidates = _hybrid_search(store, rewritten, k=12, filters=filters)
    phrases = [" ".join(phrase.lower().split()) for phrase in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", query)]
    exact_matches = [doc for doc in eligible if any(phrase in " ".join(doc.page_content.lower().split()) for phrase in phrases)]
    unique = []
    seen = set()
    ranked = _rerank([*exact_matches, *candidates], rewritten)
    if filters:
        # Unknown metadata remains searchable, but explicit matches take
        # precedence over unknowns when both reach the candidate set.
        ranked.sort(key=lambda doc: sum(bool(str(doc.metadata.get(key) or "").strip()) for key in filters), reverse=True)
    for doc in ranked:
        key = " ".join(doc.page_content.lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(doc)
        if len(unique) == 4:
            break
    logger.info("RAG reranking method=lexical_coverage candidates=%d returned=%d", len(candidates), len(unique))
    return unique


def format_documents(documents: list[Document]) -> str:
    return "\n\n".join(f"Document {index}\nSource: {doc.metadata.get('source', 'Unknown')}\nMetadata: "
                       + ", ".join(f"{key}: {doc.metadata[key]}" for key in _METADATA_FIELDS if doc.metadata.get(key))
                       + f"\nContent: {doc.page_content}" for index, doc in enumerate(documents, 1))



_CONTEXT_STOPWORDS = frozenset("""a an and are as at be by can could did do does for from has have how in is it
of on or that the their these this to was what when where which who why with would
bị bởi các cho có công của được hay hãy khi không là một những như nào này nếu theo
thì từ và về với công ty tài liệu dựa trên phân tích bao nhiêu năm khoản khả năng
điều gì ra hơn như thế nào hiện hành liên quan ảnh hưởng chính sách dụng""".split())


def prepare_multi_query_context(documents: list[Document], query: str, questions: list[str],
                                run_id: str, top_k: int = 6) -> list[Document]:
    """Conservatively select and arrange evidence for a decomposed multi-source query.

    No model-generated text is introduced. Compression only removes adjacent
    identical nonblank lines; metadata is copied verbatim and clearly labelled.
    """
    sources = {str(doc.metadata.get("source") or "").strip() for doc in documents}
    if len(questions) < 2 or len(documents) < 2 or "" in sources or len(sources) < 2:
        logger.info("RAG context run=%s status=skipped reason=single_query_or_source", run_id)
        return documents

    terms = {term for text in [query, *questions] for term in _tokens(text)
             if len(term) > 1 and term not in _CONTEXT_STOPWORDS}
    token_sets = [set(_tokens(doc.page_content)) for doc in documents]
    frequencies = Counter(term for tokens in token_sets for term in tokens if term in terms)
    ranked = []
    for index, (doc, tokens) in enumerate(zip(documents, token_sets)):
        matched = terms & tokens
        score = sum(math.log(1 + (len(documents) + 1) / (frequencies[term] + 1)) for term in matched)
        ranked.append((score, index, doc))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    # A zero lexical score is not proof of irrelevance. If every score is zero,
    # retain the original retrieval order and rely on the answer's uncertainty.
    positive = [item for item in ranked if item[0] > 0]
    selected = (positive or ranked)[:top_k]
    logger.info("RAG context rerank run=%s candidates=%d selected=%d fallback=%s ranking=%s", run_id,
                len(documents), len(selected), not bool(positive),
                [(item[2].metadata.get("source", "Unknown"), round(item[0], 3)) for item in ranked])

    prepared = []
    for score, _, doc in selected:
        metadata = {field: str(doc.metadata[field]).strip() for field in _METADATA_FIELDS
                    if doc.metadata.get(field) and str(doc.metadata[field]).strip()}
        lines = doc.page_content.splitlines()
        compressed = []
        for line in lines:
            if line.strip() and compressed and line.strip() == compressed[-1].strip():
                continue
            compressed.append(line)
        body = "\n".join(compressed)
        if metadata:
            body = "Document metadata (as provided): " + "; ".join(
                f"{field}: {value}" for field, value in metadata.items()) + "\n" + body
        prepared.append(doc.model_copy(update={"page_content": body}))
        logger.info("RAG context transform run=%s source=%s contextualized=%s lines_removed=%d", run_id,
                    doc.metadata.get("source", "Unknown"), bool(metadata), len(lines) - len(compressed))

    # Highest rank first, second highest last, third next to the beginning, etc.
    ordered = prepared[::2] + prepared[1::2][::-1]
    logger.info("RAG context order run=%s sources=%s", run_id,
                [doc.metadata.get("source", "Unknown") for doc in ordered])
    return ordered

def retrieve(query: str, metadata_filter: dict[str, str] | None = None) -> str:
    """Compatibility wrapper used by the standalone RAG tool."""
    try:
        documents = retrieve_documents(query, metadata_filter)
    except FileNotFoundError as error:
        return str(error)
    return format_documents(documents) if documents else "No relevant information was found in uploaded documents."
