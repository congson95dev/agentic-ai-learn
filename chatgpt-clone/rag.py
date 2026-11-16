"""File ingestion and retrieval for the local FAISS knowledge base."""
from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
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


def read_file_text(file_path: str | Path) -> int:
    """Read an allowed upload and add its text to FAISS. Returns chunk count."""
    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    documents: list[Document] = [doc for doc in _loader_for(path).load() if doc.page_content.strip()]
    if not documents:
        raise ValueError("No extractable text was found in this file.")
    for doc in documents:
        doc.metadata["source"] = path.name

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
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
    logger.info("RAG index updated source=%s chunks=%d", path.name, len(chunks))
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


def _vector_search(store: FAISS, query: str, k: int = 4) -> list[Document]:
    return store.similarity_search(query, k=k)


def _hybrid_search(store: FAISS, query: str, k: int = 4) -> list[Document]:
    documents = _store_documents(store)
    vector_results = _vector_search(store, query, k=max(k, 8))
    keyword_results = _keyword_search(documents, query, k=max(k, 8))
    # Reciprocal Rank Fusion is robust when vector and lexical scores have
    # different scales. Use document identity plus source/content as fallback.
    def key(doc: Document) -> str:
        return f"{doc.metadata.get('source', '')}\n{doc.page_content}"

    fused: dict[str, tuple[float, int, Document]] = {}
    for ranking, results in enumerate((vector_results, keyword_results)):
        for rank, doc in enumerate(results, 1):
            identifier = key(doc)
            score = 1.0 / (60 + rank)
            previous = fused.get(identifier)
            fused[identifier] = (score + (previous[0] if previous else 0.0), rank, doc)
    return [item[2] for item in sorted(fused.values(), key=lambda item: -item[0])[:k]]


def retrieve(query: str) -> str:
    mode = select_search_mode(query)
    logger.info("RAG search mode=%s", mode)
    if not (VECTOR_STORE_PATH / "index.faiss").exists():
        logger.warning("RAG index is unavailable")
        return "No uploaded documents are available yet. Ask the user to upload a supported file."
    store = FAISS.load_local(str(VECTOR_STORE_PATH), embeddings, allow_dangerous_deserialization=True)
    if mode == "vector":
        documents = _vector_search(store, query)
    elif mode == "keyword":
        documents = _keyword_search(_store_documents(store), query)
    else:
        documents = _hybrid_search(store, query)
    # A specific multiword name can be missed by local hash embeddings. Give
    # exact name matches precedence, and avoid returning duplicate passages from
    # copies of the same upload.
    phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", query)
    phrases = [" ".join(phrase.lower().split()) for phrase in phrases]
    exact_matches = [doc for doc in _store_documents(store)
                     if any(phrase in " ".join(doc.page_content.lower().split()) for phrase in phrases)]
    unique_documents: list[Document] = []
    seen: set[str] = set()
    for doc in [*exact_matches, *documents]:
        key = " ".join(doc.page_content.lower().split())
        if key not in seen:
            seen.add(key)
            unique_documents.append(doc)
        if len(unique_documents) == 4:
            break
    documents = unique_documents
    if not documents:
        return "No relevant information was found in uploaded documents."
    return "\n\n".join(
        f"Document {index}\nSource: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
        for index, doc in enumerate(documents, 1)
    )
