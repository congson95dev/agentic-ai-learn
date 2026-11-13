"""File ingestion and retrieval for the local FAISS knowledge base."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

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
    return len(chunks)


def retrieve(query: str) -> str:
    if not (VECTOR_STORE_PATH / "index.faiss").exists():
        return "No uploaded documents are available yet. Ask the user to upload a supported file."
    store = FAISS.load_local(str(VECTOR_STORE_PATH), embeddings, allow_dangerous_deserialization=True)
    documents = store.similarity_search(query, k=4)
    if not documents:
        return "No relevant information was found in uploaded documents."
    return "\n\n".join(
        f"Document {index}\nSource: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
        for index, doc in enumerate(documents, 1)
    )
