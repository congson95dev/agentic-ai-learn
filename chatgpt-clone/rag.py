"""File ingestion and retrieval for the local FAISS knowledge base."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
# `tools.py` imports this module before `agent.py`. Load the repository .env
# here, before OpenAIEmbeddings validates the API key at construction time.
load_dotenv(BASE_DIR.parent / ".env")
VECTOR_STORE_PATH = BASE_DIR / "vector_store"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".py", ".csv"}
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


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
