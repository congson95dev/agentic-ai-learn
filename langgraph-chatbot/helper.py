import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ================================================================================================
# Memory
# ================================================================================================

# Init SQLite as DB Memory
# "check_same_thread = False" allow to have multiple threads, it's suitable for chatbot app
conn = sqlite3.connect(database="langgraph-chatbot/chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

# ================================================================================================
# Thread and Conversation
# ================================================================================================

def generate_thread_id():
    return str(uuid.uuid4())

def get_all_thread_ids():
    all_threads = []
    seen_thread_ids = set()
    for ckpt in checkpoint.list(None):
        thread_id = ckpt.config["configurable"]["thread_id"]
        if thread_id not in seen_thread_ids:
            seen_thread_ids.add(thread_id)
            all_threads.append(thread_id)

    return all_threads

# ================================================================================================
# RAG
# ================================================================================================

# Embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    # PDFs can contain only scanned images or blank pages. Do not pass an
    # empty document list to FAISS, which otherwise fails with IndexError.
    docs = [
        document
        for document in loader.load()
        if document.page_content and document.page_content.strip()
    ]

    if not docs:
        raise ValueError(
            "No extractable text was found in this PDF. "
            "If it is a scanned document, run OCR on it before uploading."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("The PDF did not produce any text chunks to index.")

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)

def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
            folder_path=DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return retriever
