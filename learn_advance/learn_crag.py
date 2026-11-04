import re
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


load_dotenv()

UPPER_TH = 0.7
LOWER_TH = 0.3


class CRAGState(TypedDict):
    question: str
    docs: list[Document]
    good_docs: list[Document]
    verdict: Literal["CORRECT", "INCORRECT", "AMBIGUOUS"] | str
    reason: str
    strips: list[str]
    kept_strips: list[str]
    refined_context: str
    web_query: str
    web_docs: list[Document]
    answer: str


class DocEvalScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class KeepOrDrop(BaseModel):
    keep: bool


class WebQuery(BaseModel):
    """A concise keyword query generated for web search."""

    query: str


def load_retriever(pdf_path: str | Path):
    """Load a PDF, split it into chunks, and return a FAISS retriever."""
    documents = PyPDFLoader(str(pdf_path)).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150
    ).split_documents(documents)

    # Remove malformed characters before embedding, as in the source notebook.
    for chunk in chunks:
        chunk.page_content = chunk.page_content.encode("utf-8", "ignore").decode(
            "utf-8", "ignore"
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def decompose_to_sentences(text: str) -> list[str]:
    """Split context into usable sentence-level evidence strips."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 20]


def build_crag_app(pdf_path: str | Path):
    """Build the corrective-RAG graph for the supplied local PDF knowledge base."""
    retriever = load_retriever(pdf_path)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tavily = TavilySearch(max_results=5)

    doc_eval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict retrieval evaluator for RAG.\n"
                "You will be given ONE retrieved chunk and a question.\n"
                "Return a relevance score in [0.0, 1.0].\n"
                "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
                "- 0.0: chunk is irrelevant\n"
                "Be conservative with high scores. Also return a short reason.",
            ),
            ("human", "Question: {question}\n\nChunk:\n{chunk}"),
        ]
    )
    doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)

    filter_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict relevance filter. Return keep=true only if the "
                "sentence directly helps answer the question. Use only the sentence.",
            ),
            ("human", "Question: {question}\n\nSentence:\n{sentence}"),
        ]
    )
    filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)

    # Rewrite user questions into focused keyword queries before Tavily search.
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the user question into a web search query composed of keywords.\n"
                "Rules:\n"
                "- Keep it short (6-14 words).\n"
                "- If the question implies recency (for example recent, latest, last "
                "week, or last month), add a constraint such as (last 30 days).\n"
                "- Do not answer the question.",
            ),
            ("human", "Question: {question}"),
        ]
    )
    rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful ML tutor. Answer only using the provided context. "
                "If the context is empty or insufficient, say: 'I don't know.'",
            ),
            ("human", "Question: {question}\n\nRefined context:\n{refined_context}"),
        ]
    )

    def retrieve_node(state: CRAGState) -> dict:
        documents = retriever.invoke(state["question"])
        return {"docs": documents}

    def eval_each_doc_node(state: CRAGState) -> dict:
        scores: list[float] = []
        good_docs: list[Document] = []

        # Evaluate each retrieved document chunk to set scores
        for document in state["docs"]:
            result = doc_eval_chain.invoke(
                {"question": state["question"], "chunk": document.page_content}
            )
            scores.append(result.score)
            if result.score > LOWER_TH:
                good_docs.append(document)

        # If any score exceeds the upper threshold, return CORRECT
        if any(score > UPPER_TH for score in scores):
            return {
                "good_docs": good_docs,
                "verdict": "CORRECT",
                "reason": f"At least one retrieved chunk scored > {UPPER_TH}.",
            }

        # If all scores are below the lower threshold, return INCORRECT
        if scores and all(score < LOWER_TH for score in scores):
            return {
                "good_docs": [],
                "verdict": "INCORRECT",
                "reason": f"All retrieved chunks scored < {LOWER_TH}. No chunk was sufficient.",
            }

        # If scores are mixed and none exceed the upper threshold, return AMBIGUOUS
        return {
            "good_docs": good_docs,
            "verdict": "AMBIGUOUS",
            "reason": "No chunk scored highly enough, with mixed relevance signals.",
        }

    def rewrite_query_node(state: CRAGState) -> dict:
        rewritten = rewrite_chain.invoke({"question": state["question"]})
        return {"web_query": rewritten.query}

    def web_search_node(state: CRAGState) -> dict:
        search_query = state.get("web_query") or state["question"]
        search_response = tavily.invoke({"query": search_query})
        results = search_response.get("results", [])
        web_docs = []
        for result in results or []:
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "") or result.get("snippet", "")
            page_content = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
            web_docs.append(
                Document(page_content=page_content, metadata={"url": url, "title": title})
            )
        return {"web_docs": web_docs}

    def refine_node(state: CRAGState) -> dict:
        if state["verdict"] == "CORRECT":
            source_docs = state["good_docs"]
        elif state["verdict"] == "INCORRECT":
            source_docs = state["web_docs"]
        else:
            # Mixed retrieval evidence benefits from both local and web context.
            source_docs = state["good_docs"] + state["web_docs"]
        context = "\n\n".join(document.page_content for document in source_docs)
        strips = decompose_to_sentences(context)
        kept_strips = [
            sentence
            for sentence in strips
            if filter_chain.invoke(
                {"question": state["question"], "sentence": sentence}
            ).keep
        ]
        return {
            "strips": strips,
            "kept_strips": kept_strips,
            "refined_context": "\n".join(kept_strips).strip(),
        }

    def generate_node(state: CRAGState) -> dict:
        answer = (answer_prompt | llm).invoke(
            {"question": state["question"], "refined_context": state["refined_context"]}
        )
        return {"answer": answer.content}

    def route_after_eval(state: CRAGState) -> str:
        if state["verdict"] == "CORRECT":
            return "refine"
        return "rewrite_query"

    graph = StateGraph(CRAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("eval_each_doc", eval_each_doc_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("refine", refine_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "eval_each_doc")
    graph.add_conditional_edges(
        "eval_each_doc",
        route_after_eval,
        {
            "refine": "refine",
            "rewrite_query": "rewrite_query",
        },
    )
    graph.add_edge("rewrite_query", "web_search")
    graph.add_edge("web_search", "refine")
    graph.add_edge("refine", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def ask(app, question: str) -> dict:
    """Run a question through a graph returned by :func:`build_crag_app`."""
    return app.invoke(
        {
            "question": question,
            "docs": [],
            "good_docs": [],
            "verdict": "",
            "reason": "",
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
            "web_query": "",
            "web_docs": [],
            "answer": "",
        }
    )


def draw_architecture(app, output_path: str | Path) -> Path:
    """Export the compiled CRAG LangGraph architecture as a PNG.

    If the LangGraph PNG renderer is unavailable, save the Mermaid source
    instead, so the graph remains inspectable without adding a dependency.
    """
    png_path = Path(output_path)
    mermaid = app.get_graph().draw_mermaid()
    try:
        png_path.write_bytes(app.get_graph().draw_mermaid_png())
    except Exception as error:
        mermaid_path = png_path.with_suffix(".mmd")
        mermaid_path.write_text(mermaid, encoding="utf-8")
        print(
            "[ARCHITECTURE] Could not render PNG "
            f"({error.__class__.__name__}); Mermaid diagram saved to: {mermaid_path}"
        )
        return mermaid_path

    print(f"[ARCHITECTURE] PNG diagram saved to: {png_path}")
    return png_path


if __name__ == "__main__":
    app = build_crag_app("learn_advance/sample-data/my_paper.pdf")
    draw_architecture(app, Path(__file__).with_name("crag_architecture.png"))
    result = ask(app, "What is R2 score in the Development of Multiple Combined Regression Methods for Rainfall Measurement") # This question was answered correctly by the local PDF knowledge base.
    print("=" * 20)
    print("VERDICT:", result["verdict"])
    print("REASON:", result["reason"])
    print("\nOUTPUT:\n", result["answer"])

    result = ask(app, "Tell me AI news from last week") # This question was answered incorrectly by the local PDF knowledge base, so it was rewritten and searched on the web.
    print("=" * 20)
    print("VERDICT:", result["verdict"])
    print("REASON:", result["reason"])
    print("\nOUTPUT:\n", result["answer"])



