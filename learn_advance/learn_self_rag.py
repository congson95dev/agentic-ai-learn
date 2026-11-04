"""Self-RAG with retrieval, grounding, and usefulness self-correction.

The graph deliberately has no web-search fallback.  A question that cannot be
answered from the local PDF ends at ``no_answer_found`` instead.
"""

from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


load_dotenv()

MAX_SUPPORT_RETRIES = 10
MAX_USEFULNESS_RETRIES = 10
NO_ANSWER = "No answer found."


class SelfRAGState(TypedDict):
    """All state needed by the Self-RAG graph."""

    question: str
    retrieval_query: str
    need_retrieval: bool
    docs: list[Document]
    relevant_docs: list[Document]
    context: str
    answer: str
    answer_instruction: str
    support: Literal["fully_supported", "partially_supported", "no_support"] | str
    evidence: list[str]
    support_retries: int
    usefulness: Literal["useful", "not_useful"] | str
    usefulness_reason: str
    usefulness_retries: int
    status: str


class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        description="Whether the local knowledge base is needed to answer reliably."
    )


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        description="True only when the document helps answer the question."
    )


class SupportDecision(BaseModel):
    support: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: list[str] = Field(default_factory=list)


class UsefulnessDecision(BaseModel):
    usefulness: Literal["useful", "not_useful"]
    reason: Literal[
        "Answer is incomplete",
        "Poor presentation",
        "Retrieved Context is insufficient",
        "",
    ] = ""


class RewriteDecision(BaseModel):
    retrieval_query: str


def load_retriever(pdf_path: str | Path, k: int = 4):
    """Create a FAISS retriever from a local PDF."""
    documents = PyPDFLoader(str(pdf_path)).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150
    ).split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def build_self_rag_app(pdf_path: str | Path, *, k: int = 4, verbose: bool = True):
    """Build the Self-RAG workflow for one local PDF knowledge base.

    Flow:
    decide retrieval -> retrieve -> isRel -> draft -> isSup -> isUse
                                                   ^          |
                                                   +----------+

    ``isUse`` uses its reason to choose between regenerating, improving
    presentation, or rewriting the retrieval query and retrieving again.
    """
    def preview(value: str, limit: int = 180) -> str:
        """Keep console logs readable without hiding the current decision."""
        normalized = " ".join(value.split())
        return normalized if len(normalized) <= limit else f"{normalized[:limit]}..."

    def trace(step: str, message: str) -> None:
        if verbose:
            print(f"[Self-RAG | {step}] {message}")

    trace("SETUP", f"Loading PDF='{pdf_path}', retrieval k={k}")
    retriever = load_retriever(pdf_path, k=k)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    decide_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Decide whether the question needs facts from the local PDF knowledge "
                "base. Retrieve for document-specific facts; do not retrieve for general "
                "definitions or explanations. If uncertain, choose retrieval.",
            ),
            ("human", "Question: {question}"),
        ]
    )
    decide_chain = decide_prompt | llm.with_structured_output(RetrieveDecision)

    direct_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the general question concisely from your knowledge. Do not claim "
                "to know private or PDF-specific facts.",
            ),
            ("human", "Question: {question}"),
        ]
    )

    relevance_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Judge one retrieved document. It is relevant only if it contains facts "
                "that could directly help answer the question. Do not use outside knowledge.",
            ),
            ("human", "Question:\n{question}\n\nDocument:\n{document}"),
        ]
    )
    relevance_chain = relevance_prompt | llm.with_structured_output(RelevanceDecision)

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer only from the supplied context. Do not invent facts or use outside "
                "knowledge. If the context cannot answer the question, answer exactly: "
                f"{NO_ANSWER}\n\nAdditional answer instruction: {{answer_instruction}}",
            ),
            ("human", "Question:\n{question}\n\nContext:\n{context}"),
        ]
    )

    support_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Check whether the answer is grounded in the context. Classify it as "
                "fully_supported only if every meaningful claim is explicitly supported; "
                "partially_supported if only some claims are supported; and no_support if "
                "the answer's key claims are unsupported. Return up to three short direct "
                "quotes from the context as evidence. Do not use outside knowledge.",
            ),
            (
                "human",
                "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}",
            ),
        ]
    )
    support_chain = support_prompt | llm.with_structured_output(SupportDecision)

    usefulness_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Assess whether the answer is useful for the question. If useful, set "
                "usefulness='useful' and reason=''. Otherwise set usefulness='not_useful' "
                "and choose exactly one reason: 'Answer is incomplete', 'Poor presentation', "
                "or 'Retrieved Context is insufficient'. Use 'Answer is incomplete' when "
                "the answer omits answerable details; 'Poor presentation' when the facts are "
                "there but organization/clarity is the issue; and 'Retrieved Context is "
                "insufficient' when available context cannot answer the question. Do not "
                "re-evaluate grounding or use outside knowledge.",
            ),
            ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
        ]
    )
    usefulness_chain = usefulness_prompt | llm.with_structured_output(UsefulnessDecision)

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the question into a concise query for semantic search over the "
                "local PDF. Preserve entities and requested facts. Add useful domain "
                "keywords, but do not answer the question.",
            ),
            (
                "human",
                "Question: {question}\nPrevious query: {retrieval_query}\n"
                "Why retrieval was insufficient: {reason}",
            ),
        ]
    )
    rewrite_chain = rewrite_prompt | llm.with_structured_output(RewriteDecision)

    def decide_retrieval(state: SelfRAGState) -> dict:
        trace("1. DECIDE_RAG", f"Question: {state['question']}")
        decision = decide_chain.invoke({"question": state["question"]})
        trace("1. DECIDE_RAG", f"should_retrieve={decision.should_retrieve}")
        return {"need_retrieval": decision.should_retrieve}

    def generate_direct(state: SelfRAGState) -> dict:
        trace("DIRECT_ANSWER", "RAG is not needed; generating a direct answer.")
        result = (direct_prompt | llm).invoke({"question": state["question"]})
        trace("DIRECT_ANSWER", f"Answer: {preview(result.content)}")
        return {"answer": result.content, "status": "answered_without_retrieval"}

    def retrieve(state: SelfRAGState) -> dict:
        query = state.get("retrieval_query") or state["question"]
        trace("2. RETRIEVE", f"Query: {query}")
        documents = retriever.invoke(query)
        trace("2. RETRIEVE", f"Retrieved {len(documents)} document(s).")
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "unknown")
            page = document.metadata.get("page", "unknown")
            trace(
                "2. RETRIEVE",
                f"  doc {index}: source={source}, page={page}, text={preview(document.page_content, 110)}",
            )
        return {"docs": documents}

    def is_related(state: SelfRAGState) -> dict:
        trace("2. IS_REL", f"Checking relevance of {len(state.get('docs', []))} document(s).")
        related: list[Document] = []
        for index, document in enumerate(state.get("docs", []), start=1):
            decision = relevance_chain.invoke(
                {"question": state["question"], "document": document.page_content}
            )
            trace("2. IS_REL", f"  doc {index}: is_relevant={decision.is_relevant}")
            if decision.is_relevant:
                related.append(document)
        trace("2. IS_REL", f"Selected {len(related)} relevant document(s).")
        return {"relevant_docs": related}

    def generate_from_context(state: SelfRAGState) -> dict:
        context = "\n\n---\n\n".join(
            document.page_content for document in state.get("relevant_docs", [])
        )
        trace(
            "2. DRAFT",
            f"Generating from {len(state.get('relevant_docs', []))} relevant document(s), "
            f"context={len(context)} characters.",
        )
        if state.get("answer_instruction"):
            trace("2. DRAFT", f"Instruction: {state['answer_instruction']}")
        result = (answer_prompt | llm).invoke(
            {
                "question": state["question"],
                "context": context,
                "answer_instruction": state.get("answer_instruction", ""),
            }
        )
        trace("2. DRAFT", f"Draft: {preview(result.content)}")
        return {"context": context, "answer": result.content}

    def is_supported(state: SelfRAGState) -> dict:
        trace(
            "3. IS_SUP",
            f"Checking grounding (support retry {state.get('support_retries', 0)}/{MAX_SUPPORT_RETRIES}).",
        )
        decision = support_chain.invoke(
            {
                "question": state["question"],
                "answer": state["answer"],
                "context": state["context"],
            }
        )
        trace("3. IS_SUP", f"Result: {decision.support}")
        if decision.evidence:
            for evidence in decision.evidence:
                trace("3. IS_SUP", f"  Evidence: {preview(evidence, 120)}")
        return {"support": decision.support, "evidence": decision.evidence}

    def retry_after_unsupported_answer(state: SelfRAGState) -> dict:
        retry = state.get("support_retries", 0) + 1
        trace("3. IS_SUP", f"Not fully supported; regenerating answer (retry {retry}/{MAX_SUPPORT_RETRIES}).")
        return {
            "support_retries": retry,
            "answer_instruction": (
                "Regenerate a precise answer containing only claims explicitly stated "
                "in the context. Remove unsupported interpretations."
            ),
        }

    def is_useful(state: SelfRAGState) -> dict:
        trace(
            "4. IS_USE",
            f"Checking usefulness (retry {state.get('usefulness_retries', 0)}/{MAX_USEFULNESS_RETRIES}).",
        )
        decision = usefulness_chain.invoke(
            {"question": state["question"], "answer": state["answer"]}
        )
        trace(
            "4. IS_USE",
            f"Result: {decision.usefulness}; reason={decision.reason or 'N/A'}",
        )
        return {"usefulness": decision.usefulness, "usefulness_reason": decision.reason}

    def retry_incomplete_answer(state: SelfRAGState) -> dict:
        retry = state.get("usefulness_retries", 0) + 1
        trace("4. IS_USE", f"Reason='Answer is incomplete'; regenerating (retry {retry}/{MAX_USEFULNESS_RETRIES}).")
        return {
            "usefulness_retries": retry,
            "answer_instruction": (
                "Give a complete direct answer to every part of the question, using all "
                "relevant details available in the context and no outside facts."
            ),
        }

    def improve_presentation(state: SelfRAGState) -> dict:
        retry = state.get("usefulness_retries", 0) + 1
        trace("4. IS_USE", f"Reason='Poor presentation'; improving format (retry {retry}/{MAX_USEFULNESS_RETRIES}).")
        return {
            "usefulness_retries": retry,
            "answer_instruction": (
                "Regenerate the answer with clear presentation: lead with the direct answer, "
                "then use short bullets only when they improve readability. Keep it concise."
            ),
        }

    def rewrite_question(state: SelfRAGState) -> dict:
        retry = state.get("usefulness_retries", 0) + 1
        trace(
            "4. IS_USE",
            "Reason='Retrieved Context is insufficient'; rewriting query and retrieving again "
            f"(retry {retry}/{MAX_USEFULNESS_RETRIES}).",
        )
        decision = rewrite_chain.invoke(
            {
                "question": state["question"],
                "retrieval_query": state.get("retrieval_query", state["question"]),
                "reason": state["usefulness_reason"],
            }
        )
        trace("4. REWRITE_QUERY", f"New retrieval query: {decision.retrieval_query}")
        return {
            "retrieval_query": decision.retrieval_query,
            "docs": [],
            "relevant_docs": [],
            "context": "",
            "answer_instruction": "",
            "usefulness_retries": retry,
        }

    def no_answer_found(_: SelfRAGState) -> dict:
        trace("STOP", f"Returning '{NO_ANSWER}' (no web-search fallback is configured).")
        return {"answer": NO_ANSWER, "status": "no_answer_found"}

    def accept_answer(state: SelfRAGState) -> dict:
        trace("DONE", "Answer is fully supported and useful.")
        return {"status": "answered_with_rag"}

    def route_after_decide(state: SelfRAGState) -> str:
        destination = "retrieve" if state["need_retrieval"] else "generate_direct"
        trace("ROUTE", f"DECIDE_RAG -> {destination}")
        return destination

    def route_after_related(state: SelfRAGState) -> str:
        destination = "generate_from_context" if state.get("relevant_docs") else "no_answer_found"
        trace("ROUTE", f"IS_REL -> {destination}")
        return destination

    def route_after_supported(state: SelfRAGState) -> str:
        if state["support"] == "fully_supported":
            destination = "is_useful"
        elif state.get("support_retries", 0) >= MAX_SUPPORT_RETRIES:
            destination = "no_answer_found"
        else:
            destination = "retry_after_unsupported_answer"
        trace("ROUTE", f"IS_SUP -> {destination}")
        return destination

    def route_after_useful(state: SelfRAGState) -> str:
        if state["usefulness"] == "useful":
            destination = "accept_answer"
        elif state.get("usefulness_retries", 0) >= MAX_USEFULNESS_RETRIES:
            destination = "no_answer_found"
        else:
            reason = state["usefulness_reason"]
            if reason == "Answer is incomplete":
                destination = "retry_incomplete_answer"
            elif reason == "Poor presentation":
                destination = "improve_presentation"
            else:
                destination = "rewrite_question"  # Retrieved Context is insufficient
        trace("ROUTE", f"IS_USE -> {destination}")
        return destination

    graph = StateGraph(SelfRAGState)
    graph.add_node("decide_retrieval", decide_retrieval)
    graph.add_node("generate_direct", generate_direct)
    graph.add_node("retrieve", retrieve)
    graph.add_node("is_related", is_related)
    graph.add_node("generate_from_context", generate_from_context)
    graph.add_node("is_supported", is_supported)
    graph.add_node("retry_after_unsupported_answer", retry_after_unsupported_answer)
    graph.add_node("is_useful", is_useful)
    graph.add_node("retry_incomplete_answer", retry_incomplete_answer)
    graph.add_node("improve_presentation", improve_presentation)
    graph.add_node("rewrite_question", rewrite_question)
    graph.add_node("no_answer_found", no_answer_found)
    graph.add_node("accept_answer", accept_answer)

    graph.add_edge(START, "decide_retrieval")
    graph.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {"retrieve": "retrieve", "generate_direct": "generate_direct"},
    )
    graph.add_edge("generate_direct", END)
    graph.add_edge("retrieve", "is_related")
    graph.add_conditional_edges(
        "is_related",
        route_after_related,
        {
            "generate_from_context": "generate_from_context",
            "no_answer_found": "no_answer_found",
        },
    )
    graph.add_edge("generate_from_context", "is_supported")
    graph.add_conditional_edges(
        "is_supported",
        route_after_supported,
        {
            "is_useful": "is_useful",
            "retry_after_unsupported_answer": "retry_after_unsupported_answer",
            "no_answer_found": "no_answer_found",
        },
    )
    graph.add_edge("retry_after_unsupported_answer", "generate_from_context")
    graph.add_conditional_edges(
        "is_useful",
        route_after_useful,
        {
            "accept_answer": "accept_answer",
            "retry_incomplete_answer": "retry_incomplete_answer",
            "improve_presentation": "improve_presentation",
            "rewrite_question": "rewrite_question",
            "no_answer_found": "no_answer_found",
        },
    )
    graph.add_edge("retry_incomplete_answer", "generate_from_context")
    graph.add_edge("improve_presentation", "generate_from_context")
    graph.add_edge("rewrite_question", "retrieve")
    graph.add_edge("no_answer_found", END)
    graph.add_edge("accept_answer", END)
    return graph.compile()


def ask(app, question: str) -> dict:
    """Run a question through a graph returned by :func:`build_self_rag_app`."""
    return app.invoke(
        {
            "question": question,
            "retrieval_query": question,
            "need_retrieval": False,
            "docs": [],
            "relevant_docs": [],
            "context": "",
            "answer": "",
            "answer_instruction": "",
            "support": "",
            "evidence": [],
            "support_retries": 0,
            "usefulness": "",
            "usefulness_reason": "",
            "usefulness_retries": 0,
            "status": "",
        },
        # The maximum possible route needs more than the default recursion limit.
        config={"recursion_limit": 80},
    )


def draw_architecture(app, output_path: str | Path) -> Path:
    """Export the compiled LangGraph architecture as a PNG.

    LangGraph's PNG renderer can be unavailable in restricted/offline
    environments. In that case, save an equivalent Mermaid diagram instead.
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
    app = build_self_rag_app("learn_advance/sample-data/my_paper.pdf")
    draw_architecture(
        app, Path(__file__).with_name("self_rag_architecture.png")
    )
    result = ask(
        app,
        "What is R2 score in the Development of Multiple Combined Regression "
        "Methods for Rainfall Measurement?",
    )
    print("Answer:", result["answer"])
    print("Status:", result["status"])
    print("Support:", result["support"])
    print("Usefulness:", result["usefulness"], result["usefulness_reason"])
