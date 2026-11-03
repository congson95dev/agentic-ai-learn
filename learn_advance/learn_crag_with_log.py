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
    print(f"[CRAG][PDF] Loading PDF: {pdf_path}")
    documents = PyPDFLoader(str(pdf_path)).load()
    print(f"[CRAG][PDF] Read successfully: {len(documents)} page(s).")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150
    ).split_documents(documents)
    print(
        "[CRAG][CHUNK] Created "
        f"{len(chunks)} chunk(s) (chunk_size=900, chunk_overlap=150)."
    )

    # Remove malformed characters before embedding, as in the source notebook.
    for chunk in chunks:
        chunk.page_content = chunk.page_content.encode("utf-8", "ignore").decode(
            "utf-8", "ignore"
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"[CRAG][VECTOR_DB] FAISS indexed successfully: {len(chunks)} vector(s).")
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
        print(f"[CRAG][RETRIEVE] Question: {state['question']}")
        documents = retriever.invoke(state["question"])
        print(f"[CRAG][RETRIEVE] Vector DB returned {len(documents)} chunk(s).")
        for index, document in enumerate(documents, start=1):
            page = document.metadata.get("page", "unknown")
            preview = document.page_content.replace("\n", " ")[:160]
            print(f"  - Chunk {index}: page={page}, preview={preview!r}")
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
            print(
                f"[CRAG][SCORE] Chunk {len(scores)}: score={result.score:.2f}; "
                f"reason={result.reason}"
            )
            if result.score > LOWER_TH:
                good_docs.append(document)

        # If any score exceeds the upper threshold, return CORRECT
        if any(score > UPPER_TH for score in scores):
            print(
                "[CRAG][DECISION] CORRECT -> refine local documents "
                f"({len(good_docs)} relevant chunk(s))."
            )
            return {
                "good_docs": good_docs,
                "verdict": "CORRECT",
                "reason": f"At least one retrieved chunk scored > {UPPER_TH}.",
            }

        # If all scores are below the lower threshold, return INCORRECT
        if scores and all(score < LOWER_TH for score in scores):
            print(
                "[CRAG][DECISION] INCORRECT -> rewrite query and search the web "
                f"(all {len(scores)} score(s) below {LOWER_TH})."
            )
            return {
                "good_docs": [],
                "verdict": "INCORRECT",
                "reason": f"All retrieved chunks scored < {LOWER_TH}. No chunk was sufficient.",
            }

        # If scores are mixed and none exceed the upper threshold, return AMBIGUOUS
        print(
            "[CRAG][DECISION] AMBIGUOUS -> rewrite query, search the web, then "
            f"combine {len(good_docs)} local relevant chunk(s) with web results."
        )
        return {
            "good_docs": good_docs,
            "verdict": "AMBIGUOUS",
            "reason": "No chunk scored highly enough, with mixed relevance signals.",
        }

    def rewrite_query_node(state: CRAGState) -> dict:
        rewritten = rewrite_chain.invoke({"question": state["question"]})
        print(f"[CRAG][REWRITE] Web search query: {rewritten.query}")
        return {"web_query": rewritten.query}

    def web_search_node(state: CRAGState) -> dict:
        search_query = state.get("web_query") or state["question"]
        search_response = tavily.invoke({"query": search_query})
        results = search_response.get("results", [])
        print(f"[CRAG][WEB] Tavily returned {len(results)} result(s) for: {search_query}")
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
        print(
            f"[CRAG][REFINE] Refining {len(source_docs)} source document(s) "
            f"for verdict={state['verdict']}."
        )
        context = "\n\n".join(document.page_content for document in source_docs)
        strips = decompose_to_sentences(context)
        kept_strips = [
            sentence
            for sentence in strips
            if filter_chain.invoke(
                {"question": state["question"], "sentence": sentence}
            ).keep
        ]
        print(
            f"[CRAG][REFINE] Kept {len(kept_strips)}/{len(strips)} relevant sentence(s)."
        )
        return {
            "strips": strips,
            "kept_strips": kept_strips,
            "refined_context": "\n".join(kept_strips).strip(),
        }

    def generate_node(state: CRAGState) -> dict:
        print("[CRAG][GENERATE] Generating answer from refined context.")
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
    print(f"\n[CRAG][START] Running CRAG for question: {question}")
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


if __name__ == "__main__":
    app = build_crag_app("learn_advance/sample-data/my_paper.pdf")
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


# Output:

# [CRAG][PDF] Loading PDF: learn_advance/sample-data/my_paper.pdf
# [CRAG][PDF] Read successfully: 15 page(s).
# [CRAG][CHUNK] Created 49 chunk(s) (chunk_size=900, chunk_overlap=150).
# [CRAG][VECTOR_DB] FAISS indexed successfully: 49 vector(s).

# [CRAG][START] Running CRAG for question: What is R2 score in the Development of Multiple Combined Regression Methods for Rainfall Measurement
# [CRAG][RETRIEVE] Question: What is R2 score in the Development of Multiple Combined Regression Methods for RainfallMeasurement
# [CRAG][RETRIEVE] Vector DB returned 4 chunk(s).
#   - Chunk 1: page=10, preview='Rainfall Prediction 9 4.2 Statistical measurement R2 score : The R2 score could be avery critical metric that’s utilized to assess the performance of a regress'
#   - Chunk 2: page=2, preview='Development of Multiple Combined Regression Methods for Rainfall Measurement. Nusrat Jahan Prottasha 1, Md. Jashim Uddin 2, Md. Kowsher 3, Rokeya Khatun Shorna4'
#   - Chunk 3: page=1, preview='See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/publication/354831740 Development of Multiple Combined Regress'
#   - Chunk 4: page=2, preview='model. Using this model for predicting the quantity of rainfall in some diﬀerent places. Finally, the Random Forest regressor has predicted the best r2 score of'
# [CRAG][SCORE] Chunk 1: score=1.00; reason=The chunk provides a clear definition of the R2 score, its significance in assessing regression models, and includes a formula for its calculation, directly addressing the question.
# [CRAG][SCORE] Chunk 2: score=0.00; reason=The chunk does not provide any information about the R2 score or its relevance to the development of regression methods for rainfall measurement.
# [CRAG][SCORE] Chunk 3: score=0.00; reason=The chunk does not provide any information about the R2 score or its relevance to the development of regression methods for rainfall measurement.
# [CRAG][SCORE] Chunk 4: score=0.80; reason=The chunk provides a specific R2 score (0.869904217) related to the Random Forest regressor used for predicting rainfall, which directly addresses the question about the R2 score in the context of rainfall measurement. However, it lacks broader context or explanation about what R2 score represents.
# [CRAG][DECISION] CORRECT -> refine local documents (2 relevant chunk(s)).
# [CRAG][REFINE] Refining 2 source document(s) for verdict=CORRECT.
# [CRAG][REFINE] Kept 5/9 relevant sentence(s).
# [CRAG][GENERATE] Generating answer from refined context.
# ====================
# VERDICT: CORRECT
# REASON: At least one retrieved chunk scored > 0.7.

# OUTPUT:
#  The R2 score, also known as R squared or the coefficient of determination, is a critical metric used to assess theperformance of a regression-based machine learning model. In the context of rainfall measurement, the Random Forestregressor achieved the best R2 score of 0.869904217, indicating a strong fit of the model to the data. The formula for R2 is given by:

# \[ R2 = 1 - \frac{\sum_{i=1}^{n} (\hat{y}_i - y_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} \]

# where \(\hat{y}_i\) are the predicted values, \(y_i\) are the actual values, and \(\bar{y}\) is the mean of the actual values.

# [CRAG][START] Running CRAG for question: Tell me AI news from last week
# [CRAG][RETRIEVE] Question: Tell me AI news from last week
# [CRAG][RETRIEVE] Vector DB returned 4 chunk(s).
#   - Chunk 1: page=14, preview='model selection and regularization. In An introduction to statistical learning, pages 225–288. Springer, 2021. 25. Raksha Agarwal and Niladri Chatterjee. Langre'
#   - Chunk 2: page=13, preview='diction. International Journal of Computer Science and Information Technologies , 5(3):3982–3984, 2014. 7. Sunyoung Lee, Sungzoon Cho, and Patrick M Wong. Rainf'
#   - Chunk 3: page=14, preview='bilistic forecasts of mesoscale convective system initiation using the random forestdata mining technique. Weather and Forecasting, 31(2):581–599, 2016. 17. Jo'
#   - Chunk 4: page=13, preview='An eﬃcient hybrid machine learning classiﬁer for rainfall prediction. In Journal of Physics: Conference Series, volume 1770, page 012012. IOP Publishing, 2021. '
# [CRAG][SCORE] Chunk 1: score=0.00; reason=The chunk does not contain any information related to AI news from last week; it discusses academic papers and topics unrelated to current events.
# [CRAG][SCORE] Chunk 2: score=0.00; reason=The chunk contains references to academic articles about rainfall prediction using machine learning, which is unrelated to AI news from last week.
# [CRAG][SCORE] Chunk 3: score=0.00; reason=The chunk discusses weather forecasting and climate change but does not provide any news related to AI from last week.
# [CRAG][SCORE] Chunk 4: score=0.00; reason=The chunk discusses machine learning classifiers for rainfall prediction and related research, which is not relevant to AI news from last week.
# [CRAG][DECISION] INCORRECT -> rewrite query and search the web (all 4 score(s) below 0.3).
# [CRAG][REWRITE] Web search query: AI news last week
# [CRAG][WEB] Tavily returned 5 result(s) for: AI news last week
# [CRAG][REFINE] Refining 5 source document(s) for verdict=INCORRECT.
# [CRAG][REFINE] Kept 13/45 relevant sentence(s).
# [CRAG][GENERATE] Generating answer from refined context.
# ====================
# VERDICT: INCORRECT
# REASON: All retrieved chunks scored < 0.3. No chunk was sufficient.

# OUTPUT:
#  Last week in AI news, Anthropic introduced Claude Duet, and there were significant concerns regarding AI-powered cybersecurity threats. OpenAI continued to roll out new updates, including a brand new image model. Additionally, Zoom launched an AI tool aimed at providing enterprises with better visibility over AI activity. Other highlights included Ideogram's new vector model and ElevenLabs Chimera for AI video generation. There were also discussions about AIdeepfake detection and advancements in medical AI.
