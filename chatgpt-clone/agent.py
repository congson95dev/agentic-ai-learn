from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from mcp_client import load_mcp_tools
from tools import calculator, get_stock_price, purchase_stock, rag_tool, save_memory

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
logger = logging.getLogger("uvicorn.error.agent")

LOCAL_TOOLS = [calculator, get_stock_price, rag_tool, purchase_stock, save_memory]
MCP_TOOLS = load_mcp_tools()
MCP_TOOL_NAMES = frozenset(tool.name for tool in MCP_TOOLS)


def _tools_matching(*names: str) -> list[BaseTool]:
    """Return MCP tools whose names contain one of ``names``."""
    return [tool for tool in MCP_TOOLS if any(name in tool.name.lower() for name in names)]


WEATHER_TOOLS = _tools_matching("weather", "forecast")
RESEARCH_TOOLS = _tools_matching("tavily")
WEATHER_TOOL_NAMES = {tool.name for tool in WEATHER_TOOLS}
RESEARCH_TOOL_NAMES = {tool.name for tool in RESEARCH_TOOLS}
AVIATION_TOOLS = [
    tool for tool in MCP_TOOLS if tool.name not in WEATHER_TOOL_NAMES | RESEARCH_TOOL_NAMES
]

AgentName = Literal[
    "research", "weather", "aviation", "data", "documents", "memory", "transaction", "general"
]


class RouteDecision(BaseModel):
    """One or more specialists required for the current user request."""

    agents: list[AgentName] = Field(min_length=1)


class DocumentAnswer(BaseModel):
    """A direct answer grounded in the retrieved document passages."""

    answer: str = Field(description="Direct answer to the question, without unrelated document text or internal instructions.")


class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    agents: list[AgentName]
    guardrail_message: str


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/openai/gpt-oss-20b")
# ``groq/`` is a LiteLLM provider prefix. ChatGroq sends the remaining model
# ID directly to Groq's API, so strip the prefix when it is present.
if GROQ_MODEL.startswith("groq/"):
    GROQ_MODEL = GROQ_MODEL.removeprefix("groq/")


def _gemini_model(*, temperature: float = 0, streaming: bool = False):
    """Create the primary Gemini chat model when Gemini credentials exist."""
    kwargs = {"model": GEMINI_MODEL, "temperature": temperature, "streaming": streaming}
    if GEMINI_API_KEY:
        kwargs["google_api_key"] = GEMINI_API_KEY
    return ChatGoogleGenerativeAI(**kwargs)


def _groq_model(*, temperature: float = 0, streaming: bool = False):
    """Create the Groq fallback chat model."""
    return ChatGroq(model=GROQ_MODEL, temperature=temperature, streaming=streaming)


def _primary_with_groq_fallback(primary_factory, fallback_factory):
    """Use Gemini first and retry the same operation with Groq on failure."""
    fallback = fallback_factory()
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY/GOOGLE_API_KEY is not set; using Groq directly")
        return fallback
    try:
        primary = primary_factory()
    except Exception as error:
        logger.warning("Gemini model could not be initialized; using Groq: %s", error)
        return fallback
    return primary.with_fallbacks([fallback])


router_llm = _primary_with_groq_fallback(
    lambda: _gemini_model().with_structured_output(RouteDecision, method="function_calling"),
    lambda: _groq_model().with_structured_output(RouteDecision, method="function_calling"),
)
final_llm = _primary_with_groq_fallback(
    lambda: _gemini_model(streaming=True),
    lambda: _groq_model(streaming=True),
)

# This graph is assembled manually with StateGraph, rather than with
# langchain.agents.create_agent(), so AgentMiddleware cannot be attached to it
# directly.  The following pre-router node is the StateGraph equivalent of the
# ContentFilterMiddleware and PIIMiddleware examples in learn_langgraph.py.
BANNED_KEYWORDS = frozenset({"hack", "exploit", "malware", "jailbreak", "bypass"})
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
CREDIT_CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
API_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")


def _latest_user_text(messages: list[BaseMessage]) -> str:
    """Return the newest plain-text user message in a persisted thread."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content
    return ""


def sanitize_user_text(text: str) -> str:
    """Redact sensitive user data before storage or any LLM call."""
    text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    text = CREDIT_CARD_PATTERN.sub("****-****-****-####", text)
    return API_KEY_PATTERN.sub("[REDACTED_API_KEY]", text)


def _model_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return history suitable for an LLM without altering stored chat history."""
    safe_history: list[BaseMessage] = []
    for message in _complete_tool_history(messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            safe_history.append(message.model_copy(update={"content": sanitize_user_text(message.content)}))
        else:
            safe_history.append(message)
    return safe_history


async def input_guardrail_node(state: ChatState) -> dict:
    """Deterministic, zero-LLM-cost filter that runs before routing or tools."""
    user_text = _latest_user_text(state.get("messages", []))
    normalized = user_text.lower()
    if any(keyword in normalized for keyword in BANNED_KEYWORDS):
        return {
            "guardrail_message": (
                "I can't process requests containing that type of potentially harmful content. "
                "Please rephrase your request."
            )
        }
    return {"guardrail_message": ""}


def route_after_input_guardrail(state: ChatState) -> Literal["router", "final_agent"]:
    """Blocked requests bypass every specialist and every LLM call."""
    return "final_agent" if state.get("guardrail_message") else "router"


def _complete_tool_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Exclude incomplete tool-call turns left by an interrupted prior run."""
    tool_response_ids = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id
    }
    complete_call_ids: set[str] = set()
    history: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            call_ids = {call["id"] for call in message.tool_calls if call.get("id")}
            if not call_ids.issubset(tool_response_ids):
                continue
            complete_call_ids.update(call_ids)
        if isinstance(message, ToolMessage) and message.tool_call_id not in complete_call_ids:
            continue
        history.append(message)
    return history


async def router_node(state: ChatState) -> dict:
    """Choose all independent specialists needed for the current request."""
    latest_user_text = _latest_user_text(state.get("messages", []))
    user_messages = [message.content for message in state.get("messages", [])
                     if isinstance(message, HumanMessage) and isinstance(message.content, str)]
    upload_marker = "The user uploaded these files in this message:"
    has_upload = any(upload_marker in text for text in user_messages)
    is_greeting = latest_user_text.strip().lower().rstrip("!. ") in {"hi", "hello", "hey", "chào", "xin chào"}
    needs_documents = has_upload and (upload_marker in latest_user_text or not is_greeting)
    messages = [
        SystemMessage(content="""You route requests for a ChatGPT-like assistant. Select all
independent specialists needed to answer the latest user request:
- research: current information from the web
- weather: current weather or a forecast
- aviation: flights, airports, airlines, routes, or aircraft
- data: calculations or live stock prices
- documents: questions about uploaded files
- memory: explicitly saving a fact/preference, or retrieving a saved one
- transaction: simulated stock purchases that require approval
- general: ordinary conversation, writing, explanation, or anything needing no specialist tool

Choose more than one when the request genuinely needs multiple independent
sources, for example weather and aviation. Do not select research merely
because a question is general. Return only the routing decision."""),
        *_model_history(state["messages"]),
    ]
    try:
        decision = await router_llm.ainvoke(messages)
        selected_agents = list(dict.fromkeys(decision.agents))
        if needs_documents and "documents" not in selected_agents:
            selected_agents = [name for name in selected_agents if name != "general"] + ["documents"]
        return {"agents": selected_agents}
    except Exception as error:
        # Keep ordinary chat usable if both providers fail or return malformed
        # structured output.
        query = _latest_user_text(state.get("messages", [])).lower()
        document_terms = ("uploaded", "upload", "file", "document", "tệp", "tài liệu")
        fallback = "documents" if needs_documents or any(term in query for term in document_terms) else "general"
        logger.warning("Router structured output failed; falling back to %s: %s", fallback, error)
        return {"agents": [fallback]}


def _tool_result_text(result: object) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


def _clean_user_facing_text(content: object) -> str:
    """Remove provider reasoning wrappers before text reaches the frontend."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or (block.get("type") not in {"thinking", "reasoning"} and not block.get("thought"))
        )
    else:
        text = str(content)

    # Some models emit a visible chain-of-thought wrapper or an agent-style
    # marker even when asked for a direct answer. Keep only the answer section.
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    marker = re.search(r"(?i)\bfinal\s+answer\s*:\s*", text)
    if marker:
        text = text[marker.end():].strip()
    return text


def _invalid_final_response(text: str, *, document_request: bool) -> bool:
    """Reject exposed instructions, private reasoning and invented tool calls."""
    if not text.strip():
        return True
    if re.search(r"(?i)do not mention routing, agents, internal notes|the user (?:is asking|said)|\bi need to use\b", text):
        return True
    return document_request and bool(re.search(
        r"(?i)\btool\s*call\b|\btool_code\b|\brag_tool\b|```\s*(?:json|python)", text
    ))


async def _document_answer(state: ChatState) -> AIMessage:
    """Generate and validate a structured answer before exposing it to the user."""
    question = _latest_user_text(state.get("messages", []))
    if question.startswith("The user uploaded these files in this message:") and "\n\nUser message: " in question:
        question = question.split("\n\nUser message: ", 1)[1]
    evidence = next((message.content for message in reversed(state.get("messages", []))
                     if isinstance(message, AIMessage) and isinstance(message.content, str)
                     and message.content.startswith("Document retrieval result:\n")), "")
    if not evidence or evidence.startswith("Document retrieval result:\nDocument retrieval failed:"):
        return AIMessage(content="I couldn't retrieve the document content. Please try again.")

    history = [message for message in _model_history(state.get("messages", []))
               if not (isinstance(message, AIMessage) and isinstance(message.content, str)
                       and message.content.startswith("Document retrieval result:\n"))]
    messages = [
        SystemMessage(content="""Answer the user's latest question using the supplied document evidence.
Return only the answer in the answer field. Be direct and concise: for a simple
question use at most three sentences. Do not add a summary, quotations, source
headers, document metadata, invented tool calls, or unrelated background unless
the user explicitly asks for them. Distinguish a person's role in the document
from their current occupation. If the question asks what a person does and the
evidence identifies them as a co-author, mention that role and their documented
affiliation; say if their current occupation is not established.
Never claim a fact is absent from the entire document based on excerpts alone."""),
        *history,
        HumanMessage(content=f"Latest question: {question}\n\nDocument evidence:\n{evidence}"),
    ]
    detailed = bool(re.search(r"(?i)\b(?:detail(?:ed|s)?|summari[sz]e|summary|explain|analysis|compare|full)\b|chi tiết|tóm tắt|giải thích|phân tích", question))
    providers = ([('Gemini', _gemini_model)] if GEMINI_API_KEY else []) + [('Groq', _groq_model)]
    for name, factory in providers:
        try:
            result = await factory(streaming=False).with_structured_output(
                DocumentAnswer, method="function_calling"
            ).ainvoke(messages)
            text = _clean_user_facing_text(result.answer)
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            if (_invalid_final_response(text, document_request=True)
                    or (not detailed and (len(text) > 750 or len(sentences) > 3))):
                raise ValueError("document answer failed output validation")
            return AIMessage(content=text)
        except Exception as error:
            logger.warning("Document answer failed provider=%s error=%s", name, type(error).__name__)
    return AIMessage(content="I couldn't produce a reliable answer from the document. Please try again.")


def make_specialist(name: AgentName, tools: list[BaseTool], instruction: str):
    """Build an isolated tool-using specialist node."""
    llm = _primary_with_groq_fallback(
        lambda: _gemini_model(streaming=True).bind_tools(
            tools, tool_choice="auto", parallel_tool_calls=False
        ),
        lambda: _groq_model(streaming=True).bind_tools(
            tools, tool_choice="auto", parallel_tool_calls=False
        ),
    )
    tools_by_name = {tool.name: tool for tool in tools}

    async def specialist_node(state: ChatState) -> dict:
        if name not in state["agents"]:
            return {}
        if name == "documents":
            # Groq's GPT-OSS models can occasionally hallucinate a tool name
            # during tool-call generation. RAG is deterministic, so invoke it
            # directly and pass the retrieved text to the final answer agent.
            query = _latest_user_text(state.get("messages", []))
            if query.startswith("The user uploaded these files in this message:") and "\n\nUser message: " in query:
                query = query.split("\n\nUser message: ", 1)[1]
            try:
                result = _tool_result_text(await rag_tool.ainvoke({"query": query}))
            except Exception as error:
                logger.exception("RAG retrieval failed")
                result = f"Document retrieval failed: {error}"
            return {"messages": [AIMessage(content=f"Document retrieval result:\n{result}")]}
        if not tools:
            return {"messages": [AIMessage(content=f"Internal {name} draft: answer the user's request directly.")]}

        response = await llm.ainvoke([
            SystemMessage(content=f"""You are the {name} specialist in a multi-agent system.
{instruction}
Use a relevant tool whenever one is needed. A separate final-answer agent will
use your findings, so keep any non-tool text as a short internal note and do
not address the user directly."""),
            *_model_history(state["messages"]),
        ])
        tool_messages: list[ToolMessage] = []
        for call in response.tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                content = f"Tool unavailable: {call['name']}"
            else:
                try:
                    content = _tool_result_text(await tool.ainvoke(call["args"]))
                except Exception as error:
                    content = f"Tool failed: {error}"
            tool_messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": [response, *tool_messages]}

    return specialist_node


async def final_agent_node(state: ChatState) -> dict:
    """Create the sole user-facing answer from all specialist findings."""
    if guardrail_message := state.get("guardrail_message"):
        return {"messages": [AIMessage(content=guardrail_message)]}
    if "documents" in state.get("agents", []):
        return {"messages": [await _document_answer(state)]}
    messages = [
        SystemMessage(content="""You are the final response agent for a multi-agent chatbot.
Answer the user's latest request directly and accurately using specialist tool
results in the conversation. Do not mention routing, agents, internal notes,
or raw tool payloads. If a selected tool failed or is unavailable, state the
relevant limitation and still answer with available information. Never claim a
stock purchase succeeded unless its tool result says so.

Return only the answer that should be shown to the user. Never output your
analysis, reasoning, phrases such as "The user said...", or a "Final Answer:"
label. Answer concisely and directly; do not reproduce raw document excerpts,
document headers, metadata, or unrelated details. If retrieved excerpts do not
answer the question, say so; do not claim a person or fact is absent from the
entire document based only on retrieved excerpts."""),
        *_model_history(state["messages"]),
    ]
    response = await final_llm.ainvoke(messages)
    text = _clean_user_facing_text(response.content)
    document_request = "documents" in state.get("agents", [])
    if _invalid_final_response(text, document_request=document_request):
        logger.warning("Final response contained internal instructions or a simulated tool call; retrying with Groq")
        try:
            response = await _groq_model(streaming=False).ainvoke(messages)
            text = _clean_user_facing_text(response.content)
        except Exception:
            logger.exception("Final response retry failed")
            text = ""
        if _invalid_final_response(text, document_request=document_request):
            text = "I couldn't produce a reliable answer from the available information. Please try again."
    return {"messages": [response.model_copy(update={"content": text})]}


graph = StateGraph(ChatState)
graph.add_node("input_guardrail", input_guardrail_node)
graph.add_node("router", router_node)
graph.add_node("research_agent", make_specialist("research", RESEARCH_TOOLS, "Find current web information with research tools."))
graph.add_node("weather_agent", make_specialist("weather", WEATHER_TOOLS, "Retrieve requested current weather or a forecast."))
graph.add_node("aviation_agent", make_specialist("aviation", AVIATION_TOOLS, "Retrieve requested live aviation information."))
graph.add_node("data_agent", make_specialist("data", [calculator, get_stock_price], "Perform calculations or retrieve stock prices."))
graph.add_node("documents_agent", make_specialist("documents", [rag_tool], "Answer questions from uploaded files using RAG."))
graph.add_node("memory_agent", make_specialist("memory", [save_memory], "Save or retrieve long-term memory only when appropriate."))
graph.add_node("transaction_agent", make_specialist("transaction", [purchase_stock], "Handle simulated stock purchases and obtain approval through the tool."))
graph.add_node("general_agent", make_specialist("general", [], "Provide a concise internal draft for a general request."))
graph.add_node("final_agent", final_agent_node)

graph.add_edge(START, "input_guardrail")
graph.add_conditional_edges("input_guardrail", route_after_input_guardrail)
SPECIALIST_NODES = [
    "research_agent", "weather_agent", "aviation_agent", "data_agent",
    "documents_agent", "memory_agent", "transaction_agent", "general_agent",
]
# This fan-out creates a single concurrent superstep. Unselected nodes no-op,
# while selected independent specialists run at the same time.
for node in SPECIALIST_NODES:
    graph.add_edge("router", node)
graph.add_edge(SPECIALIST_NODES, "final_agent")
graph.add_edge("final_agent", END)


def create_chatbot(checkpoint: AsyncSqliteSaver):
    """Compile the async graph after FastAPI has opened its SQLite connection."""
    return graph.compile(checkpointer=checkpoint)


async def run_agent_evaluation():
    """Run the offline regression suite without changing the live chat flow."""
    from evaluation import run_evaluation

    return await run_evaluation(
        input_guardrail=input_guardrail_node,
        router=router_node,
        sanitize=sanitize_user_text,
        build_chatbot=graph.compile,
    )


async def pending_interrupt(chatbot, thread_id: str):
    """Return the unresolved LangGraph interrupt for a conversation, if any."""
    snapshot = await chatbot.aget_state({"configurable": {"thread_id": thread_id}})
    direct = getattr(snapshot, "interrupts", ()) or ()
    if direct:
        return direct[0]
    for task in getattr(snapshot, "tasks", ()) or ():
        interrupts = getattr(task, "interrupts", ()) or ()
        if interrupts:
            return interrupts[0]
    return None
