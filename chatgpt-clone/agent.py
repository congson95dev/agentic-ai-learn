from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from mcp_client import load_mcp_tools
from tools import calculator, get_stock_price, purchase_stock, rag_tool, save_memory

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

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


class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    agents: list[AgentName]


MODEL_NAME = "gpt-3.5-turbo"
router_llm = ChatOpenAI(model=MODEL_NAME, temperature=0).with_structured_output(
    RouteDecision, method="function_calling"
)
final_llm = ChatOpenAI(model=MODEL_NAME, streaming=True)


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
    decision = await router_llm.ainvoke([
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
        *_complete_tool_history(state["messages"]),
    ])
    return {"agents": list(dict.fromkeys(decision.agents))}


def _tool_result_text(result: object) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


def make_specialist(name: AgentName, tools: list[BaseTool], instruction: str):
    """Build an isolated tool-using specialist node."""
    llm = ChatOpenAI(model=MODEL_NAME, streaming=True).bind_tools(tools, parallel_tool_calls=False)
    tools_by_name = {tool.name: tool for tool in tools}

    async def specialist_node(state: ChatState) -> dict:
        if name not in state["agents"]:
            return {}
        if not tools:
            return {"messages": [AIMessage(content=f"Internal {name} draft: answer the user's request directly.")]}

        response = await llm.ainvoke([
            SystemMessage(content=f"""You are the {name} specialist in a multi-agent system.
{instruction}
Use a relevant tool whenever one is needed. A separate final-answer agent will
use your findings, so keep any non-tool text as a short internal note and do
not address the user directly."""),
            *_complete_tool_history(state["messages"]),
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
    response = await final_llm.ainvoke([
        SystemMessage(content="""You are the final response agent for a multi-agent chatbot.
Answer the user's latest request directly and accurately using specialist tool
results in the conversation. Do not mention routing, agents, internal notes,
or raw tool payloads. If a selected tool failed or is unavailable, state the
relevant limitation and still answer with available information. Never claim a
stock purchase succeeded unless its tool result says so."""),
        *_complete_tool_history(state["messages"]),
    ])
    return {"messages": [response]}


graph = StateGraph(ChatState)
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

graph.add_edge(START, "router")
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
