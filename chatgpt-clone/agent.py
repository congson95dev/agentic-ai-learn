from __future__ import annotations

from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from mcp_client import load_mcp_tools
from tools import calculator, get_stock_price, purchase_stock, rag_tool, save_memory

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")


LOCAL_TOOLS = [calculator, get_stock_price, rag_tool, purchase_stock, save_memory]
MCP_TOOLS = load_mcp_tools()
TOOLS = [*LOCAL_TOOLS, *MCP_TOOLS]
# Let LangGraph emit model tokens incrementally for the FastAPI streaming UI.
llm_with_tools = ChatOpenAI(model="gpt-3.5-turbo", streaming=True).bind_tools(TOOLS, parallel_tool_calls=False)
MCP_TOOL_NAMES = frozenset(tool.name for tool in MCP_TOOLS)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _complete_tool_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Exclude incomplete tool-call turns left by an interrupted prior run.

    OpenAI requires every assistant tool call to have a matching ToolMessage.
    A failed tool run can leave an unmatched call in a persisted LangGraph
    checkpoint; retaining it makes all later user turns fail with HTTP 400.
    """
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


async def chat_node(state: ChatState):
    instructions = SystemMessage(content="""You are a helpful Agentic Chatbot.
Use rag_tool for questions about files uploaded by the user; never invent file content.
Use the local weather MCP tools for live current weather and five-day forecasts. Use the Tavily MCP tools for current or internet information and the Aviationstack MCP tools for live flight, airport, airline, route, and aircraft data. Use calculator for math and get_stock_price for live stock quotes.
Use purchase_stock when asked to simulate buying a stock; it requires user approval.
When the user explicitly asks you to remember a fact or preference, call save_memory with action='save'. When a saved preference/fact may help answer, call save_memory with action='retrieve' using relevant query words. Do not claim a memory was saved unless the tool says it was.
Answer directly when no tool is needed. After a tool result, give a clear, natural final answer.""")
    history = _complete_tool_history(state["messages"])
    return {"messages": [await llm_with_tools.ainvoke([instructions, *history])]}


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
# External MCP tools can fail independently (network, rate limit, or an MCP
# process restart). Return that failure as a ToolMessage so the model can
# recover instead of terminating the entire chat stream.
graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")


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
