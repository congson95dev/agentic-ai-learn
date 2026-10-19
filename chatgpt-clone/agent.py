from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import calculator, get_current_weather, get_stock_price, purchase_stock, rag_tool, save_memory, search_tool

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")


TOOLS = [search_tool, calculator, get_stock_price, get_current_weather, rag_tool, purchase_stock, save_memory]
# Let LangGraph emit model tokens incrementally for the FastAPI streaming UI.
llm_with_tools = ChatOpenAI(model="gpt-3.5-turbo", streaming=True).bind_tools(TOOLS)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    instructions = SystemMessage(content="""You are a helpful Agentic Chatbot.
Use rag_tool for questions about files uploaded by the user; never invent file content.
Use search_tool for current or internet information, calculator for math, get_stock_price for live stock quotes, and get_current_weather for live weather.
Use purchase_stock when asked to simulate buying a stock; it requires user approval.
When the user explicitly asks you to remember a fact or preference, call save_memory with action='save'. When a saved preference/fact may help answer, call save_memory with action='retrieve' using relevant query words. Do not claim a memory was saved unless the tool says it was.
Answer directly when no tool is needed. After a tool result, give a clear, natural final answer.""")
    return {"messages": [llm_with_tools.invoke([instructions, *state["messages"]])]}


_connection = sqlite3.connect(BASE_DIR / "databases" / "langgraph_checkpoints.sqlite3", check_same_thread=False)
checkpoint = SqliteSaver(_connection)
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
chatbot = graph.compile(checkpointer=checkpoint)


def pending_interrupt(thread_id: str):
    """Return the unresolved LangGraph interrupt for a conversation, if any."""
    snapshot = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    direct = getattr(snapshot, "interrupts", ()) or ()
    if direct:
        return direct[0]
    for task in getattr(snapshot, "tasks", ()) or ():
        interrupts = getattr(task, "interrupts", ()) or ()
        if interrupts:
            return interrupts[0]
    return None
