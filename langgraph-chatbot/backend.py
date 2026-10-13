from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition
from helper import checkpoint
from tools import search_tool, calculator, get_stock_price, get_current_weather, rag_tool, purchase_stock

load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Import tool to LLM
tools = [search_tool,calculator, get_stock_price,get_current_weather, rag_tool, purchase_stock]
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"

            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"
            "- Use `purchase_stock` when the user wants to purchase a stock.\n"

            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )
    messages = [
        system_message,
        *state["messages"]
    ]
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode(tools)



graph = StateGraph(ChatState)

# Add nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

# Add edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node", tools_condition) # tools_condition: this pre-built function automatically detect if the LLM need to use tool or not
graph.add_edge('tools', 'chat_node') # after using tool, redirect back to chat_node so it can convert the response more user-friendly, otherwise, it will return json output, which is hard to read for non-tech person

# Compile
chatbot = graph.compile(checkpointer=checkpoint)

