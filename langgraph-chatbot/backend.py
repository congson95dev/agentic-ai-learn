from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition
from tools import search_tool, calculator, get_stock_price, get_current_weather

load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Import tool to LLM
tools = [search_tool,calculator, get_stock_price,get_current_weather]
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode(tools)

# Init SQLite as DB Memory
# "check_same_thread = False" allow to have multiple threads, it's suitable for chatbot app
conn = sqlite3.connect(database="langgraph-chatbot/chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

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

