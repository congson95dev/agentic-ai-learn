from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import streamlit as st
import uuid 

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

# Init SQLite as DB Memory
# "check_same_thread = False" allow to have multiple threads, it's suitable for chatbot app
conn = sqlite3.connect(database="langgraph-chatbot/chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

# Add nodes
graph.add_node('chat_node', chat_node)

# Add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# Compile
chatbot = graph.compile(checkpointer=checkpoint)

def generate_thread_id():
    return str(uuid.uuid4())

def assign_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# When create a new thread, we will:
# 1. generate new thread id and add it to the thread list
# 2. clear current chat messages from the UI
def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    assign_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []

# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):
    # Get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])

def get_all_thread_ids():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)

st.title("Agentic Chatbot with LangGraph")

# Create message_history variable when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Create thread_id variable when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# Create chat_threads variable when the app runs for the first time
# If there are previous threads saved in the DB Memory, load them and display in the sidebar
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_thread_ids()

# When user click on any thread, we'll display the messages from that thread in the main chat interface
assign_thread(st.session_state["thread_id"])

# ========================= Sidebar threading feature =========================

# Display the sidebar title
st.sidebar.title("My Conversations")

# Create a button for starting a new conversation
if st.sidebar.button("New Chat"):
    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()

# Display all conversation threads in reverse order
for thread_id in st.session_state["chat_threads"][::-1]:
    # Create one sidebar button for every thread
    if st.sidebar.button(
        str(thread_id),
        key=thread_id
    ):
        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the content messages of every threads in the sidebar and pre-load it under the background so it will load when we click on the sidebar
        messages = load_conversation(thread_id)

        message_history = []

        # Pre-load all of the messages under the background so it will load when we click on the sidebar
        for message in messages:
            # Check whether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = "user"

            # Check whether the message was sent by the AI
            elif isinstance(message, AIMessage):
                role = "assistant"

            # Ignore other message types
            else:
                continue

            message_history.append({
                "role": role,
                "content": message.content
            })

        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = message_history

        # Rerun the application to display the loaded messages
        st.rerun()

# ========================= Main chat interface =========================

# Display message as user or AI bubble depending on the role of the message
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# Create the chat input box
user_input = st.chat_input("Type here")

# Run this block after the user submits a message
if user_input:
    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.text(user_input)

    # Set thread_id to CONFIG so Langgraph can fetch chat history in the DB Memory
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }

    # AI response as streaming
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages"
            )
            if isinstance(message_chunk, AIMessage) # Display only messages from AI, and ignore other message types
        )

    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })