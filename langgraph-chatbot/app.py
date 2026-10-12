from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
from backend import chatbot
from helper import generate_thread_id, get_all_thread_ids, load_conversation

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
        for tool_name in message.get("tool_calls", []):
            with st.status(
                f"🔧 Calling tool: {tool_name}",
                expanded=False
            ):
                st.write(f"Tool: `{tool_name}`")
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
        tool_container = st.container()
        response_container = st.container()
        ai_chunks = []
        shown_tool_calls = set()
        tool_calls = []

        for message_chunk, metadata in chatbot.stream(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=CONFIG,
            stream_mode="messages"
        ):
            if isinstance(message_chunk, AIMessage): # Display only messages from AI, and ignore other message types
                # Display the tool currently being called
                for tool_call in getattr(message_chunk, "tool_call_chunks", []):
                    tool_name = tool_call.get("name")
                    if tool_name and tool_name not in shown_tool_calls:
                        shown_tool_calls.add(tool_name)
                        tool_calls.append(tool_name)

                        with tool_container:
                            with st.status(
                                f"🔧 Calling tool: {tool_name}",
                                expanded=False
                            ):
                                st.write(f"Tool: `{tool_name}`")

                # Collect AI response chunks
                if message_chunk.content:
                    ai_chunks.append(message_chunk.content)

        ai_message = "".join(
            chunk if isinstance(chunk, str) else str(chunk)
            for chunk in ai_chunks
        )

        with response_container:
            st.write(ai_message)

    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message,
        "tool_calls": tool_calls
    })