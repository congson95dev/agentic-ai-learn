import os
import tempfile
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
import streamlit as st
from backend import chatbot
from helper import generate_thread_id, get_all_thread_ids, ingest_rag_document

def assign_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        # Keep the most recently created conversation at the top of the sidebar.
        st.session_state["chat_threads"].insert(0, thread_id)

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

# ========================= HITL helper functions =========================

def get_pending_interrupt(thread_id):
    """
    Return the first unresolved LangGraph interrupt for a thread.

    Returns:
        The pending Interrupt object, or None.
    """

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        # Read the current checkpoint state
        state_snapshot = chatbot.get_state(config)

        # Some LangGraph versions expose interrupts directly
        direct_interrupts = getattr(
            state_snapshot,
            "interrupts",
            ()
        ) or ()

        if direct_interrupts:
            return direct_interrupts[0]

        # Other LangGraph versions store interrupts inside tasks
        tasks = getattr(
            state_snapshot,
            "tasks",
            ()
        ) or ()

        for task in tasks:

            task_interrupts = getattr(
                task,
                "interrupts",
                ()
            ) or ()

            if task_interrupts:
                return task_interrupts[0]

    except Exception:

        # A newly created thread may not have a checkpoint yet
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    """
    Save the pending interrupt information inside Streamlit state.
    """

    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):
    """
    Synchronize Streamlit HITL state with the LangGraph checkpoint.

    This allows a pending approval request to reappear after:
    - a Streamlit rerun
    - a browser refresh
    - switching between conversations
    """

    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:

        save_pending_interrupt(
            thread_id,
            pending_interrupt
        )

    else:

        current_pending = st.session_state.get(
            "pending_hitl"
        )

        if (
            current_pending is not None
            and current_pending.get("thread_id") == thread_id
        ):
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):
    """
    Resume an interrupted LangGraph execution.

    Args:
        decision:
            "yes" approves the stock purchase.
            "no" rejects the stock purchase.
    """

    pending_hitl = st.session_state.get(
        "pending_hitl"
    )

    if not pending_hitl:

        st.warning(
            "There is no pending action to approve or reject."
        )

        return

    # Get the thread that originally triggered the interrupt
    interrupted_thread_id = pending_hitl["thread_id"]

    # The same thread ID must be used when resuming
    resume_config = {
        "configurable": {
            "thread_id": interrupted_thread_id
        },
        "metadata": {
            "thread_id": interrupted_thread_id
        },
        "run_name": "hitl_resume_trace",
    }

    try:

        # Display the resumed response
        with st.chat_message("assistant"):

            status_holder = {
                "box": st.status(
                    "🔄 Resuming the requested action...",
                    expanded=True
                )
            }

            def resumed_ai_only_stream():

                # Resume the graph with the human decision
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):

                    # Update tool execution status
                    if isinstance(
                        message_chunk,
                        ToolMessage
                    ):

                        tool_name = getattr(
                            message_chunk,
                            "name",
                            "tool"
                        )

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                    # Stream only assistant-generated text
                    if isinstance(
                        message_chunk,
                        AIMessage
                    ):

                        if message_chunk.content:
                            yield message_chunk.content

            # Display the streamed final answer
            resumed_ai_message = st.write_stream(
                resumed_ai_only_stream()
            )

            # Check whether another interrupt occurred
            next_interrupt = get_pending_interrupt(
                interrupted_thread_id
            )

            if next_interrupt is not None:

                save_pending_interrupt(
                    interrupted_thread_id,
                    next_interrupt
                )

                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False
                )

            else:

                # No more pending approval
                st.session_state["pending_hitl"] = None

                status_holder["box"].update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False
                )

        # Save the assistant response in Streamlit UI history
        if resumed_ai_message:

            st.session_state["message_history"].append({
                "role": "assistant",
                "content": resumed_ai_message
            })

        # Rerun so the response appears in normal chat order
        st.rerun()

    except Exception as error:

        st.error(
            f"Could not resume the requested action: {error}"
        )

# ========================= HITL ADDED =========================

# Store the currently pending human approval request
if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

# When user click on any thread, we'll display the messages from that thread in the main chat interface
assign_thread(st.session_state["thread_id"])

# Recover pending approval after page refresh or rerun
sync_pending_interrupt(
    st.session_state["thread_id"]
)

# ========================= Sidebar threading feature =========================

# Display the sidebar title
st.sidebar.title("My Conversations")

# Create a button for starting a new conversation
if st.sidebar.button("New Chat"):
    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()


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

# Display all conversation threads. Display the newest conversation first.
for thread_id in st.session_state["chat_threads"]:
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

            # Pre-load tool call history
            tool_calls = []
            if isinstance(message, AIMessage):
                tool_calls = [
                    tool_call["name"]
                    for tool_call in message.tool_calls
                    if tool_call.get("name")
                ]

            message_history.append({
                "role": role,
                "content": message.content,
                "tool_calls": tool_calls,
            })

        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = message_history

        # Restore any pending approval for this conversation
        sync_pending_interrupt(thread_id)

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
        if message["content"]:
            st.text(message["content"])

# ========================= HITL approval interface =========================

# Get the currently pending approval request
pending_hitl = st.session_state.get(
    "pending_hitl"
)

# Check whether the pending approval belongs to
# the currently selected conversation
current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id")
    == st.session_state["thread_id"]
)

# Display approval controls
if current_thread_has_pending_hitl:
    st.warning(
        "🧑 Human approval required\n\n"
        f"{pending_hitl['prompt']}"
    )
    approve_column, reject_column = st.columns(2)
    # Approve button
    with approve_column:
        if st.button(
            "✅ Approve Purchase",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            use_container_width=True
        ):
            # Send "yes" back to interrupt()
            resume_hitl_execution("yes")
    # Reject button
    with reject_column:
        if st.button(
            "❌ Reject Purchase",
            key=f"reject_{st.session_state['thread_id']}",
            use_container_width=True
        ):
            # Send "no" back to interrupt()
            resume_hitl_execution("no")

# ========================= Chat Input =========================

user_input = st.chat_input(
    "Attach a PDF, ask a question, then press Send",
    accept_file="multiple",
    file_type=["pdf"],
    disabled=current_thread_has_pending_hitl # Disable input while waiting for human approval
)

# With file uploads enabled, Streamlit returns a ChatInputValue object rather than a plain string.
# Extract the text before it can be sent to LangChain.
user_text = ""
ingested_file_names = []
if user_input:
    user_text = user_input.text.strip()

    # Process uploaded PDF files
    if user_input.files:
        for uploaded_file in user_input.files:
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                with st.spinner(f"Processing {uploaded_file.name}..."):
                    ingest_rag_document(temp_path)

                ingested_file_names.append(uploaded_file.name)
                st.success(f"{uploaded_file.name} ingested successfully!")

            except Exception as e:
                st.error(f"Failed to ingest {uploaded_file.name}: {e}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

# Run this block only when the message contains a text.
if user_text:
    # The files are indexed before this message reaches the graph. Include that
    # fact in the agent-only message so a question submitted with a PDF is
    # answered from the newly uploaded document in the same chat turn.
    agent_input = user_text
    if ingested_file_names:
        filenames = ", ".join(ingested_file_names)
        agent_input = (
            f"The user attached and uploaded these PDF(s) in this same message: "
            f"{filenames}. Use rag_tool to answer from those PDF(s).\n\n"
            f"User question: {user_text}"
        )

    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_text
    })

    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.text(user_text)

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
                    HumanMessage(content=agent_input)
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

    # `interrupt()` stops the graph after this stream has started.  The
    # approval widgets above were already rendered during this Streamlit run,
    # so refresh the UI state from the checkpoint *now* and rerun below.  If
    # this is omitted, `pending_hitl` remains stale until a browser refresh
    # causes the next top-to-bottom execution of this script.
    sync_pending_interrupt(st.session_state["thread_id"])

    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message,
        "tool_calls": tool_calls
    })

    # Approval controls are rendered earlier in the script. Rerun after the
    # post-stream checkpoint sync so they appear immediately after interrupt().
    if (
        st.session_state.get("pending_hitl") is not None
        and st.session_state["pending_hitl"].get("thread_id")
        == st.session_state["thread_id"]
    ):
        st.rerun()
