import uuid
from backend import chatbot, checkpoint

def generate_thread_id():
    return str(uuid.uuid4())

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
    all_threads = []
    seen_thread_ids = set()
    for ckpt in checkpoint.list(None):
        thread_id = ckpt.config["configurable"]["thread_id"]
        if thread_id not in seen_thread_ids:
            seen_thread_ids.add(thread_id)
            all_threads.append(thread_id)

    return all_threads
