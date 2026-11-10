import os

AGENT_ID = "agent-d"
NAME = "Agent D - Coding"
SKILL = "coding"
PORT = int(os.getenv("AGENT_D_PORT", "8004"))
BASE_URL = os.getenv("AGENT_D_BASE_URL", f"http://127.0.0.1:{PORT}")
RPC_PATH = "/a2a/coding"
