import os

AGENT_ID = "agent-c"
NAME = "Agent C - Web Research"
SKILL = "web-research"
PORT = int(os.getenv("AGENT_C_PORT", "8003"))
BASE_URL = os.getenv("AGENT_C_BASE_URL", f"http://127.0.0.1:{PORT}")
RPC_PATH = "/a2a/web-research"
