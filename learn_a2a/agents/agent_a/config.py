import os

AGENT_ID = "agent-a"
NAME = "Agent A - Coordinator"
SKILL = "coordinate-workflow"
PORT = int(os.getenv("AGENT_A_PORT", "8001"))
BASE_URL = os.getenv("AGENT_A_BASE_URL", f"http://127.0.0.1:{PORT}")
RPC_PATH = "/a2a/coordinator"
