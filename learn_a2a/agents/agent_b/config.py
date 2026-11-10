import os

AGENT_ID = "agent-b"
NAME = "Agent B - Data Analysis"
SKILL = "data-analysis"
PORT = int(os.getenv("AGENT_B_PORT", "8002"))
BASE_URL = os.getenv("AGENT_B_BASE_URL", f"http://127.0.0.1:{PORT}")
RPC_PATH = "/a2a/data-analysis"
