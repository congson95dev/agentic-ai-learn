import os

AGENT_ID = "agent-e"
NAME = "Agent E - Report"
SKILL = "report-writing"
PORT = int(os.getenv("AGENT_E_PORT", "8005"))
BASE_URL = os.getenv("AGENT_E_BASE_URL", f"http://127.0.0.1:{PORT}")
RPC_PATH = "/a2a/report"
