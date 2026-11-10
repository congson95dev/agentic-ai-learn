from agents.agent_c.config import AGENT_ID, BASE_URL, NAME, RPC_PATH, SKILL
from agents.agent_c.graph import workflow
from shared.a2a.server_factory import create_agent_app
from shared.service_lifecycle import registry_lifespan

app = create_agent_app(
    name=NAME, description="Research thị trường mẫu.", skill_id=SKILL,
    base_url=BASE_URL, rpc_path=RPC_PATH, skill=workflow,
    lifespan=registry_lifespan(AGENT_ID, NAME, BASE_URL, [SKILL]),
)
