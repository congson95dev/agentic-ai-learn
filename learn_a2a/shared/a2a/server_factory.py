"""Tạo FastAPI app có Agent Card và A2A JSON-RPC routes."""

from collections.abc import Awaitable, Callable

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from fastapi import FastAPI

from shared.a2a.task_helpers import SkillExecutor


def create_agent_app(
    *,
    name: str,
    description: str,
    skill_id: str,
    base_url: str,
    rpc_path: str,
    skill: Callable[[str], Awaitable[str]],
    lifespan=None,
) -> FastAPI:
    """Mỗi agent gọi helper này để expose đúng một A2A skill."""
    card = AgentCard(
        name=name,
        description=description,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                url=f"{base_url.rstrip('/')}{rpc_path}",
                protocol_version="1.0",
            )
        ],
        skills=[
            AgentSkill(
                id=skill_id,
                name=skill_id.replace("-", " ").title(),
                description=description,
                tags=["demo", "a2a", skill_id],
                examples=["demo task"],
                input_modes=["text/plain"],
                output_modes=["text/plain"],
            )
        ],
    )
    handler = DefaultRequestHandler(
        agent_executor=SkillExecutor(skill),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = create_agent_card_routes(card)
    routes.extend(create_jsonrpc_routes(handler, rpc_path))
    app = FastAPI(routes=routes, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "agent": name}

    return app
