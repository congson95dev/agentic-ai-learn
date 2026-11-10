from fastapi import FastAPI, Response

from services.registry.models import AgentRecord, RegisterAgentRequest
from services.registry.registry import RegistryService
from services.registry.storage import InMemoryRegistry


service = RegistryService(InMemoryRegistry())
app = FastAPI(title="Agent Registry")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.put("/agents/{agent_id}", response_model=AgentRecord)
def register(agent_id: str, request: RegisterAgentRequest) -> AgentRecord:
    return service.register(request.model_copy(update={"agent_id": agent_id}))


@app.get("/agents", response_model=list[AgentRecord])
def list_agents() -> list[AgentRecord]:
    return service.storage.list()


@app.get("/agents/{agent_id}", response_model=AgentRecord)
def get_agent(agent_id: str) -> AgentRecord:
    return service.get(agent_id)


@app.get("/agents/by-skill/{skill}", response_model=AgentRecord)
def find_by_skill(skill: str) -> AgentRecord:
    return service.find_skill(skill)


@app.delete("/agents/{agent_id}", status_code=204)
def delete_agent(agent_id: str) -> Response:
    service.storage.delete(agent_id)
    return Response(status_code=204)
