from fastapi import HTTPException

from services.registry.models import AgentRecord, RegisterAgentRequest
from services.registry.storage import InMemoryRegistry


class RegistryService:
    def __init__(self, storage: InMemoryRegistry) -> None:
        self.storage = storage

    def register(self, request: RegisterAgentRequest) -> AgentRecord:
        return self.storage.upsert(AgentRecord(**request.model_dump()))

    def get(self, agent_id: str) -> AgentRecord:
        record = self.storage.get(agent_id)
        if not record:
            raise HTTPException(status_code=404, detail="Agent không tồn tại.")
        return record

    def find_skill(self, skill: str) -> AgentRecord:
        record = self.storage.find_skill(skill)
        if not record:
            raise HTTPException(status_code=404, detail=f"Không có agent cho skill '{skill}'.")
        return record
