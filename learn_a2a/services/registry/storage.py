from services.registry.models import AgentRecord


class InMemoryRegistry:
    """Storage thay thế được bằng database về sau."""

    def __init__(self) -> None:
        self.records: dict[str, AgentRecord] = {}

    def upsert(self, record: AgentRecord) -> AgentRecord:
        self.records[record.agent_id] = record
        return record

    def delete(self, agent_id: str) -> None:
        self.records.pop(agent_id, None)

    def list(self) -> list[AgentRecord]:
        return list(self.records.values())

    def get(self, agent_id: str) -> AgentRecord | None:
        return self.records.get(agent_id)

    def find_skill(self, skill: str) -> AgentRecord | None:
        return next((item for item in self.records.values() if skill in item.skills), None)
