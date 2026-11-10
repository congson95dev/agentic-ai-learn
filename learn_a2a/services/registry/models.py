from pydantic import BaseModel, Field


class AgentRecord(BaseModel):
    agent_id: str
    name: str
    base_url: str
    card_url: str
    skills: list[str] = Field(default_factory=list)


class RegisterAgentRequest(AgentRecord):
    pass
