"""HTTP client cho Registry; Registry không proxy A2A task traffic."""

import httpx

from shared.config.settings import registry_url
from shared.exceptions.errors import RegistryError


async def register_agent(
    agent_id: str, name: str, base_url: str, skills: list[str]
) -> None:
    payload = {
        "agent_id": agent_id,
        "name": name,
        "base_url": base_url,
        "card_url": f"{base_url}/.well-known/agent-card.json",
        "skills": skills,
    }
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.put(f"{registry_url()}/agents/{agent_id}", json=payload)
        response.raise_for_status()


async def find_agent(skill: str) -> dict:
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            response = await client.get(f"{registry_url()}/agents/by-skill/{skill}")
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise RegistryError(f"Không tìm thấy agent cho skill '{skill}': {error}") from error
        return response.json()
