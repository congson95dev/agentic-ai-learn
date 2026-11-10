"""Agent A dùng Registry discovery rồi gọi trực tiếp A2A Server đích."""

from shared.a2a.client_factory import send_task
from shared.registry_client import find_agent


async def call_skill(skill: str, task: str) -> str:
    record = await find_agent(skill)
    return await send_task(record["base_url"], task)
