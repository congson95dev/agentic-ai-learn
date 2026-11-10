"""Đăng ký agent với Registry khi FastAPI service khởi động."""

from contextlib import asynccontextmanager

from shared.logging.logger import get_logger
from shared.registry_client import register_agent


def registry_lifespan(agent_id: str, name: str, base_url: str, skills: list[str]):
    @asynccontextmanager
    async def lifespan(app):
        logger = get_logger(agent_id)
        try:
            await register_agent(agent_id, name, base_url, skills)
            logger.info("registered agent_id=%s base_url=%s", agent_id, base_url)
        except Exception as error:
            logger.warning("registry unavailable at startup: %s", error)
        yield

    return lifespan
