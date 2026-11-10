"""Đọc Agent Card để kiểm tra discovery trong integration test/debug."""

import httpx


async def load_card(base_url: str) -> dict:
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(
            f"{base_url.rstrip('/')}/.well-known/agent-card.json"
        )
        response.raise_for_status()
        return response.json()
