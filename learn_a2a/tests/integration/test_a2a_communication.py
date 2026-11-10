"""Chạy sau docker compose: python -m tests.integration.test_a2a_communication"""

import asyncio

from shared.a2a.card_loader import load_card
from shared.a2a.client_factory import send_task


async def main() -> None:
    card = await load_card("http://127.0.0.1:8003")
    assert card["skills"][0]["id"] == "web-research"
    result = await send_task("http://127.0.0.1:8003", "AI market")
    assert "Research findings" in result
    print("PASS: discovery Agent Card và A2A task tới Agent C")


if __name__ == "__main__":
    asyncio.run(main())
