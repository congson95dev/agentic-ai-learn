"""Chạy sau docker compose: python -m tests.e2e.test_multi_agent_workflow"""

import asyncio

from shared.a2a.client_factory import send_task


async def main() -> None:
    result = await send_task(
        "http://127.0.0.1:8001",
        "Research thị trường AI, phân tích dữ liệu và tạo báo cáo.",
    )
    assert "Final AI market report" in result
    assert "Research findings" in result
    assert "Data analysis" in result
    print("PASS: Agent A -> C -> B -> D -> E qua A2A")


if __name__ == "__main__":
    asyncio.run(main())
