"""A2A client: đọc Agent Card rồi gọi trực tiếp Agent đích."""

from __future__ import annotations

import asyncio
import uuid

from a2a.client import create_client
from a2a.helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest, TaskState


class AgentTaskError(RuntimeError):
    """Agent đích trả task failed/rejected hoặc không phản hồi đúng."""


async def send_task(base_url: str, text: str, timeout_seconds: float = 20) -> str:
    """Resolve Card từ base URL, gửi task và lấy artifact cuối cùng."""
    async def call() -> str:
        client = await create_client(base_url.rstrip("/"))
        request = SendMessageRequest(
            message=new_text_message(
                text, role=Role.ROLE_USER, context_id=str(uuid.uuid4())
            )
        )
        result = ""
        try:
            async for event in client.send_message(request):
                if event.HasField("artifact_update"):
                    result = get_artifact_text(event.artifact_update.artifact)
                elif event.HasField("status_update"):
                    status = event.status_update.status
                    if status.state in {
                        TaskState.TASK_STATE_FAILED,
                        TaskState.TASK_STATE_REJECTED,
                        TaskState.TASK_STATE_CANCELED,
                    }:
                        message = (
                            get_message_text(status.message)
                            if status.HasField("message")
                            else "Task không hoàn thành."
                        )
                        raise AgentTaskError(message)
        finally:
            await client.close()
        if not result:
            raise AgentTaskError("Agent không trả artifact kết quả.")
        return result

    try:
        return await asyncio.wait_for(call(), timeout=timeout_seconds)
    except TimeoutError as error:
        raise AgentTaskError(f"Timeout khi gọi {base_url}") from error
