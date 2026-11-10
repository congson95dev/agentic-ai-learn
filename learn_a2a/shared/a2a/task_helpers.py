"""Boilerplate tối thiểu để một function async trở thành A2A server."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from a2a.helpers import (
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState

from shared.logging.logger import get_logger


logger = get_logger("a2a.tasks")


class SkillExecutor(AgentExecutor):
    """Gọi ``skill(text)`` rồi trả kết quả qua A2A task artifact."""

    def __init__(self, skill: Callable[[str], Awaitable[str]]) -> None:
        self.skill = skill

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task_from_user_message(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        logger.info("task_id=%s status=received", task.id)
        text = context.get_user_input().strip()
        if not text:
            await updater.reject(new_text_message("Input text không được để trống."))
            return

        await updater.update_status(
            TaskState.TASK_STATE_WORKING,
            new_text_message("Task đang được xử lý."),
        )
        try:
            result = await self.skill(text)
        except Exception as error:
            logger.error("task_id=%s status=failed error=%s", task.id, error)
            await updater.failed(new_text_message(f"Task thất bại: {error}"))
            return

        await updater.add_artifact(
            parts=[new_text_part(result, media_type="text/plain")],
            name="result",
            last_chunk=True,
        )
        await updater.complete(new_text_message("Task hoàn thành."))
        logger.info("task_id=%s status=completed", task.id)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task:
            updater = TaskUpdater(
                event_queue, context.current_task.id, context.current_task.context_id
            )
            await updater.cancel(new_text_message("Task bị hủy."))
