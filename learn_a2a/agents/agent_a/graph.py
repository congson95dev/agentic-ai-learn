"""Workflow A -> C -> B -> D -> E; mọi lời gọi liên agent đều qua A2A."""

import json

from agents.agent_a.client import call_skill


async def run_workflow(user_request: str) -> str:
    research = await call_skill("web-research", user_request)
    analysis = await call_skill("data-analysis", research)
    coding = await call_skill("coding", user_request)
    report_input = json.dumps(
        {"research": research, "analysis": analysis, "coding": coding}
    )
    return await call_skill("report-writing", report_input)
