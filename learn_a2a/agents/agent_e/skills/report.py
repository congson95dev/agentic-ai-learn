import json


async def run(text: str) -> str:
    data = json.loads(text)
    return (
        "Final AI market report\n"
        f"- Research: {data['research']}\n"
        f"- Analysis: {data['analysis']}\n"
        f"- Coding: {data['coding']}"
    )
