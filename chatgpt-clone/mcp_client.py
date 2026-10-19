"""MCP connections used by the chatbot agent.

Tavily is a hosted Streamable HTTP MCP server. Aviationstack is a local stdio
MCP server process managed by ``uvx``.  The latter receives its API key under
the name expected by the Aviationstack MCP package.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

TAVILY_MCP_URL = "https://mcp.tavily.com/mcp"


def _uvx_command() -> str | None:
    """Find uvx, including when it was installed into this virtual environment."""
    configured = os.getenv("AVIATIONSTACK_MCP_COMMAND")
    if configured:
        return shutil.which(configured)

    path_command = shutil.which("uvx")
    if path_command:
        return path_command

    venv_command = Path(sys.executable).with_name("uvx")
    return str(venv_command) if venv_command.is_file() else None


def _connections() -> dict[str, dict]:
    """Build only the MCP connections whose required credentials are present."""
    connections: dict[str, dict] = {}

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        connections["tavily"] = {
            "transport": "streamable_http",
            "url": os.getenv("TAVILY_MCP_URL", TAVILY_MCP_URL),
            # Keep the secret out of the URL, which can otherwise leak into
            # HTTP error messages or proxy access logs.
            "headers": {"Authorization": f"Bearer {tavily_api_key}"},
        }
    else:
        logger.warning("Tavily MCP is disabled: set TAVILY_API_KEY to enable it.")

    aviationstack_api_key = os.getenv("AVIATIONSTACK_API_KEY")
    uvx_command = _uvx_command()
    if aviationstack_api_key and uvx_command:
        connections["aviationstack"] = {
            "transport": "stdio",
            "command": uvx_command,
            # aviationstack-mcp currently requires Python >= 3.13. uvx will
            # download a managed interpreter if the host environment is older.
            "args": ["--python", "3.13", "aviationstack-mcp"],
            # Keep the child environment minimal while preserving the values
            # uvx needs to be found and to locate its cache.
            "env": {
                "AVIATION_STACK_API_KEY": aviationstack_api_key,
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", ""),
            },
        }
    elif aviationstack_api_key:
        logger.warning(
            "Aviationstack MCP is disabled: uvx was not found. "
            "Run 'pip install -r requirements.txt' or set AVIATIONSTACK_MCP_COMMAND.",
        )
    else:
        logger.warning("Aviationstack MCP is disabled: set AVIATIONSTACK_API_KEY to enable it.")

    return connections


async def _load_server_tools(name: str, connection: dict) -> list[BaseTool]:
    """Discover one server independently so another server's failure is isolated."""
    try:
        tools = await MultiServerMCPClient({name: connection}).get_tools()
        logger.info("Loaded MCP tools from %s: %s", name, ", ".join(tool.name for tool in tools))
        return tools
    except Exception:
        logger.exception("Could not load tools from the %s MCP server.", name)
        return []


async def _load_mcp_tools() -> list[BaseTool]:
    loaded = await asyncio.gather(
        *(_load_server_tools(name, connection) for name, connection in _connections().items())
    )
    return [tool for server_tools in loaded for tool in server_tools]


def load_mcp_tools() -> list[BaseTool]:
    """Discover LangChain tools exposed by configured Tavily and Aviationstack MCP servers.

    This is deliberately synchronous because ``agent.py`` is imported during
    FastAPI startup. Tool discovery failure is non-fatal; the remaining local
    tools and any successfully connected MCP server stay available.
    """
    def run_loader() -> list[BaseTool]:
        return asyncio.run(_load_mcp_tools())

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Normal synchronous import / command-line execution.
        return run_loader()

    # Uvicorn imports the ASGI app from inside its event loop. asyncio.run()
    # cannot nest in that loop, so create an isolated event loop in a short-
    # lived worker thread for MCP discovery.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-discovery") as executor:
        return executor.submit(run_loader).result()
