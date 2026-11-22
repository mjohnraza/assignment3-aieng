"""MCP Client Manager for Wandering Kettle Assistant.

Connects to the FastMCP server over Stdio transport and executes tool calls
via the standard MCP SDK ClientSession.call_tool() protocol.
Explicitly inspects and propagates MCP tool error status (is_error).
"""

import asyncio
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_SCRIPT = PROJECT_ROOT / "server.py"


@dataclass
class MCPToolCallResponse:
    """Structured response from an MCP tool call capturing content and error flag."""
    content: str
    is_error: bool = False

    def __str__(self) -> str:
        return self.content


class MCPClientManager:
    """Manages an active MCP ClientSession over stdio transport to server.py."""

    def __init__(self, server_path: Optional[str] = None):
        self.server_path = server_path or str(SERVER_SCRIPT)
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=["-u", self.server_path],
        )
        self._stdio_cm = None
        self._session_cm = None
        self.session: Optional[ClientSession] = None
        self.server_info: Optional[Any] = None
        self.protocol_version: Optional[str] = None

    async def connect(self) -> ClientSession:
        """Initialize and open the stdio connection and client session."""
        self._stdio_cm = stdio_client(self.server_params)
        read, write = await self._stdio_cm.__aenter__()

        self._session_cm = ClientSession(read, write)
        self.session = await self._session_cm.__aenter__()

        init_result = await self.session.initialize()
        self.server_info = getattr(init_result, "server_info", getattr(init_result, "serverInfo", None))
        self.protocol_version = getattr(init_result, "protocol_version", getattr(init_result, "protocolVersion", None))
        return self.session

    async def list_tools(self) -> List[Any]:
        """Query the MCP server for available tools via tools/list."""
        if not self.session:
            await self.connect()
        res = await self.session.list_tools()
        return res.tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPToolCallResponse:
        """
        Execute an MCP tool via tools/call across the stdio transport.
        Explicitly checks result.is_error / result.isError to detect server-side
        tool execution failures without relying solely on Python exceptions.
        """
        if not self.session:
            await self.connect()

        result = await self.session.call_tool(name, arguments=arguments)

        # Inspect standard MCP error flags
        is_error = getattr(result, "is_error", getattr(result, "isError", False))

        output_parts: List[str] = []
        for content_item in result.content:
            if hasattr(content_item, "text"):
                output_parts.append(content_item.text)
            else:
                output_parts.append(str(content_item))

        full_content = "\n".join(output_parts)
        return MCPToolCallResponse(content=full_content, is_error=bool(is_error))

    async def close(self) -> None:
        """Safely tear down the MCP session and stdio process."""
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_cm = None
            self.session = None

        if self._stdio_cm:
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._stdio_cm = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def call_tool_sync(self, name: str, arguments: Dict[str, Any]) -> MCPToolCallResponse:
        """Synchronous wrapper to execute an MCP tool call."""
        return asyncio.run(self.call_tool(name, arguments))
