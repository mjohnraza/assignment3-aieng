"""Part B (Steps 8-11): MCP Client implementation using Stdio transport."""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = PROJECT_ROOT / "server.py"

# Configure server stdio transport parameters
server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(SERVER_SCRIPT)],
)


async def run_client() -> None:
    print("=" * 80)
    print("PART B: MCP SDK CLIENT DEMONSTRATION")
    print(f"Connecting to MCP Server: {SERVER_SCRIPT}")
    print("=" * 80)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Initialize session
            print("\n[STEP 1] Initializing MCP Session...")
            init_result = await session.initialize()
            server_info = getattr(init_result, "server_info", getattr(init_result, "serverInfo", None))
            server_name = server_info.name if server_info else "CafeKnowledgeServer"
            server_ver = server_info.version if server_info else "1.0.0"
            proto_ver = getattr(init_result, "protocol_version", getattr(init_result, "protocolVersion", "2024-11-05"))
            print(f"Connected to: {server_name} v{server_ver}")
            print(f"Protocol version: {proto_ver}")

            # 2. List tools (tools/list)
            print("\n[STEP 2] Discovering tools via 'tools/list'...")
            tools_response = await session.list_tools()
            print(f"Discovered {len(tools_response.tools)} exposed tools:")
            for tool in tools_response.tools:
                desc = tool.description.splitlines()[0] if tool.description else ""
                print(f" - {tool.name}: {desc}")

            # 3. Call Tool 1: search
            print("\n[STEP 3] Executing Tool Call 1: 'search' (tools/call)...")
            search_query = "What plant-based beverage options do you have for warm drinks?"
            print(f"Client Request -> tool: 'search', arguments: {{'query': '{search_query}', 'top_k': 2}}")

            result_1 = await session.call_tool(
                "search",
                arguments={"query": search_query, "top_k": 2},
            )
            print("Server Response <- content:")
            for item in result_1.content:
                if hasattr(item, "text"):
                    print(item.text)
                else:
                    print(item)

            # 4. Call Tool 2: lookup_cafe_policy
            print("\n[STEP 4] Executing Tool Call 2: 'lookup_cafe_policy' (tools/call)...")
            policy_topic = "pet policy outdoor terrace"
            print(f"Client Request -> tool: 'lookup_cafe_policy', arguments: {{'topic': '{policy_topic}'}}")

            result_2 = await session.call_tool(
                "lookup_cafe_policy",
                arguments={"topic": policy_topic},
            )
            print("Server Response <- content:")
            for item in result_2.content:
                if hasattr(item, "text"):
                    print(item.text)
                else:
                    print(item)

            # 5. Call Tool 3: search_menu
            print("\n[STEP 5] Executing Tool Call 3: 'search_menu' (tools/call)...")
            result_3 = await session.call_tool(
                "search_menu",
                arguments={"category": "food", "include_allergens": True},
            )
            print("Server Response <- content:")
            for item in result_3.content:
                if hasattr(item, "text"):
                    print(item.text[:400] + ("..." if len(item.text) > 400 else ""))
                else:
                    print(item)

            print("\n" + "=" * 80)
            print("MCP EXECUTION TRACE COMPLETE")
            print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_client())
