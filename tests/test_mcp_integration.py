"""Integration tests connecting to the real FastMCP Server over Stdio transport.

Tests actual end-to-end communication via ClientSession:
- Protocol handshake and capability exchange
- Tool discovery (tools/list)
- Live tool execution (tools/call) for search, lookup_cafe_policy, and search_menu
- Integration with ReAct evidence-completeness check
"""

import pytest
import asyncio
from pathlib import Path

from wandering_kettle_assistant.mcp_client import MCPClientManager, MCPToolCallResponse
from wandering_kettle_assistant.react_agent import (
    extract_query_facets,
    check_evidence_coverage,
    ReActAgent,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = PROJECT_ROOT / "server.py"


@pytest.mark.asyncio
async def test_real_mcp_server_handshake_and_tool_discovery():
    """Verify live stdio connection, initialize handshake, and tool discovery."""
    async with MCPClientManager(str(SERVER_SCRIPT)) as client:
        assert client.session is not None
        assert client.server_info is not None
        assert "CafeKnowledgeServer" in client.server_info.name

        tools = await client.list_tools()
        tool_names = [t.name for t in tools]

        assert "search" in tool_names
        assert "lookup_cafe_policy" in tool_names
        assert "search_menu" in tool_names


@pytest.mark.asyncio
async def test_real_mcp_server_lookup_policy_execution():
    """Verify live tool execution of lookup_cafe_policy via ClientSession.call_tool()."""
    async with MCPClientManager(str(SERVER_SCRIPT)) as client:
        resp = await client.call_tool("lookup_cafe_policy", {"topic": "pet_policy"})
        assert isinstance(resp, MCPToolCallResponse)
        assert not resp.is_error
        # Confirms dogs on terrace policy from ground truth faq.txt
        assert "terrace" in resp.content.lower()
        assert "dog" in resp.content.lower()


@pytest.mark.asyncio
async def test_real_mcp_server_search_menu_execution():
    """Verify live tool execution of search_menu via ClientSession.call_tool()."""
    async with MCPClientManager(str(SERVER_SCRIPT)) as client:
        resp = await client.call_tool("search_menu", {"category": "food", "include_allergens": True})
        assert isinstance(resp, MCPToolCallResponse)
        assert not resp.is_error
        assert len(resp.content) > 50


@pytest.mark.asyncio
async def test_real_mcp_server_search_execution():
    """Verify live tool execution of search via ClientSession.call_tool()."""
    async with MCPClientManager(str(SERVER_SCRIPT)) as client:
        resp = await client.call_tool("search", {"query": "opening hours weekend", "top_k": 2})
        assert isinstance(resp, MCPToolCallResponse)
        assert not resp.is_error
        assert "hours" in resp.content.lower() or "saturday" in resp.content.lower() or "sunday" in resp.content.lower()


def test_react_agent_evidence_completeness_extractor():
    """Verify that multi-part user questions are correctly analyzed for evidence requirements."""
    question = "Can I bring my cat outside on the terrace, what food can I eat, and are there gluten allergy options?"
    facets = extract_query_facets(question)
    assert "pet_policy" in facets
    assert "food_menu" in facets
    assert "allergens_diet" in facets

    # Mock trace covering only pet_policy and food_menu
    trace = [
        {
            "tool_name": "lookup_cafe_policy",
            "arguments": {"topic": "pet_policy"},
            "execution_status": "success",
            "is_error": False,
            "observation": {
                "content": "Dogs are welcome on the small terrace.",
            },
        },
        {
            "tool_name": "search_menu",
            "arguments": {"category": "food"},
            "execution_status": "success",
            "is_error": False,
            "observation": {
                "content": "Ham and comte on baguette, roasted vegetable sandwich.",
            },
        },
    ]

    covered, missing = check_evidence_coverage(facets, trace)
    assert len(covered) >= 2
    assert any("Allergens" in m for m in missing)
