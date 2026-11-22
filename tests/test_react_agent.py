"""Comprehensive tests for the ReAct Agent iterative loop and stopping safeguards."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from wandering_kettle_assistant.llm_client import parse_react_response
from wandering_kettle_assistant.react_agent import ReActAgent
from wandering_kettle_assistant.validation import ToolCallDecision, ToolObservation


def test_parse_react_response_action():
    text = """Thought: I need to check the pet policy for outdoor seating.
Action: lookup_cafe_policy
Action Input: {"topic": "pet_policy"}
"""
    decision = parse_react_response(text)
    assert not decision.is_final_answer
    assert decision.tool_name == "lookup_cafe_policy"
    assert decision.arguments == {"topic": "pet_policy"}
    assert "check the pet policy" in decision.thought


def test_parse_react_response_final_answer():
    text = """Thought: I have gathered all necessary information.
Final Answer: Pets are welcome on the outdoor terrace, and various vegan options are available.
"""
    decision = parse_react_response(text)
    assert decision.is_final_answer
    assert "outdoor terrace" in decision.final_answer


def test_react_agent_initialization():
    agent = ReActAgent(max_steps=4)
    assert agent.max_steps == 4
    assert "search" in agent.available_tools
    assert "lookup_cafe_policy" in agent.available_tools
    assert "search_menu" in agent.available_tools


@pytest.mark.asyncio
async def test_react_agent_single_step_completion():
    """Test when LLM immediately returns final answer without tool calls."""
    mock_mcp = AsyncMock()
    agent = ReActAgent(max_steps=5, mcp_client=mock_mcp)

    fake_response = {
        "message": {
            "content": "The cafe opens at 8am.",
            "tool_calls": None,
        }
    }

    with patch("ollama.chat", return_value=fake_response):
        result = await agent.arun("When do you open?", mcp_client=mock_mcp)

    assert result["status"] == "completed"
    assert result["iterations"] == 1
    assert "opens at 8am" in result["final_answer"]
    mock_mcp.call_tool.assert_not_called()


@pytest.mark.asyncio
async def test_react_agent_multi_step_dependent_tool_calls():
    """Test multi-turn execution: step 1 calls policy, step 2 calls menu, step 3 returns final answer."""
    mock_mcp = AsyncMock()
    mock_mcp.call_tool.side_effect = [
        "Retrieved pet policy: Dogs allowed on terrace.",
        "Retrieved food menu: Sourdough toast, rice porridge.",
    ]
    agent = ReActAgent(max_steps=5, mcp_client=mock_mcp)

    # Step 1 response: calls lookup_cafe_policy
    resp_step1 = {
        "message": {
            "content": "I should first check the terrace pet policy.",
            "tool_calls": [
                {
                    "id": "call_pet_001",
                    "function": {"name": "lookup_cafe_policy", "arguments": {"topic": "pet_policy"}},
                }
            ],
        }
    }

    # Step 2 response: calls search_menu
    resp_step2 = {
        "message": {
            "content": "Now I will search for food options.",
            "tool_calls": [
                {
                    "id": "call_menu_002",
                    "function": {"name": "search_menu", "arguments": {"category": "food", "include_allergens": True}},
                }
            ],
        }
    }

    # Step 3 response: final answer
    resp_step3 = {
        "message": {
            "content": "Final Answer: Dogs are allowed on the terrace and food includes rice porridge.",
            "tool_calls": None,
        }
    }

    with patch("ollama.chat", side_effect=[resp_step1, resp_step2, resp_step3]):
        result = await agent.arun("Can I bring my pet and what can I eat?", mcp_client=mock_mcp)

    assert result["status"] == "completed"
    assert result["iterations"] == 3
    assert mock_mcp.call_tool.call_count == 2
    assert "Dogs are allowed on the terrace" in result["final_answer"]


@pytest.mark.asyncio
async def test_react_agent_max_steps_safeguard():
    """Test that agent halts when reaching max_steps and triggers grounded synthesis."""
    mock_mcp = AsyncMock()
    mock_mcp.call_tool.return_value = "Retrieved policy data."
    agent = ReActAgent(max_steps=2, mcp_client=mock_mcp)

    # Infinite tool calling response
    infinite_tool_call = {
        "message": {
            "content": "Still searching for more info...",
            "tool_calls": [
                {
                    "id": "call_loop",
                    "function": {"name": "lookup_cafe_policy", "arguments": {"topic": "hours"}},
                }
            ],
        }
    }

    # Final forced synthesis response
    forced_synthesis = {
        "message": {
            "content": "Final safe answer based on retrieved evidence.",
        }
    }

    with patch("ollama.chat", side_effect=[infinite_tool_call, infinite_tool_call, forced_synthesis]):
        result = await agent.arun("Infinite loop test query", mcp_client=mock_mcp)

    assert result["status"] == "max_steps_reached"
    assert result["iterations"] == 2
    assert "Final safe answer" in result["final_answer"]


@pytest.mark.asyncio
async def test_react_agent_validation_error_feedback():
    """Test that schema validation errors are caught and passed as error observation to LLM."""
    mock_mcp = AsyncMock()
    agent = ReActAgent(max_steps=3, mcp_client=mock_mcp)

    # Step 1: Malformed argument (missing required 'topic')
    bad_tool_call = {
        "message": {
            "content": "Checking policy without topic...",
            "tool_calls": [
                {
                    "id": "call_bad_01",
                    "function": {"name": "lookup_cafe_policy", "arguments": {}},
                }
            ],
        }
    }

    # Step 2: Corrected tool call
    good_tool_call = {
        "message": {
            "content": "Correcting topic parameter.",
            "tool_calls": [
                {
                    "id": "call_good_02",
                    "function": {"name": "lookup_cafe_policy", "arguments": {"topic": "pet_policy"}},
                }
            ],
        }
    }
    mock_mcp.call_tool.return_value = "Valid policy data."

    # Step 3: Final answer
    final_resp = {
        "message": {
            "content": "Final answer after correcting validation error.",
            "tool_calls": None,
        }
    }

    with patch("ollama.chat", side_effect=[bad_tool_call, good_tool_call, final_resp]):
        result = await agent.arun("Test argument validation self-correction", mcp_client=mock_mcp)

    assert result["status"] == "completed"
    assert result["iterations"] == 3
    # First call failed validation, so MCP call_tool was only invoked once on the corrected arguments
    assert mock_mcp.call_tool.call_count == 1


@pytest.mark.asyncio
async def test_react_agent_mcp_is_error_flag_propagation():
    """Test that when MCP tool call returns is_error=True, agent records observation.is_error=True and status='error'."""
    from wandering_kettle_assistant.mcp_client import MCPToolCallResponse

    mock_mcp = AsyncMock()
    mock_mcp.call_tool.return_value = MCPToolCallResponse(
        content="Error: Database index is unavailable.",
        is_error=True,
    )

    agent = ReActAgent(max_steps=2, mcp_client=mock_mcp)

    step1_call = {
        "message": {
            "content": "Checking policy...",
            "tool_calls": [
                {
                    "id": "call_err_01",
                    "function": {"name": "lookup_cafe_policy", "arguments": {"topic": "pet_policy"}},
                }
            ],
        }
    }
    step2_final = {
        "message": {
            "content": "Final answer: Service temporarily unavailable.",
            "tool_calls": None,
        }
    }

    with patch("ollama.chat", side_effect=[step1_call, step2_final]):
        result = await agent.arun("Test query for error propagation", mcp_client=mock_mcp)

    assert result["status"] == "completed"
    assert result["trace"][0]["observation_status"] == "error"


@pytest.mark.asyncio
async def test_react_agent_duplicate_call_safeguard():
    """Test that identical (tool, arguments) repeated across steps are skipped."""
    mock_mcp = AsyncMock()
    mock_mcp.call_tool.return_value = "Retrieved hours data."

    agent = ReActAgent(max_steps=3, mcp_client=mock_mcp)

    # Step 1: calls lookup_cafe_policy with hours
    step1_call = {
        "message": {
            "content": "Checking hours...",
            "tool_calls": [
                {
                    "id": "call_dup_01",
                    "function": {"name": "lookup_cafe_policy", "arguments": {"topic": "hours"}},
                }
            ],
        }
    }

    # Step 2: repeats the exact same call
    step2_call = {
        "message": {
            "content": "Checking hours again...",
            "tool_calls": [
                {
                    "id": "call_dup_02",
                    "function": {"name": "lookup_cafe_policy", "arguments": {"topic": "hours"}},
                }
            ],
        }
    }

    # Step 3: final answer
    step3_final = {
        "message": {
            "content": "Final answer: Hours are 8:00 to 18:00.",
            "tool_calls": None,
        }
    }

    with patch("ollama.chat", side_effect=[step1_call, step2_call, step3_final]):
        result = await agent.arun("When are you open?", mcp_client=mock_mcp)

    assert result["status"] == "completed"
    # The duplicate call in step 2 should NOT have hit MCP call_tool a second time
    assert mock_mcp.call_tool.call_count == 1
    assert any(t.get("observation_status") == "duplicate_skipped" for t in result["trace"])
