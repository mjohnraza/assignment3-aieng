"""Comprehensive tests for the ReAct Agent iterative loop, constraint-based evidence, and stopping safeguards."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from wandering_kettle_assistant.llm_client import parse_react_response
from wandering_kettle_assistant.mcp_client import MCPToolCallResponse
from wandering_kettle_assistant.react_agent import (
    ReActAgent,
    extract_query_facets,
    check_evidence_coverage,
    _extract_constraints,
    check_evidence_coverage_with_constraints,
    QueryConstraint,
)
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
    """Test when LLM immediately returns final answer without tool calls for simple single query."""
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
        "Retrieved pet policy: Dogs are welcome on the small terrace.",
        "Retrieved food menu: Sourdough toast, rice porridge with pickled plum.",
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
        result = await agent.arun("Can I bring my dog and what food is available?", mcp_client=mock_mcp)

    assert result["status"] == "completed"
    assert result["iterations"] == 3
    assert mock_mcp.call_tool.call_count == 2
    assert "Dogs are allowed on the terrace" in result["final_answer"]


@pytest.mark.asyncio
async def test_react_agent_live_evidence_interceptor_triggers_on_missing_constraint():
    """PROVE that live ReActAgent.arun() invokes the constraint checker and refuses
    to mark incomplete evidence as complete when the LLM attempts to stop prematurely.
    """
    mock_mcp = AsyncMock()
    # Step 1 returns dog policy only (food option is missing)
    mock_mcp.call_tool.side_effect = [
        "Dogs are welcome on the small terrace at the top of the stairs.",
        "Food menu: Roasted vegetable sandwich, rice porridge.",
    ]
    agent = ReActAgent(max_steps=4, mcp_client=mock_mcp)

    # Step 1: Tool call for pet policy
    step1_call = {
        "message": {
            "content": "Checking pet policy...",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "lookup_cafe_policy", "arguments": {"topic": "pet_policy"}}}
            ],
        }
    }

    # Step 2: Premature final answer attempt (food is not yet retrieved!)
    premature_attempt = {
        "message": {
            "content": "Dogs can sit on the terrace. I know enough to answer.",
            "tool_calls": None,
        }
    }

    # Step 3: LLM forced to call food tool after interceptor intervention
    step3_call = {
        "message": {
            "content": "I still need to look up food options.",
            "tool_calls": [
                {"id": "call_2", "function": {"name": "search_menu", "arguments": {"category": "food"}}}
            ],
        }
    }

    # Step 4: True final answer with all constraints verified
    final_answer = {
        "message": {
            "content": "Final Answer: Dogs are welcome outside, and food includes rice porridge.",
            "tool_calls": None,
        }
    }

    with patch("ollama.chat", side_effect=[step1_call, premature_attempt, step3_call, final_answer]):
        result = await agent.arun(
            "Can I bring my dog outside, and what food options do you have?",
            mcp_client=mock_mcp,
        )

    # Proves the interceptor intervened: iterations reached 4 instead of stopping at 2
    assert result["status"] == "completed"
    assert result["iterations"] == 4
    assert mock_mcp.call_tool.call_count == 2
    assert len(result["missing_facets"]) == 0


@pytest.mark.asyncio
async def test_react_agent_live_cat_question_leaves_constraint_unverified():
    """PROVE that asking about a cat when only dog evidence was returned leaves the cat constraint
    unverified in live ReActAgent.arun() and triggers the deterministic safeguard notice.
    """
    mock_mcp = AsyncMock()
    # Observation only mentions dogs and assistance animals (no cats)
    mock_mcp.call_tool.return_value = "Dogs are welcome on the small terrace. Assistance animals welcome anywhere."

    agent = ReActAgent(max_steps=1, mcp_client=mock_mcp)

    step1_call = {
        "message": {
            "content": "Checking animal policy...",
            "tool_calls": [
                {"id": "call_cat_01", "function": {"name": "lookup_cafe_policy", "arguments": {"topic": "pet_policy"}}}
            ],
        }
    }

    final_synth = {
        "message": {
            "content": "The café allows dogs on the terrace.",
        }
    }

    with patch("ollama.chat", side_effect=[step1_call, final_synth]):
        result = await agent.arun("Can I bring my cat to sit outside on the terrace?", mcp_client=mock_mcp)

    # Cat policy must be marked as unverified/missing
    assert any("Cat" in m or "cat" in m for m in result["missing_facets"])
    assert "Deterministic Safeguard Notice" in result["final_answer"]
    assert "Cat policy" in result["final_answer"]


@pytest.mark.asyncio
async def test_react_agent_failed_mcp_call_cannot_satisfy_live_constraint():
    """PROVE that an MCP call returning is_error=True cannot satisfy constraints in live ReActAgent.arun()."""
    mock_mcp = AsyncMock()
    # Tool call returns is_error=True despite having relevant keywords in the error message
    mock_mcp.call_tool.return_value = MCPToolCallResponse(
        content="Error: Dogs are welcome on the small terrace. Internal ChromaDB connection timeout.",
        is_error=True,
    )

    agent = ReActAgent(max_steps=1, mcp_client=mock_mcp)

    step1_call = {
        "message": {
            "content": "Checking policy...",
            "tool_calls": [
                {"id": "call_err_01", "function": {"name": "lookup_cafe_policy", "arguments": {"topic": "pet_policy"}}}
            ],
        }
    }

    final_synth = {
        "message": {
            "content": "The system encountered an error checking pet policy.",
        }
    }

    with patch("ollama.chat", side_effect=[step1_call, final_synth]):
        result = await agent.arun("Can I bring my dog outside on the terrace?", mcp_client=mock_mcp)

    # Failed call must not count as verified evidence
    assert len(result["missing_facets"]) >= 1
    assert "Deterministic Safeguard Notice" in result["final_answer"]


@pytest.mark.asyncio
async def test_react_agent_duplicate_call_safeguard():
    """Test that identical (tool, arguments) repeated across steps are skipped."""
    mock_mcp = AsyncMock()
    mock_mcp.call_tool.return_value = "Retrieved hours data."

    agent = ReActAgent(max_steps=3, mcp_client=mock_mcp)

    step1_call = {
        "message": {
            "content": "Checking hours...",
            "tool_calls": [
                {"id": "call_dup_01", "function": {"name": "lookup_cafe_policy", "arguments": {"topic": "hours"}}}
            ],
        }
    }

    step2_call = {
        "message": {
            "content": "Checking hours again...",
            "tool_calls": [
                {"id": "call_dup_02", "function": {"name": "lookup_cafe_policy", "arguments": {"topic": "hours"}}}
            ],
        }
    }

    step3_final = {
        "message": {
            "content": "Final answer: Hours are 8:00 to 18:00.",
            "tool_calls": None,
        }
    }

    with patch("ollama.chat", side_effect=[step1_call, step2_call, step3_final]):
        result = await agent.arun("When are you open?", mcp_client=mock_mcp)

    assert result["status"] == "completed"
    assert mock_mcp.call_tool.call_count == 1
    assert any(t.get("observation_status") == "duplicate_skipped" for t in result["trace"])


# ---------------------------------------------------------------------------
# Part D: Unit Tests for Query-Specific Constraint Checking Logic
# ---------------------------------------------------------------------------

def test_cat_question_not_satisfied_by_dog_evidence():
    """'Cat' question must NOT be satisfied by dog-only evidence."""
    constraints = _extract_constraints("Can I bring my cat to the terrace?")
    assert len(constraints) >= 1

    trace = [
        {
            "execution_status": "success",
            "is_error": False,
            "observation": {
                "content": "Dogs are welcome on the small terrace at the top of the stairs.",
            },
        }
    ]

    covered, missing = check_evidence_coverage_with_constraints(constraints, trace)
    assert len(missing) >= 1, "Cat question must NOT be satisfied by dog-only evidence"


def test_dog_question_satisfied_by_dog_evidence():
    """'Dog' question IS satisfied by dog evidence."""
    constraints = _extract_constraints("Can I bring my dog to the terrace?")

    trace = [
        {
            "execution_status": "success",
            "is_error": False,
            "observation": {
                "content": "Dogs are welcome on the small terrace at the top of the stairs.",
            },
        }
    ]

    covered, missing = check_evidence_coverage_with_constraints(constraints, trace)
    assert len(covered) == 1
    assert len(missing) == 0


def test_gluten_question_not_satisfied_by_nut_evidence():
    """Gluten question must NOT be satisfied by nut allergy evidence."""
    constraints = _extract_constraints("What options do you have for someone avoiding gluten?")

    trace = [
        {
            "execution_status": "success",
            "is_error": False,
            "observation": {
                "content": "Almond croissants are made with almond paste. We do not use peanuts anywhere.",
            },
        }
    ]

    covered, missing = check_evidence_coverage_with_constraints(constraints, trace)
    assert len(missing) >= 1, "Gluten question must NOT be satisfied by nut-only evidence"


def test_gluten_question_satisfied_by_gluten_evidence():
    """Gluten question IS satisfied by proper gluten/wheat evidence."""
    constraints = _extract_constraints("What options do you have for someone avoiding gluten?")

    trace = [
        {
            "execution_status": "success",
            "is_error": False,
            "observation": {
                "content": "We have no gluten free bread. The rice porridge and the soup of the day are usually safe.",
            },
        }
    ]

    covered, missing = check_evidence_coverage_with_constraints(constraints, trace)
    assert len(covered) == 1
    assert len(missing) == 0


def test_vegan_not_satisfied_by_generic_vegetable_hummus():
    """Vegan constraint cannot be satisfied by vegetable/hummus keywords without plant-based statement."""
    constraints = _extract_constraints("What vegan food options are available?")

    # Generic menu line without plant-based declaration
    trace = [
        {
            "execution_status": "success",
            "is_error": False,
            "observation": {
                "content": "Roasted vegetable and hummus - $5.60 on milk bread.",
            },
        }
    ]

    covered, missing = check_evidence_coverage_with_constraints(constraints, trace)
    assert len(missing) >= 1, "Vegan requirement must NOT be satisfied by vegetable keywords alone"


def test_saturday_only_evidence_does_not_satisfy_both_days_constraint():
    """Saturday+Sunday question must NOT be satisfied by Saturday-only evidence."""
    q = "What are the opening hours on Saturday and Sunday?"
    constraints = _extract_constraints(q)
    assert len(constraints) >= 2  # Saturday hours AND Sunday hours

    # Only Saturday evidence
    trace = [
        {
            "execution_status": "success",
            "is_error": False,
            "observation": {
                "content": "Saturday: 9:00 to 22:00",
            },
        }
    ]

    covered, missing = check_evidence_coverage_with_constraints(constraints, trace)
    assert len(missing) >= 1, "Saturday-only evidence must not satisfy both Saturday and Sunday constraints"


def test_per_observation_checking_not_combined():
    """Keywords split across TWO observations must NOT collectively satisfy a single constraint."""
    q = "What are the opening hours on Saturday?"
    constraints = _extract_constraints(q)

    # Observation 1 has 'Saturday' but no hours, Observation 2 has '9:00 to 22:00' but no day name
    trace = [
        {
            "execution_status": "success",
            "is_error": False,
            "observation": {"content": "We are open on Saturday for guests."},
        },
        {
            "execution_status": "success",
            "is_error": False,
            "observation": {"content": "Hours: 9:00 to 22:00."},
        },
    ]

    covered, missing = check_evidence_coverage_with_constraints(constraints, trace)
    assert len(missing) >= 1, "Keywords split across observations must not satisfy a single constraint"
