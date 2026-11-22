"""Comprehensive tests for the ReAct Agent iterative loop and stopping safeguards."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from wandering_kettle_assistant.llm_client import parse_react_response
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


# ---------------------------------------------------------------------------
# Part D: Evidence Verification & Edge-Case Tests
# ---------------------------------------------------------------------------

def test_evidence_checker_successful_pet_policy_accepted():
    """Test 1: Successful pet-policy lookup is accepted as verified evidence."""
    facets = extract_query_facets("Can I bring my dog on the outdoor terrace?")
    assert "pet_policy" in facets

    trace = [
        {
            "tool_name": "lookup_cafe_policy",
            "arguments": {"topic": "pet_policy"},
            "execution_status": "success",
            "is_error": False,
            "observation": {
                "content": "Dogs are welcome on the small terrace at the top of the stairs, but not inside.",
            },
        }
    ]

    covered, missing = check_evidence_coverage(facets, trace)
    assert len(covered) == 1
    assert "pet" in covered[0].lower() or "outdoor" in covered[0].lower()
    assert len(missing) == 0


def test_evidence_checker_failed_pet_policy_rejected():
    """Test 2: Failed pet-policy lookup (is_error=True or execution_status='error') is strictly rejected."""
    facets = extract_query_facets("Can I bring my dog on the outdoor terrace?")
    assert "pet_policy" in facets

    # Trace with an error observation
    trace = [
        {
            "tool_name": "lookup_cafe_policy",
            "arguments": {"topic": "pet_policy"},
            "execution_status": "error",
            "is_error": True,
            "observation": {
                "content": "Database error: connection to ChromaDB timed out.",
            },
        }
    ]

    covered, missing = check_evidence_coverage(facets, trace)
    assert len(covered) == 0
    assert len(missing) == 1
    assert "pet" in missing[0].lower() or "outdoor" in missing[0].lower()


def test_evidence_checker_irrelevant_observation_rejected():
    """Test 3: Successful tool execution returning irrelevant text is rejected."""
    facets = extract_query_facets("Can I bring my dog on the outdoor terrace?")
    assert "pet_policy" in facets

    # Successful call, but returning Wi-Fi password instead of pet policy
    trace = [
        {
            "tool_name": "lookup_cafe_policy",
            "arguments": {"topic": "wifi"},
            "execution_status": "success",
            "is_error": False,
            "observation": {
                "content": "The network is kettle-guest and the password is on the coaster.",
            },
        }
    ]

    covered, missing = check_evidence_coverage(facets, trace)
    assert len(covered) == 0
    assert len(missing) == 1


def test_evidence_checker_multi_facet_partial_retrieval():
    """Test 4: Multi-facet query with only one facet retrieved leaves the other facet missing."""
    facets = extract_query_facets("Can I bring my dog outside, and what vegan options do you have?")
    assert "pet_policy" in facets
    assert "allergens_diet" in facets
    assert "food_menu" not in facets

    # Only pet policy was retrieved
    trace = [
        {
            "tool_name": "lookup_cafe_policy",
            "arguments": {"topic": "pet_policy"},
            "execution_status": "success",
            "is_error": False,
            "observation": {
                "content": "Dogs are welcome on the small terrace at the top of the stairs.",
            },
        }
    ]

    covered, missing = check_evidence_coverage(facets, trace)
    assert len(covered) == 1
    assert len(missing) == 1
    assert any("Allergen" in m or "diet" in m for m in missing)


@pytest.mark.asyncio
async def test_react_agent_max_iterations_refuses_unverified_claim():
    """Test 5: Agent at max iterations with missing evidence includes deterministic safe-stop notice."""
    mock_mcp = AsyncMock()
    # Step 1 returns pet policy only
    mock_mcp.call_tool.return_value = "Dogs are welcome on the small terrace."

    agent = ReActAgent(max_steps=1, mcp_client=mock_mcp)

    step1_call = {
        "message": {
            "content": "Checking terrace policy...",
            "tool_calls": [
                {
                    "id": "call_pet_only",
                    "function": {"name": "lookup_cafe_policy", "arguments": {"topic": "pet_policy"}},
                }
            ],
        }
    }

    final_synth = {
        "message": {
            "content": "Dogs are allowed on the terrace.",
        }
    }

    # Query asks for pet policy AND food menu AND gluten allergy
    query = "Can I bring my dog outside, what food options are available, and what about gluten allergy?"

    with patch("ollama.chat", side_effect=[step1_call, final_synth]):
        result = await agent.arun(query, mcp_client=mock_mcp)

    assert result["status"] == "max_steps_reached"
    # Deterministic safeguard notice must be present indicating unverified aspects
    assert "Deterministic Safeguard Notice" in result["final_answer"]
    assert len(result["missing_facets"]) > 0


# ---------------------------------------------------------------------------
# Part D: Query-Specific Constraint-Based Evidence Checks
# ---------------------------------------------------------------------------

def test_cat_question_not_satisfied_by_dog_evidence():
    """'Cat' question must NOT be satisfied by dog-only evidence when using constraints."""
    constraints = _extract_constraints(
        "Can I bring my cat to the terrace?",
        extract_query_facets("Can I bring my cat to the terrace?"),
    )
    assert len(constraints) >= 1

    # Only dog evidence (no mention of cat or generic animal)
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
    constraints = _extract_constraints(
        "Can I bring my dog to the terrace?",
        extract_query_facets("Can I bring my dog to the terrace?"),
    )

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
    constraints = _extract_constraints(
        "What options do you have for someone avoiding gluten?",
        extract_query_facets("What options do you have for someone avoiding gluten?"),
    )

    # Nut allergy evidence only
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
    constraints = _extract_constraints(
        "What options do you have for someone avoiding gluten?",
        extract_query_facets("What options do you have for someone avoiding gluten?"),
    )

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


def test_saturday_only_evidence_does_not_satisfy_both_days_constraint():
    """Saturday+Sunday question must NOT be satisfied by Saturday-only evidence."""
    q = "What are the opening hours on Saturday and Sunday?"
    constraints = _extract_constraints(q, extract_query_facets(q))

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
    assert len(missing) >= 1, "Saturday-only evidence must not satisfy Saturday+Sunday constraint"


def test_both_days_evidence_satisfies_both_days_constraint():
    """Saturday+Sunday evidence satisfies Saturday+Sunday constraint."""
    q = "What are the opening hours on Saturday and Sunday?"
    constraints = _extract_constraints(q, extract_query_facets(q))

    trace = [
        {
            "execution_status": "success",
            "is_error": False,
            "observation": {
                "content": "Saturday: 9:00 to 22:00  Sunday: 9:00 to 17:00",
            },
        }
    ]

    covered, missing = check_evidence_coverage_with_constraints(constraints, trace)
    assert len(covered) == 1
    assert len(missing) == 0


def test_failed_calls_with_relevant_text_not_counted():
    """Failed calls (is_error=True) must not count as evidence even if text contains keywords."""
    constraints = _extract_constraints(
        "Can I bring my dog to the terrace?",
        extract_query_facets("Can I bring my dog to the terrace?"),
    )

    trace = [
        {
            "execution_status": "error",
            "is_error": True,
            "observation": {
                "content": "Error: Dogs are welcome on the small terrace (cached). ChromaDB connection failed.",
            },
        }
    ]

    covered, missing = check_evidence_coverage_with_constraints(constraints, trace)
    assert len(covered) == 0
    assert len(missing) >= 1


def test_per_observation_checking_not_combined():
    """Keywords split across TWO observations must NOT collectively satisfy a single constraint."""
    q = "What are the opening hours on Saturday and Sunday?"
    constraints = _extract_constraints(q, extract_query_facets(q))

    # Saturday in one observation, Sunday in another — should NOT satisfy the combined constraint
    trace = [
        {
            "execution_status": "success",
            "is_error": False,
            "observation": {"content": "Saturday: 9:00 to 22:00"},
        },
        {
            "execution_status": "success",
            "is_error": False,
            "observation": {"content": "Sunday: 9:00 to 17:00"},
        },
    ]

    covered, missing = check_evidence_coverage_with_constraints(constraints, trace)
    assert len(missing) >= 1, "Split evidence across observations must not satisfy combined constraint"


def test_three_facets_one_retrieved():
    """Three-facet query with only one facet retrieved leaves two missing."""
    q = "Can I bring my dog, what food do you have, and do you have gluten-free options?"
    facets = extract_query_facets(q)
    assert len(facets) >= 3  # pet_policy, food_menu, allergens_diet

    constraints = _extract_constraints(q, facets)
    assert len(constraints) >= 3

    # Only pet policy evidence
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
    assert len(missing) >= 2
