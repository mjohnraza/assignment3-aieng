"""Unit tests for ReAct parsing and agent loop."""

import pytest
from wandering_kettle_assistant.llm_client import parse_react_response
from wandering_kettle_assistant.react_agent import ReActAgent


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
