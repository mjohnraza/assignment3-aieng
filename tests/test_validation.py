"""Unit tests for search argument and tool call validation."""

import pytest
from wandering_kettle_assistant.validation import (
    ValidatedSearchArgs,
    ValidationError,
    RejectionResult,
    ToolCallDecision,
    ToolObservation,
    validate_search_args,
    validate_and_execute_tool,
)


def test_validate_search_args_valid():
    raw = {"query": "matcha latte", "top_k": 3}
    result = validate_search_args(raw)
    assert isinstance(result, ValidatedSearchArgs)
    assert result.query == "matcha latte"
    assert result.top_k == 3


def test_validate_search_args_default_top_k():
    raw = {"query": "espresso"}
    result = validate_search_args(raw)
    assert isinstance(result, ValidatedSearchArgs)
    assert result.query == "espresso"
    assert result.top_k == 5


def test_validate_search_args_missing_query():
    raw = {"top_k": 3}
    result = validate_search_args(raw)
    assert isinstance(result, RejectionResult)
    assert not result.is_valid
    assert any(e.field == "query" for e in result.errors)


def test_validate_search_args_empty_query():
    raw = {"query": "   "}
    result = validate_search_args(raw)
    assert isinstance(result, RejectionResult)
    assert any(e.field == "query" for e in result.errors)


def test_validate_search_args_top_k_bounds():
    # Out of range top_k
    raw_high = {"query": "coffee", "top_k": 50}
    result_high = validate_search_args(raw_high)
    assert isinstance(result_high, RejectionResult)

    raw_low = {"query": "coffee", "top_k": 0}
    result_low = validate_search_args(raw_low)
    assert isinstance(result_low, RejectionResult)


def test_validate_and_execute_tool_unknown_tool():
    decision = ToolCallDecision(
        tool_name="hallucinated_tool",
        arguments={"x": 1},
        thought="testing unknown",
    )
    obs = validate_and_execute_tool(decision, available_tools=["search", "lookup_policy"])
    assert obs.is_error
    assert obs.status == "error"
    assert "Unknown tool" in obs.result


def test_validate_and_execute_tool_success():
    decision = ToolCallDecision(
        tool_name="search",
        arguments={"query": "tea"},
        thought="testing success",
    )
    obs = validate_and_execute_tool(
        decision,
        available_tools=["search"],
        executor_fn=lambda name, args: f"Executed {name} with query '{args['query']}'",
    )
    assert not obs.is_error
    assert obs.status == "success"
    assert "Executed search with query 'tea'" in obs.result
