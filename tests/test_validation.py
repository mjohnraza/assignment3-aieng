"""Unit tests for search argument and tool call validation across all café schemas."""

import pytest
from wandering_kettle_assistant.validation import (
    ValidatedSearchArgs,
    ValidatedPolicyArgs,
    ValidatedMenuArgs,
    ValidationError,
    RejectionResult,
    ToolCallDecision,
    ToolObservation,
    validate_search_args,
    validate_policy_args,
    validate_menu_args,
    validate_tool_arguments,
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


def test_validate_search_args_top_k_bounds():
    raw_high = {"query": "coffee", "top_k": 50}
    assert isinstance(validate_search_args(raw_high), RejectionResult)

    raw_low = {"query": "coffee", "top_k": 0}
    assert isinstance(validate_search_args(raw_low), RejectionResult)


def test_validate_policy_args():
    # Valid
    valid = validate_policy_args({"topic": "pet_policy"})
    assert isinstance(valid, ValidatedPolicyArgs)
    assert valid.topic == "pet_policy"

    # Missing topic
    missing = validate_policy_args({})
    assert isinstance(missing, RejectionResult)

    # Empty whitespace topic
    empty = validate_policy_args({"topic": "   "})
    assert isinstance(empty, RejectionResult)

    # Non-string topic
    non_str = validate_policy_args({"topic": 123})
    assert isinstance(non_str, RejectionResult)


def test_validate_menu_args():
    # Valid
    valid = validate_menu_args({"category": "food", "include_allergens": True})
    assert isinstance(valid, ValidatedMenuArgs)
    assert valid.category == "food"
    assert valid.include_allergens is True

    # Invalid category
    bad_cat = validate_menu_args({"category": "pizza"})
    assert isinstance(bad_cat, RejectionResult)

    # Boolean string coercion
    str_bool = validate_menu_args({"category": "drinks", "include_allergens": "false"})
    assert isinstance(str_bool, ValidatedMenuArgs)
    assert str_bool.include_allergens is False


def test_validate_tool_arguments_dispatch():
    # Search
    s_val = validate_tool_arguments("search", {"query": "tea", "top_k": 2})
    assert isinstance(s_val, dict)
    assert s_val["query"] == "tea"

    # Policy
    p_val = validate_tool_arguments("lookup_cafe_policy", {"topic": "wifi"})
    assert isinstance(p_val, dict)
    assert p_val["topic"] == "wifi"

    # Unknown tool
    u_val = validate_tool_arguments("fake_tool", {})
    assert isinstance(u_val, RejectionResult)


def test_validate_and_execute_tool_validation_error():
    decision = ToolCallDecision(
        tool_name="lookup_cafe_policy",
        arguments={},  # missing required 'topic'
        thought="testing missing topic",
        call_id="call_001",
    )
    obs = validate_and_execute_tool(decision, available_tools=["lookup_cafe_policy"])
    assert obs.is_error
    assert obs.status == "validation_error"
    assert "Argument schema validation failed" in obs.result


def test_validation_non_dict_arguments():
    """Verify that malformed arguments (None, list, int) do not raise AttributeError."""
    assert isinstance(validate_search_args(None), RejectionResult)
    assert isinstance(validate_search_args(["query", "tea"]), RejectionResult)
    assert isinstance(validate_policy_args(None), RejectionResult)
    assert isinstance(validate_policy_args("hours"), RejectionResult)
    assert isinstance(validate_menu_args(None), RejectionResult)
    assert isinstance(validate_menu_args(42), RejectionResult)
    assert isinstance(validate_tool_arguments("search", None), RejectionResult)
    assert isinstance(validate_tool_arguments("lookup_cafe_policy", [1, 2]), RejectionResult)


def test_verify_negative_response_safeguard():
    """Verify negative response assertion correctly distinguishes grounded absence from hallucinations."""
    from wandering_kettle_assistant.validation import verify_negative_response, classify_amenity_status

    # Case 1: Grounded honest response: parking confirmed unavailable, EV undocumented
    safe_text = "The café has no car park. EV charging is not mentioned in café documentation and cannot be confirmed."
    res = verify_negative_response(safe_text)
    assert res["is_safe"] is True
    assert res["parking_pass"] is True
    assert res["ev_pass"] is True
    assert res["details"]["parking_classification"] == "confirmed_unavailable"
    assert res["details"]["ev_classification"] == "undocumented"

    # Case 2: Hallucination case - states parking absent but hallucinates EV charging on site
    hallucination_ev = "The café has no car park, but EV charging stations are available on site."
    res_ev = verify_negative_response(hallucination_ev)
    assert res_ev["parking_pass"] is True
    assert res_ev["ev_pass"] is False
    assert res_ev["is_safe"] is False
    assert res_ev["details"]["ev_classification"] == "confirmed_available"

    # Case 3: False positive parking - claims customer parking is available
    hallucination_parking = "Dedicated customer parking is available. EV charging is not documented."
    res_p = verify_negative_response(hallucination_parking)
    assert res_p["parking_pass"] is False
    assert res_p["is_safe"] is False
    assert res_p["details"]["parking_classification"] == "confirmed_available"

    # Case 4: Rejection of bare 'unavailable' as evidence of undocumented EV charging
    # Merely saying EV charging is unavailable asserts confirmed absence rather than unmentioned status
    bare_unavailable = "The café has no car park. EV charging is unavailable."
    res_bare = verify_negative_response(bare_unavailable)
    assert res_bare["parking_pass"] is True
    assert res_bare["ev_pass"] is False  # Fails because undocumented phrasing is missing
    assert res_bare["is_safe"] is False
    assert res_bare["details"]["ev_classification"] == "confirmed_unavailable"

    # Off-site coin lot with explicit unmentioned EV passes
    coin_lot_text = "There is a coin lot behind the post office. Electric vehicle charging is unmentioned in café documents."
    res_coin = verify_negative_response(coin_lot_text)
    assert res_coin["is_safe"] is True
    assert res_coin["details"]["parking_classification"] == "confirmed_unavailable"
    assert res_coin["details"]["ev_classification"] == "undocumented"


def test_classify_amenity_status_all_four_quadrants():
    """Verify epistemic classification for available, unavailable, undocumented, and unknown."""
    from wandering_kettle_assistant.validation import classify_amenity_status

    # Parking tests
    assert classify_amenity_status("Dedicated customer parking is available.", "parking") == "confirmed_available"
    assert classify_amenity_status("The café has no parking.", "parking") == "confirmed_unavailable"
    assert classify_amenity_status("Parking cannot be confirmed.", "parking") == "undocumented"

    # EV charging tests
    assert classify_amenity_status("EV charging stations are available on site.", "ev") == "confirmed_available"
    assert classify_amenity_status("EV charging is not provided.", "ev") == "confirmed_unavailable"
    assert classify_amenity_status("EV charging availability cannot be confirmed in the docs.", "ev") == "undocumented"

