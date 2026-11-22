"""Unit tests for retrieval, citations, and ground-truth passage evaluation."""

import sys
from pathlib import Path

import pytest
from wandering_kettle_assistant.citations import Citation, format_citation
from wandering_kettle_assistant.retrieval import (
    search_knowledge_base,
    lookup_policy,
    search_menu,
)

# Ensure experiments module is importable
_root_dir = str(Path(__file__).resolve().parents[1])
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from experiments.compare_retrieval import GROUND_TRUTH_PASSAGES, calculate_metrics


def test_citation_formatting():
    c = format_citation(
        text="Dogs are welcome on the small terrace.",
        source="faq.txt",
        section="faq",
        chunk_id="faq.txt_chunk_4",
        score=0.92,
    )
    formatted = c.format()
    assert "[faq.txt § faq (#faq.txt_chunk_4)]" in formatted
    assert "Dogs are welcome" in formatted


def test_search_knowledge_base():
    result = search_knowledge_base("opening hours weekend", top_k=2)
    assert isinstance(result, str)
    assert "Retrieved" in result
    assert "hours" in result.lower() or "source:" in result.lower()


def test_lookup_policy():
    result = lookup_policy("pet policy")
    assert isinstance(result, str)
    assert "terrace" in result.lower() or "faq.txt" in result.lower()


def test_search_menu():
    result = search_menu(category="food", include_allergens=True)
    assert isinstance(result, str)
    assert "allergens" in result.lower() or "menu" in result.lower()


def test_overlapping_chunks_from_same_ground_truth_passage_deduplicate():
    """Verify that multiple overlapping chunks covering the SAME ground-truth passage
    do NOT artificially inflate passage recall or count as multiple passages.
    """
    # Simulate top-3 retrieved results where all 3 are overlapping chunks of the terrace policy
    overlapping_results = [
        {
            "id": "faq.txt_chunk_4",
            "source": "faq.txt",
            "text": "Dogs are welcome on the small terrace at the top of the stairs, but not inside the room.",
        },
        {
            "id": "faq.txt_chunk_5",
            "source": "faq.txt",
            "text": "ANIMALS: Dogs are welcome on the small terrace. There is a water bowl by the terrace door.",
        },
        {
            "id": "faq.txt_chunk_6",
            "source": "faq.txt",
            "text": "terrace at the top of the stairs. Assistance animals are welcome anywhere.",
        },
    ]

    expected_sources = ["faq.txt"]
    expected_passage_ids = ["gt_animals_terrace"]

    metrics = calculate_metrics(
        results=overlapping_results,
        expected_sources=expected_sources,
        expected_passage_ids=expected_passage_ids,
    )

    # All 3 chunks map to the single ground-truth passage gt_animals_terrace.
    # Deduplication ensures passage recall is exactly 1.0 (1 of 1 expected target passages retrieved),
    # NOT 3.0 or inflated.
    assert metrics["passage_recall_at_k"] == 1.0
    # Passage precision is 1 unique relevant passage out of 3 retrieved chunks = 0.333
    assert metrics["passage_precision_at_k"] == 0.333
    # Source recall is 1.0 (faq.txt retrieved)
    assert metrics["source_recall_at_k"] == 1.0
    # MRR is 1.0 because rank 1 was a relevant hit
    assert metrics["reciprocal_rank"] == 1.0


# ---------------------------------------------------------------------------
# Phase 1: Ground-Truth Matcher Strictness Tests
# ---------------------------------------------------------------------------

def test_weekend_hours_matcher_requires_both_days():
    """Saturday-only text must NOT match gt_hours_weekend; both Saturday AND Sunday required."""
    matcher = GROUND_TRUTH_PASSAGES["gt_hours_weekend"]["matcher"]

    # Only Saturday — must be rejected
    assert matcher("Saturday: 9:00 to 22:00") is False
    # Only Sunday — must be rejected
    assert matcher("Sunday: 9:00 to 17:00") is False
    # Both days — must be accepted
    assert matcher("Saturday: 9:00 to 22:00  Sunday: 9:00 to 17:00") is True


def test_plant_based_matcher_rejects_generic_vegetable():
    """Generic 'roasted vegetable sandwich' without plant-based context must be rejected."""
    matcher = GROUND_TRUTH_PASSAGES["gt_plant_based_sandwich"]["matcher"]

    # Generic vegetable + sandwich without 'plant based' or 'without animal products' — rejected
    assert matcher("Roasted vegetable and hummus sandwich - $5.60") is False
    # Explicit plant-based statement — accepted
    assert matcher("Fully plant based: the roasted vegetable and hummus sandwich") is True
    assert matcher("WITHOUT ANIMAL PRODUCTS: roasted vegetable and hummus sandwich, rice porridge") is True


def test_gluten_matcher_requires_limitation_and_alternative():
    """Must verify BOTH the no-gluten-free-bread limitation AND safe alternatives."""
    matcher = GROUND_TRUTH_PASSAGES["gt_gluten_wheat_alternatives"]["matcher"]

    # Only mentions gluten and bread without limitation — rejected
    assert matcher("Gluten free bread is available") is False
    # Mentions gluten and porridge but no bread limitation — rejected
    assert matcher("gluten-free porridge is available") is False
    # Correct: mentions limitation (no gluten free bread) and alternative
    assert matcher("We have no gluten free bread. The rice porridge and the soup of the day are usually safe.") is True


def test_laptop_wifi_sockets_are_distinct_passages():
    """Laptop policy, Wi-Fi credentials, and power sockets are separate ground-truth passages."""
    assert "gt_laptop_policy" in GROUND_TRUTH_PASSAGES
    assert "gt_wifi_credentials" in GROUND_TRUTH_PASSAGES
    assert "gt_power_sockets" in GROUND_TRUTH_PASSAGES

    laptop_matcher = GROUND_TRUTH_PASSAGES["gt_laptop_policy"]["matcher"]
    wifi_matcher = GROUND_TRUTH_PASSAGES["gt_wifi_credentials"]["matcher"]
    sockets_matcher = GROUND_TRUTH_PASSAGES["gt_power_sockets"]["matcher"]

    laptop_text = "On Friday and Saturday after six in the evening we put the laptops away and the room becomes a place for conversation instead."
    wifi_text = "The network is called kettle-guest and the password is printed on the back of the paper coaster."
    sockets_text = "There are six sockets along the window bench and two more beside the shelf of secondhand books."

    assert laptop_matcher(laptop_text) is True
    assert wifi_matcher(wifi_text) is True
    assert sockets_matcher(sockets_text) is True

    # Cross-check: laptop text should NOT match wifi or sockets matchers
    assert wifi_matcher(laptop_text) is False
    assert sockets_matcher(laptop_text) is False


def test_irrelevant_passage_from_expected_source_rejected():
    """A passage from the correct source but with irrelevant content must not match."""
    metrics = calculate_metrics(
        results=[
            {"id": "faq_chunk_99", "source": "faq.txt", "text": "We do not accept cheques or foreign currency."},
        ],
        expected_sources=["faq.txt"],
        expected_passage_ids=["gt_animals_terrace"],
    )
    # Source matches, but passage content is about payment, not animals
    assert metrics["source_precision_at_k"] == 1.0
    assert metrics["passage_recall_at_k"] == 0.0


def test_empty_results_metrics():
    """Empty result set produces all-zero metrics."""
    metrics = calculate_metrics(
        results=[],
        expected_sources=["faq.txt"],
        expected_passage_ids=["gt_animals_terrace"],
    )
    assert metrics["source_precision_at_k"] == 0.0
    assert metrics["passage_recall_at_k"] == 0.0
    assert metrics["reciprocal_rank"] == 0.0
