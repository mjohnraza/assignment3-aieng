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

    # Deduplication ensures passage recall is exactly 1.0 (1 of 1 expected target passages retrieved)
    assert metrics["passage_recall_at_k"] == 1.0
    # Passage precision is 1 unique relevant passage out of 3 retrieved chunks = 0.333
    assert metrics["passage_precision_at_k"] == 0.333
    # Source recall is 1.0 (faq.txt retrieved)
    assert metrics["source_recall_at_k"] == 1.0
    # MRR is 1.0 because rank 1 was a relevant hit
    assert metrics["reciprocal_rank"] == 1.0


# ---------------------------------------------------------------------------
# Part A: Comprehensive Ground-Truth Matcher Positive & Negative Tests
# ---------------------------------------------------------------------------

def test_gt_hours_weekend_matcher_positive_and_negative():
    """Verify exact Saturday (9:00-22:00) AND Sunday (9:00-17:00) hours are strictly required."""
    matcher = GROUND_TRUTH_PASSAGES["gt_hours_weekend"]["matcher"]

    # Positive: exact hours for both days
    valid_text = "Saturday: 9:00 to 22:00\nSunday: 9:00 to 17:00"
    assert matcher(valid_text) is True

    # Negative 1: Saturday only (Sunday missing)
    assert matcher("Saturday: 9:00 to 22:00") is False

    # Negative 2: Sunday only (Saturday missing)
    assert matcher("Sunday: 9:00 to 17:00") is False

    # Negative 3: Both days present but missing exact closing hours
    assert matcher("Saturday and Sunday: open from 9:00 onwards") is False


def test_gt_animals_terrace_matcher_positive_and_negative():
    """Verify outdoor terrace dog policy requires dogs/terrace/welcome."""
    matcher = GROUND_TRUTH_PASSAGES["gt_animals_terrace"]["matcher"]

    # Positive
    assert matcher("Dogs are welcome on the small terrace at the top of the stairs.") is True
    assert matcher("Animals: Dogs on terrace with water bowl by the door.") is True

    # Negative: Indoor only or cats
    assert matcher("Cats are allowed inside the main room.") is False
    assert matcher("Payment options include contactless cards.") is False


def test_gt_plant_based_sandwich_matcher_positive_and_negative():
    """Verify plant-based dietary passage requires explicit plant-based/dairy-free declaration."""
    matcher = GROUND_TRUTH_PASSAGES["gt_plant_based_sandwich"]["matcher"]

    # Positive
    assert matcher("Fully plant based: the roasted vegetable and hummus sandwich and rice porridge") is True
    assert matcher("WITHOUT ANIMAL PRODUCTS: roasted vegetable sandwich, rice porridge") is True

    # Negative: Generic vegetable sandwich without plant-based certification
    assert matcher("Roasted vegetable and hummus sandwich - $5.60 on sourdough") is False
    assert matcher("Ham and comte on baguette - $5.80") is False


def test_gt_gluten_wheat_alternatives_positive_and_negative():
    """Verify gluten matcher checks limitation, alternatives, AND stock/barley caveat."""
    matcher = GROUND_TRUTH_PASSAGES["gt_gluten_wheat_alternatives"]["matcher"]

    # Positive: full paragraph with limitation, safe porridge/soup, and barley stock caveat
    valid_text = (
        "WITHOUT WHEAT: We have no gluten free bread. The rice porridge and the soup of the day "
        "are usually safe, but the soup sometimes uses a stock that contains barley, so ask on the day."
    )
    assert matcher(valid_text) is True

    # Negative 1: Mentions gluten free bread without limitation
    assert matcher("We offer gluten free bread for all sandwiches.") is False

    # Negative 2: Mentions porridge without the bread limitation and barley caveat
    assert matcher("Rice porridge is safe without wheat.") is False


def test_gt_laptop_wifi_sockets_are_distinct():
    """Verify laptop curfew, Wi-Fi credentials, and power sockets do not cross-match."""
    laptop_m = GROUND_TRUTH_PASSAGES["gt_laptop_policy"]["matcher"]
    wifi_m = GROUND_TRUTH_PASSAGES["gt_wifi_credentials"]["matcher"]
    socket_m = GROUND_TRUTH_PASSAGES["gt_power_sockets"]["matcher"]

    laptop_text = "On Friday and Saturday after six in the evening we put the laptops away for conversation."
    wifi_text = "The network is called kettle-guest and the password is on the coaster."
    socket_text = "There are six sockets along the window bench and two more beside the book shelf."

    # Positives
    assert laptop_m(laptop_text) is True
    assert wifi_m(wifi_text) is True
    assert socket_m(socket_text) is True

    # Mutually exclusive negatives
    assert wifi_m(laptop_text) is False
    assert socket_m(laptop_text) is False
    assert laptop_m(wifi_text) is False
    assert socket_m(wifi_text) is False


def test_gt_warm_winter_drinks_positive_and_negative():
    """Verify winter beverage matcher checks warm drink items."""
    matcher = GROUND_TRUTH_PASSAGES["gt_warm_winter_drinks"]["matcher"]

    assert matcher("Thick drinking chocolate - $4.80 for cold days") is True
    assert matcher("Assam with steamed milk - $3.60 cup") is True
    assert matcher("Chamomile with honey - $3.50 tea") is True

    # Negative: cold brew or espresso house blend
    assert matcher("Cold brew, steeped eighteen hours - $4.00") is False


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
