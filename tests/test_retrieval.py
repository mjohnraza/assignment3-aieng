"""Unit tests for retrieval and citations."""

import pytest
from wandering_kettle_assistant.citations import Citation, format_citation
from wandering_kettle_assistant.retrieval import (
    search_knowledge_base,
    lookup_policy,
    search_menu,
)


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
    import sys
    from pathlib import Path
    root_dir = str(Path(__file__).resolve().parents[1])
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    from experiments.compare_retrieval import calculate_metrics

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

