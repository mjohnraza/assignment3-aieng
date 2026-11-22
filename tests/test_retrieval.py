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
