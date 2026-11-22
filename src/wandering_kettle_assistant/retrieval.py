"""ChromaDB knowledge retrieval service for Wandering Kettle café."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from wandering_kettle_assistant.citations import Citation, retrieve_with_citations
from wandering_kettle_assistant.validation import validate_search_args, RejectionResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "cafe_knowledge_overlapping"
BASELINE_COLLECTION = "cafe_knowledge"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

_collection_instance = None


def get_chroma_collection():
    """Retrieve or initialize the ChromaDB collection singleton."""
    global _collection_instance
    if _collection_instance is not None:
        return _collection_instance

    client = chromadb.PersistentClient(path=str(PERSIST_DIRECTORY))
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu",
    )

    existing_collections = {
        item.name if hasattr(item, "name") else item
        for item in client.list_collections()
    }

    target_name = COLLECTION_NAME if COLLECTION_NAME in existing_collections else BASELINE_COLLECTION
    if target_name not in existing_collections:
        raise RuntimeError(
            f"ChromaDB collection '{COLLECTION_NAME}' not found. Run ingestion first."
        )

    _collection_instance = client.get_collection(
        name=target_name,
        embedding_function=embedding_function,
    )
    return _collection_instance


def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """
    Search the Wandering Kettle café knowledge base.

    Returns:
        Formatted string containing retrieved passages and source metadata with citations.
    """
    validation = validate_search_args({"query": query, "top_k": top_k})
    if isinstance(validation, RejectionResult):
        error_msgs = "; ".join(err.message for err in validation.errors)
        return f"Argument validation failed: {error_msgs}"

    clean_query = validation.query
    sanitized_top_k = validation.top_k

    try:
        collection = get_chroma_collection()
    except Exception as exc:
        return f"Knowledge base connection error: {exc}"

    # Use multi-query variant expansion for robust matching
    variants = [clean_query]
    if "hours" in clean_query.lower() or "open" in clean_query.lower():
        variants.append("opening hours schedule weekend weekdays")
    elif "pet" in clean_query.lower() or "cat" in clean_query.lower() or "dog" in clean_query.lower():
        variants.append("animals dogs terrace outside patio")
    elif "wifi" in clean_query.lower() or "laptop" in clean_query.lower():
        variants.append("wireless internet power outlets laptop work")
    elif "vegan" in clean_query.lower() or "plant" in clean_query.lower():
        variants.append("plant based animal products dairy milk")

    citations: List[Citation] = retrieve_with_citations(
        query_variants=variants,
        top_k=sanitized_top_k,
        collection=collection,
    )

    if not citations:
        return f"No records found for query: '{clean_query}'."

    output_lines = [
        f"Retrieved {len(citations)} records for query: '{clean_query}':\n"
    ]
    for idx, c in enumerate(citations, start=1):
        output_lines.append(
            f"[{idx}] Source: {c.source} | Chunk: {c.chunk_id} | Score: {c.score:.3f}\n"
            f"    \"{c.text}\"\n"
        )

    return "\n".join(output_lines)


def lookup_policy(topic: str) -> str:
    """
    Lookup specific café policies such as pet policy, outdoor seating, Wi-Fi, laptops, or hours.
    """
    query = f"policy regarding {topic}"
    return search_knowledge_base(query=query, top_k=3)


def search_menu(category: str = "all", include_allergens: bool = True) -> str:
    """
    Search menu items, seasonal offerings, and allergen warnings.
    """
    query = f"menu items {category}"
    if include_allergens:
        query += " allergens dietary restrictions gluten nuts dairy"
    return search_knowledge_base(query=query, top_k=4)
