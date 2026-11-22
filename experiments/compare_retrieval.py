
import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_db"
EVIDENCE_DIRECTORY = PROJECT_ROOT / "evidence"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

TOP_K = 3

COLLECTIONS = {
    "baseline": "cafe_knowledge",
    "overlapping": "cafe_knowledge_overlapping",
}

# Five assignment evaluation questions plus one semantic-search example.
TEST_QUERIES = [
    {
        "id": 1,
        "topic": "Opening hours",
        "query": "What are the cafe opening hours on Saturday and Sunday?",
        "expected_sources": ["hours_access.txt"],
    },
    {
        "id": 2,
        "topic": "Outdoor seating and pets",
        "query": "Can I bring my dog and sit outside on the terrace?",
        "expected_sources": ["faq.txt"],
    },
    {
        "id": 3,
        "topic": "Plant-based food",
        "query": "Which food options are suitable for someone following a fully plant-based diet?",
        "expected_sources": [
            "allergens.txt",
            "menu_food.txt",
        ],
    },
    {
        "id": 4,
        "topic": "Working environment",
        "query": "Can I work on my laptop, and where can I find Wi-Fi and power sockets?",
        "expected_sources": ["faq.txt"],
    },
    {
        "id": 5,
        "topic": "Food allergies",
        "query": "Which menu items are suitable for someone avoiding wheat or gluten?",
        "expected_sources": ["allergens.txt"],
    },
    {
        "id": 6,
        "topic": "Semantic search example",
        "query": "I want something warm to drink because it is freezing outside.",
        "expected_sources": ["menu_drinks.txt"],
    },
]


def load_collections(
    client: chromadb.Client,
    embedding_function: SentenceTransformerEmbeddingFunction,
) -> dict[str, chromadb.Collection]:

    existing = {
        item.name if hasattr(item, "name") else item
        for item in client.list_collections()
    }

    missing = [
        name
        for name in COLLECTIONS.values()
        if name not in existing
    ]

    if missing:
        raise RuntimeError(
            f"Missing collections: {missing}. "
            "Run both indexing scripts first."
        )

    return {
        label: client.get_collection(
            name=name,
            embedding_function=embedding_function,
        )
        for label, name in COLLECTIONS.items()
    }


def retrieve(
    collection: chromadb.Collection,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:

    count = collection.count()

    if count == 0:
        return []

    results = collection.query(
        query_texts=[f"query: {query}"],
        n_results=min(top_k, count),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved = []

    for doc_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):

        retrieved.append({
            "id": doc_id,
            "source": metadata.get("source", "unknown"),
            "chunk_index": metadata.get(
                "chunk_index",
                metadata.get("chunk_id"),
            ),
            "start_offset": metadata.get("start_offset"),
            "end_offset": metadata.get("end_offset"),
            "distance": float(distance),
            "text": document.removeprefix("passage: "),
        })

    return retrieved


def main() -> None:

    client = chromadb.PersistentClient(
        path=str(PERSIST_DIRECTORY)
    )

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu",
    )

    collections = load_collections(
        client,
        embedding_function,
    )

    report: dict[str, Any] = {
        "embedding_model": EMBEDDING_MODEL,
        "top_k": TOP_K,
        "experiments": {},
    }

    print("=" * 80)
    print("PART A: BASELINE VS OVERLAPPING CHUNK RETRIEVAL")
    print("=" * 80)

    for label, collection in collections.items():

        print(f"\n{'#' * 80}")
        print(f"EXPERIMENT: {label.upper()}")
        print(f"COLLECTION: {collection.name}")
        print(f"RECORD COUNT: {collection.count()}")
        print("#" * 80)

        report["experiments"][label] = {}

        for test in TEST_QUERIES:

            query_id = test["id"]
            query = test["query"]
            expected_sources = test["expected_sources"]

            results = retrieve(
                collection,
                query,
                TOP_K,
            )

            print(f"\nQUESTION {query_id}: {test['topic']}")
            print(f"Query: {query}")
            print(f"Expected sources: {expected_sources}")

            for rank, item in enumerate(results, start=1):

                source_match = (
                    item["source"] in expected_sources
                )

                print(f"\n  Result #{rank}")
                print(f"  ID: {item['id']}")
                print(f"  Source: {item['source']}")
                print(f"  Distance: {item['distance']:.4f}")
                print(f"  Expected source match: {source_match}")
                print(f"  Text: {item['text']}")

            report["experiments"][label][str(query_id)] = {
                "topic": test["topic"],
                "query": query,
                "expected_sources": expected_sources,
                "retrieved": results,
                "expected_source_found": any(
                    item["source"] in expected_sources
                    for item in results
                ),
                "manual_relevance_assessment": "",
            }

    EVIDENCE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        EVIDENCE_DIRECTORY / "part_a_retrieval_comparison.json"
    )

    output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print(f"Evidence saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
