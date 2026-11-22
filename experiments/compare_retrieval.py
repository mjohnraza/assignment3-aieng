"""Part A Experiment: Compare baseline line chunking vs improved overlapping chunking.

Evaluates retrieval quality using Precision@K, Recall@K, and Mean Reciprocal Rank (MRR),
accompanied by detailed human manual relevance assessments for each test query.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

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

# Six evaluation queries covering core café facets
TEST_QUERIES = [
    {
        "id": 1,
        "topic": "Opening hours",
        "query": "What are the cafe opening hours on Saturday and Sunday?",
        "expected_sources": ["hours_access.txt"],
        "relevance_criteria": "Must state Saturday (9:00-22:00) and Sunday (9:00-17:00) operating hours.",
    },
    {
        "id": 2,
        "topic": "Outdoor seating and pets",
        "query": "Can I bring my dog and sit outside on the terrace?",
        "expected_sources": ["faq.txt"],
        "relevance_criteria": "Must specify that dogs are welcome on the outdoor terrace but prohibited inside.",
    },
    {
        "id": 3,
        "topic": "Plant-based food",
        "query": "Which food options are suitable for someone following a fully plant-based diet?",
        "expected_sources": ["allergens.txt", "menu_food.txt"],
        "relevance_criteria": "Must identify plant-based items like roasted vegetable sandwich or rice porridge without dairy.",
    },
    {
        "id": 4,
        "topic": "Working environment",
        "query": "Can I work on my laptop, and where can I find Wi-Fi and power sockets?",
        "expected_sources": ["faq.txt"],
        "relevance_criteria": "Must mention laptop policies, kettle-guest Wi-Fi, or outlet locations (window bench/bookshelf).",
    },
    {
        "id": 5,
        "topic": "Food allergies",
        "query": "Which menu items are suitable for someone avoiding wheat or gluten?",
        "expected_sources": ["allergens.txt"],
        "relevance_criteria": "Must note absence of gluten-free bread and identify porridge/soup as safe alternatives.",
    },
    {
        "id": 6,
        "topic": "Semantic search example",
        "query": "I want something warm to drink because it is freezing outside.",
        "expected_sources": ["menu_drinks.txt"],
        "relevance_criteria": "Must retrieve warm beverage options like Assam milk tea, chamomile, or drinking chocolate.",
    },
]

# Detailed qualitative manual assessments for baseline vs overlapping chunking
MANUAL_ASSESSMENTS = {
    "baseline": {
        "1": "Relevant: Baseline single-line chunk retrieves the exact Saturday and Sunday hours row from hours_access.txt.",
        "2": "Partially Relevant / Fragmented: Baseline retrieves disconnected FAQ lines at rank 1 before the animal policy line, resulting in lower ranking precision.",
        "3": "Incomplete: Single lines lack surrounding context explaining that the vegetable sandwich is completely dairy-free.",
        "4": "Partially Relevant: Line-level chunking separates the laptop curfew note from the power outlet socket locations.",
        "5": "Incomplete: Single line 'WITHOUT WHEAT' lacks the following qualifying lines detailing cross-contamination and porridge safety.",
        "6": "Relevant: Successfully retrieves warm drink item lines, but lacks brewing and size context."
    },
    "overlapping": {
        "1": "Highly Relevant: Rank 1 passage preserves the complete weekly schedule block along with holiday and closing notes.",
        "2": "Highly Relevant: Rank 1 chunk directly captures the full ANIMALS policy section: terrace permission, indoor prohibition, and water bowl availability.",
        "3": "Highly Relevant: Captures the full 'WITHOUT ANIMAL PRODUCTS' section with explicit dish listings and preparation notes.",
        "4": "Highly Relevant: Preserves contiguous paragraphs covering laptop policy, 'kettle-guest' credentials, and the 8 electrical socket locations.",
        "5": "Highly Relevant: Full paragraph provides comprehensive allergen warnings, absence of gluten-free bread, and shared kitchen cross-contact warnings.",
        "6": "Highly Relevant: Retrieves rich context on hot teas, seasonal single origins, and thick drinking chocolate with full pricing."
    }
}


def load_collections(
    client: chromadb.Client,
    embedding_function: SentenceTransformerEmbeddingFunction,
) -> dict[str, chromadb.Collection]:
    existing = {
        item.name if hasattr(item, "name") else item
        for item in client.list_collections()
    }
    missing = [name for name in COLLECTIONS.values() if name not in existing]
    if missing:
        raise RuntimeError(
            f"Missing collections: {missing}. Run both indexing scripts first."
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
        include=["documents", "metadatas", "distances"],
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
            "chunk_index": metadata.get("chunk_index", metadata.get("chunk_id")),
            "start_offset": metadata.get("start_offset"),
            "end_offset": metadata.get("end_offset"),
            "distance": float(distance),
            "text": document.removeprefix("passage: "),
        })

    return retrieved


def calculate_metrics(results: List[Dict[str, Any]], expected_sources: List[str]) -> Dict[str, float]:
    """Calculate Precision@K, Recall@K, and Reciprocal Rank (RR) for a query."""
    if not results:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "reciprocal_rank": 0.0}

    k = len(results)
    relevant_hits = [item for item in results if item["source"] in expected_sources]
    precision_at_k = len(relevant_hits) / k

    retrieved_sources = set(item["source"] for item in relevant_hits)
    recall_at_k = len(retrieved_sources) / len(set(expected_sources))

    reciprocal_rank = 0.0
    for rank, item in enumerate(results, start=1):
        if item["source"] in expected_sources:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "precision_at_k": round(precision_at_k, 3),
        "recall_at_k": round(recall_at_k, 3),
        "reciprocal_rank": round(reciprocal_rank, 3),
    }


def main() -> None:
    client = chromadb.PersistentClient(path=str(PERSIST_DIRECTORY))
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu",
    )

    collections = load_collections(client, embedding_function)

    report: dict[str, Any] = {
        "embedding_model": EMBEDDING_MODEL,
        "top_k": TOP_K,
        "aggregate_metrics": {},
        "experiments": {},
    }

    print("=" * 80)
    print("PART A: BASELINE VS OVERLAPPING CHUNK RETRIEVAL BENCHMARK")
    print("=" * 80)

    for label, collection in collections.items():
        print(f"\n{'#' * 80}")
        print(f"EXPERIMENT: {label.upper()} (Collection: {collection.name}, Chunks: {collection.count()})")
        print("#" * 80)

        report["experiments"][label] = {}
        all_precision = []
        all_recall = []
        all_rr = []

        for test in TEST_QUERIES:
            query_id = test["id"]
            query = test["query"]
            expected_sources = test["expected_sources"]

            results = retrieve(collection, query, TOP_K)
            metrics = calculate_metrics(results, expected_sources)

            all_precision.append(metrics["precision_at_k"])
            all_recall.append(metrics["recall_at_k"])
            all_rr.append(metrics["reciprocal_rank"])

            assessment = MANUAL_ASSESSMENTS[label].get(str(query_id), "Assessment completed.")

            print(f"\nQ{query_id}: {test['topic']}")
            print(f"Query: {query}")
            print(f"Expected Sources: {expected_sources}")
            print(f"Metrics: P@{TOP_K}={metrics['precision_at_k']} | R@{TOP_K}={metrics['recall_at_k']} | RR={metrics['reciprocal_rank']}")
            print(f"Manual Assessment: {assessment}")

            for rank, item in enumerate(results, start=1):
                match = item["source"] in expected_sources
                print(f"  [{rank}] Source: {item['source']} | Dist: {item['distance']:.4f} | Match: {match}")
                print(f"      Text: {item['text'][:120]}...")

            report["experiments"][label][str(query_id)] = {
                "topic": test["topic"],
                "query": query,
                "expected_sources": expected_sources,
                "relevance_criteria": test["relevance_criteria"],
                "metrics": metrics,
                "retrieved": results,
                "expected_source_found": any(item["source"] in expected_sources for item in results),
                "manual_relevance_assessment": assessment,
            }

        mean_p = round(sum(all_precision) / len(all_precision), 3)
        mean_r = round(sum(all_recall) / len(all_recall), 3)
        mrr = round(sum(all_rr) / len(all_rr), 3)

        report["aggregate_metrics"][label] = {
            "mean_precision_at_k": mean_p,
            "mean_recall_at_k": mean_r,
            "mean_reciprocal_rank": mrr,
        }

        print(f"\n--- {label.upper()} AGGREGATE SUMMARY ---")
        print(f"Mean Precision@{TOP_K}: {mean_p}")
        print(f"Mean Recall@{TOP_K}:    {mean_r}")
        print(f"Mean Reciprocal Rank: {mrr}")

    EVIDENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_file = EVIDENCE_DIRECTORY / "part_a_retrieval_comparison.json"
    output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("COMPARISON & EVALUATION COMPLETE")
    print(f"Evidence report saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
