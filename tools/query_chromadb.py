
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "cafe_knowledge_overlapping"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


def query_knowledge_base(
    query_text: str,
    top_k: int = 3,
) -> None:

    client = chromadb.PersistentClient(
        path=str(PERSIST_DIRECTORY)
    )

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu",
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    if collection.count() == 0:
        print("Collection is empty.")
        return

    results = collection.query(
        query_texts=[f"query: {query_text}"],
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    print(f"Query: {query_text}\n")

    for rank, (doc_id, doc, meta, distance) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):

        print(f"========== RESULT #{rank} ==========")
        print(f"ID: {doc_id}")
        print(f"Source: {meta['source']}")
        print(f"Distance: {distance:.4f}")
        print(
            f"Offsets: {meta['start_offset']}"
            f"-{meta['end_offset']}"
        )

        print(
            "Text:\n"
            f"{doc.removeprefix('passage: ')}\n"
        )


if __name__ == "__main__":

    sample_query = (
        "What plant-based beverage options "
        "do you have for warm drinks?"
    )

    query_knowledge_base(sample_query, top_k=3)
