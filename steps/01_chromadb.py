
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "cafe_knowledge"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


def main() -> None:

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    doc_files = sorted(DOCUMENTS_DIR.glob("*.txt"))

    if not doc_files:
        raise FileNotFoundError("No café documents found.")

    client = chromadb.PersistentClient(
        path=str(PERSIST_DIRECTORY)
    )

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu",
    )

    # Rebuild the baseline collection using E5.
    existing = {
        item.name if hasattr(item, "name") else item
        for item in client.list_collections()
    }

    if COLLECTION_NAME in existing:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
    )

    documents = []
    ids = []
    metadatas = []

    for file_path in doc_files:

        filename = file_path.name

        content = file_path.read_text(
            encoding="utf-8"
        ).strip()

        # Baseline: one non-empty line per chunk.
        chunks = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

        for idx, chunk in enumerate(chunks):

            documents.append(f"passage: {chunk}")

            ids.append(
                f"{filename}_chunk_{idx}"
            )

            metadatas.append({
                "source": filename,
                "chunk_index": idx,
            })

    collection.upsert(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )

    print("\n========== BASELINE INDEXING COMPLETE ==========")
    print(f"Database: {PERSIST_DIRECTORY}")
    print(f"Collection: {collection.name}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Source files: {len(doc_files)}")
    print(f"Indexed chunks: {collection.count()}")


if __name__ == "__main__":
    main()
