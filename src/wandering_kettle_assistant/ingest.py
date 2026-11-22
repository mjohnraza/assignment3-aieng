
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "cafe_knowledge_overlapping"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

DOCUMENT_PREFIX = "passage: "

CHUNK_SIZE = 300
OVERLAP = 50


def generate_overlapping_chunks_with_offsets(
    text: str,
    source_name: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """
    Generate overlapping chunks with original character offsets.

    Returns:
        chunks: Extracted text chunks.
        metadatas: Source information and character offsets.
        ids: Unique chunk identifiers.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or chunk_size <= overlap:
        raise ValueError(
            "chunk_size must be strictly greater than overlap"
        )

    chunks: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    step = chunk_size - overlap
    start_offset = 0
    text_length = len(text)
    chunk_index = 0

    while start_offset < text_length:

        end_offset = min(
            start_offset + chunk_size,
            text_length,
        )

        chunk_text = text[start_offset:end_offset]

        chunk_id = f"{source_name}_chunk_{chunk_index}"

        chunks.append(chunk_text)

        metadatas.append({
            "source": source_name,
            "chunk_id": chunk_index,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "char_length": len(chunk_text),
        })

        ids.append(chunk_id)

        chunk_index += 1

        if end_offset >= text_length:
            break

        start_offset += step

    return chunks, metadatas, ids


def create_collection(
    client: chromadb.Client,
    embedding_function: SentenceTransformerEmbeddingFunction,
) -> chromadb.Collection:
    """Create a clean collection for the improved experiment."""

    existing = client.list_collections()

    existing_names = {
        item.name if hasattr(item, "name") else item
        for item in existing
    }

    if COLLECTION_NAME in existing_names:
        client.delete_collection(name=COLLECTION_NAME)

    return client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
    )


def chunk_and_index_document(
    collection: chromadb.Collection,
    document_text: str,
    source_name: str,
) -> int:
    """Generate chunks and store their E5 embeddings in ChromaDB."""

    chunks, metadatas, ids = (
        generate_overlapping_chunks_with_offsets(
            text=document_text,
            source_name=source_name,
        )
    )

    if not chunks:
        return 0

    # E5 passage prefix is applied before embedding.
    embedding_documents = [
        f"{DOCUMENT_PREFIX}{chunk}"
        for chunk in chunks
    ]

    collection.upsert(
        documents=embedding_documents,
        metadatas=metadatas,
        ids=ids,
    )

    print(
        f"Indexed {len(chunks)} chunks for {source_name}"
    )

    return len(chunks)


def main() -> None:
    """Load café documents and index overlapping E5 embeddings."""

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    doc_files = sorted(DOCUMENTS_DIR.glob("*.txt"))

    if not doc_files:
        raise FileNotFoundError(
            f"No documents found in {DOCUMENTS_DIR}"
        )

    # Initialize ChromaDB.
    client = chromadb.PersistentClient(
        path=str(PERSIST_DIRECTORY)
    )

    # Initialize the required E5 embedding model.
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu",
    )

    collection = create_collection(
        client,
        embedding_function,
    )

    total_chunks = 0

    for file_path in doc_files:

        filename = file_path.name

        content = file_path.read_text(
            encoding="utf-8"
        ).strip()

        chunk_count = chunk_and_index_document(
            collection=collection,
            document_text=content,
            source_name=filename,
        )

        total_chunks += chunk_count

    print("\n========== INGESTION COMPLETE ==========")
    print(f"Database: {PERSIST_DIRECTORY}")
    print(f"Collection: {collection.name}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Source files: {len(doc_files)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Stored records: {collection.count()}")


if __name__ == "__main__":
    main()
