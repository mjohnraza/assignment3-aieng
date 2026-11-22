"""Citation data models and retrieval with citation tracking."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChunkMetadata:
    source: str
    section: str
    chunk_id: str
    category: Optional[str] = None


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    relevance_score: float


@dataclass
class Citation:
    text: str
    source: str
    section: str
    chunk_id: str
    score: float

    def format(self) -> str:
        return f"[{self.source} § {self.section} (#{self.chunk_id})] {self.text}"


def format_citation(text: str, source: str, section: str, chunk_id: str, score: float) -> Citation:
    return Citation(
        text=text,
        source=source,
        section=section,
        chunk_id=chunk_id,
        score=score,
    )


def retrieve_with_citations(
    query_variants: List[str],
    top_k: int,
    collection: Any,
) -> List[Citation]:
    """
    Perform multi-query expansion, search the collection, deduplicate,
    and return citations formatted with source metadata.
    """
    candidate_chunks: List[SearchResult] = []

    # Iterative multi-query expansion to broaden candidate coverage
    for query in query_variants:
        if not query.strip():
            continue
        try:
            results = collection.query(
                query_texts=[f"query: {query}"],
                n_results=min(top_k, max(1, collection.count())),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RuntimeError(f"ChromaDB collection query failed: {exc}") from exc

        if not results or not results.get("ids") or not results["ids"][0]:
            continue

        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Cosine distance to similarity score
            score = 1.0 - float(dist)
            source = meta.get("source", "unknown")
            section = meta.get("section", source.replace(".txt", ""))
            clean_text = doc.removeprefix("passage: ").strip()

            candidate_chunks.append(
                SearchResult(
                    chunk_id=str(doc_id),
                    text=clean_text,
                    metadata=ChunkMetadata(
                        source=source,
                        section=section,
                        chunk_id=str(doc_id),
                        category=meta.get("category"),
                    ),
                    relevance_score=score,
                )
            )

    # Deduplicate overlapping chunks by unique ID and rank by highest relevance
    unique_candidates: Dict[str, SearchResult] = {}
    for chunk in candidate_chunks:
        existing = unique_candidates.get(chunk.chunk_id)
        if existing is None or chunk.relevance_score > existing.relevance_score:
            unique_candidates[chunk.chunk_id] = chunk

    sorted_chunks = sorted(
        unique_candidates.values(),
        key=lambda c: c.relevance_score,
        reverse=True,
    )

    # Thread chunk metadata directly into citation evidence for the LLM
    citations = [
        format_citation(
            text=c.text,
            source=c.metadata.source,
            section=c.metadata.section,
            chunk_id=c.metadata.chunk_id,
            score=c.relevance_score,
        )
        for c in sorted_chunks[:top_k]
    ]

    return citations
