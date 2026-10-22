from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    text: str


def chunk_text(
    text: str,
    source: str = "<memory>",
    chunk_size: int = 80,
    overlap: int = 15,
) -> list[Chunk]:
    """Keep paragraphs separate and split oversized paragraphs with overlap."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in re.split(r"\n\s*\n", text.strip())
        if paragraph.strip()
    ]

    chunks = []

    def add_chunk(words: list[str]) -> None:
        chunks.append(
            Chunk(
                source=source,
                index=len(chunks),
                text=" ".join(words),
            )
        )

    for paragraph in paragraphs:
        words = paragraph.split()

        if len(words) <= chunk_size:
            add_chunk(words)
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            add_chunk(words[start:end])

            if end == len(words):
                break

            start = end - overlap

    return chunks


def load_chunks(
    knowledge_dir: str | Path,
    chunk_size: int = 80,
    overlap: int = 15,
) -> list[Chunk]:
    """Load .txt knowledge files and split them into chunks."""
    directory = Path(knowledge_dir)

    if not directory.is_dir():
        raise FileNotFoundError(f"Knowledge directory not found: {directory}")

    chunks = []

    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8-sig").strip()
        if text:
            chunks.extend(
                chunk_text(
                    text,
                    source=path.name,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )

    return chunks


def _terms(text: str) -> set[str]:
    """Return normalized word terms, excluding common English stop words."""
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
        "to", "what", "when", "where", "which", "while", "with",
    }
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if term not in stop_words
    }


def retrieve(
    query: str,
    chunks: list[Chunk],
    top_k: int = 3,
) -> list[tuple[Chunk, int]]:
    """Return the highest-scoring chunks as (chunk, score) pairs."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_terms = _terms(query)
    if not query_terms:
        return []

    scored = []

    for chunk in chunks:
        chunk_terms = _terms(chunk.text)
        score = len(query_terms & chunk_terms)
        if score > 0:
            scored.append((chunk, score))

    scored.sort(key=lambda item: (-item[1], item[0].source, item[0].index))
    return scored[:top_k]


def recall_at_k(
    retrieved: list[Chunk],
    relevant: set[tuple[str, int]],
) -> float:
    """Return the fraction of known relevant chunks retrieved."""
    if not relevant:
        raise ValueError("The relevant set must not be empty.")

    retrieved_ids = {(chunk.source, chunk.index) for chunk in retrieved}
    return len(retrieved_ids & relevant) / len(relevant)


def precision_at_k(
    retrieved: list[Chunk],
    relevant: set[tuple[str, int]],
) -> float:
    """Return the fraction of retrieved chunks that are relevant."""
    if not retrieved:
        return 0.0

    retrieved_ids = {(chunk.source, chunk.index) for chunk in retrieved}
    return len(retrieved_ids & relevant) / len(retrieved)
