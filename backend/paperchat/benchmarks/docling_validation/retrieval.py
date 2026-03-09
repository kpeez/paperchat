import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Final

from paperchat.benchmarks.docling_validation.models import NormalizedChunk

TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: NormalizedChunk
    score: float


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(text.lower()))


def term_frequencies(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0

    numerator = sum(left[token] * right[token] for token in left.keys() & right.keys())
    if numerator == 0:
        return 0.0

    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    return numerator / (left_norm * right_norm)


def rank_chunks(query: str, chunks: tuple[NormalizedChunk, ...]) -> tuple[ScoredChunk, ...]:
    query_vector = term_frequencies(query)
    scored_chunks = [
        ScoredChunk(
            chunk=chunk,
            score=cosine_similarity(query_vector, term_frequencies(chunk.retrieval_text)),
        )
        for chunk in chunks
    ]
    ranked = sorted(
        scored_chunks,
        key=lambda scored_chunk: (
            -scored_chunk.score,
            scored_chunk.chunk.doc_id,
            scored_chunk.chunk.chunk_index,
        ),
    )
    return tuple(ranked)
