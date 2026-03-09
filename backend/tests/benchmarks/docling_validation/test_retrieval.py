from paperchat_backend.benchmarks.docling_validation.models import NormalizedChunk
from paperchat_backend.benchmarks.docling_validation.retrieval import (
    cosine_similarity,
    rank_chunks,
    term_frequencies,
    tokenize,
)


def make_chunk(*, chunk_index: int, retrieval_text: str) -> NormalizedChunk:
    return NormalizedChunk(
        doc_id="doc-1",
        chunker_id="hybrid",
        chunk_index=chunk_index,
        text=retrieval_text,
        retrieval_text=retrieval_text,
        page_numbers=(1,),
        headings=("Intro",),
        warning_codes=(),
    )


def test_tokenize_normalizes_case_and_punctuation():
    assert tokenize("Attention, Attention!") == ("attention", "attention")


def test_cosine_similarity_returns_zero_for_disjoint_vectors():
    assert cosine_similarity(term_frequencies("alpha"), term_frequencies("beta")) == 0.0


def test_rank_chunks_orders_by_similarity_then_chunk_index():
    chunks = (
        make_chunk(chunk_index=1, retrieval_text="query terms and more"),
        make_chunk(chunk_index=0, retrieval_text="query terms"),
        make_chunk(chunk_index=2, retrieval_text="different content"),
    )

    ranked = rank_chunks("query terms", chunks)

    assert [item.chunk.chunk_index for item in ranked] == [0, 1, 2]
