"""Tests for the index, BM25, and hybrid fusion."""

import numpy as np
import pytest

from askdocs.chunk import Chunk
from askdocs.retrieve import search
from askdocs.store import BM25, Index, Manifest, normalize_rows, tokenize


def make_chunk(number: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc:{number:04d}",
        doc_id="doc",
        url=f"https://example.com/p{number}",
        title=f"Page {number}",
        section="docs",
        breadcrumb=f"Page {number}",
        anchor=None,
        text=text,
        tokens=len(text) // 4 or 1,
        doc_sha256="hash",
        source="test",
    )


def make_index(texts, dims=8):
    chunks = [make_chunk(i, t) for i, t in enumerate(texts)]
    rng = np.random.default_rng(0)
    vectors = normalize_rows(rng.random((len(chunks), dims), dtype=np.float32))
    manifest = Manifest(
        format_version=1,
        embed_model="test-model",
        dimensions=dims,
        chunk_count=len(chunks),
        source="test",
        dataset="none",
        built_at="2026-01-01T00:00:00+00:00",
        chunk_config={},
        doc_hashes={"doc": "hash"},
    )
    return Index(chunks, vectors, manifest)


# -- tokenizer ------------------------------------------------------------

def test_identifier_is_indexed_whole_and_in_parts():
    """Both 'AEM_BACKEND_FETCH_FAILED' and 'backend fetch' must find it."""
    tokens = tokenize("AEM_BACKEND_FETCH_FAILED")
    assert "aem_backend_fetch_failed" in tokens
    assert "backend" in tokens and "fetch" in tokens


# -- bm25 -----------------------------------------------------------------

def test_bm25_finds_a_rare_literal():
    docs = [
        tokenize("the quick brown fox jumps"),
        tokenize("error AEM_BACKEND_FETCH_FAILED means the fetch failed"),
        tokenize("general prose about performance"),
    ]
    scores = BM25(docs).score(tokenize("AEM_BACKEND_FETCH_FAILED"))
    assert int(np.argmax(scores)) == 1


def test_bm25_scores_zero_for_absent_terms():
    scores = BM25([tokenize("alpha beta")]).score(tokenize("nonexistent"))
    assert scores.sum() == 0


def test_bm25_handles_empty_corpus():
    assert BM25([]).score(tokenize("anything")).size == 0


# -- index ----------------------------------------------------------------

def test_mismatched_chunks_and_vectors_is_rejected():
    with pytest.raises(ValueError):
        Index(
            [make_chunk(0, "a")],
            np.zeros((3, 4), dtype=np.float32),
            Manifest(1, "m", 4, 1, "s", "d", "t", {}, {}),
        )


def test_save_and_load_round_trip(tmp_path):
    index = make_index(["alpha text", "beta text"])
    path = tmp_path / "index"
    index.save(path)

    loaded = Index.load(path)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in index.chunks]
    assert np.allclose(loaded.vectors, index.vectors)
    assert loaded.manifest.embed_model == "test-model"


def test_save_is_atomic_and_leaves_no_temp_dirs(tmp_path):
    path = tmp_path / "index"
    make_index(["one"]).save(path)
    make_index(["one", "two"]).save(path)
    assert Index.load(path).manifest.chunk_count == 2
    assert not (tmp_path / "index.tmp").exists()
    assert not (tmp_path / "index.old").exists()


def test_load_without_an_index_explains_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="askdocs index"):
        Index.load(tmp_path / "missing")


def test_wrong_format_version_is_rejected(tmp_path):
    path = tmp_path / "index"
    index = make_index(["one"])
    index.manifest.format_version = 99
    index.save(path)
    with pytest.raises(ValueError, match="Rebuild"):
        Index.load(path)


# -- retrieval ------------------------------------------------------------

def test_hybrid_finds_a_literal_the_vectors_would_miss():
    index = make_index([
        "general talk about making websites fast",
        "AEM_BACKEND_FETCH_FAILED means the backend could not be reached",
        "unrelated prose about authoring documents",
    ])
    # A vector orthogonal to everything, so only BM25 can contribute.
    hits = search(index, "AEM_BACKEND_FETCH_FAILED", np.zeros(8, dtype=np.float32), top_k=1)
    assert hits[0].chunk.chunk_id == "doc:0001"
    assert hits[0].keyword_rank is not None


def test_search_returns_at_most_top_k():
    index = make_index([f"text number {i}" for i in range(20)])
    hits = search(index, "text", np.zeros(8, dtype=np.float32), top_k=3)
    assert len(hits) <= 3


def test_found_by_reports_which_retriever_fired():
    index = make_index(["alpha unique_term_here", "beta"])
    hits = search(index, "unique_term_here", np.zeros(8, dtype=np.float32), top_k=2)
    assert hits[0].found_by in {"keyword", "both"}


def test_empty_index_returns_nothing():
    index = make_index([])
    assert search(index, "anything", np.zeros(8, dtype=np.float32)) == []
