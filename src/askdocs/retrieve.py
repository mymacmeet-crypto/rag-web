"""Hybrid retrieval: dense vectors fused with BM25.

Neither retriever is good enough alone on a documentation corpus:

  * dense vectors handle paraphrase ("how do I make it fast" -> a page about
    performance) but are weak on rare literal strings
  * BM25 nails `AEM_BACKEND_FETCH_FAILED` and `helix-query.yaml` but misses
    anything worded differently

They are combined with Reciprocal Rank Fusion, which merges by *rank* rather
than score. That matters because a cosine similarity of 0.82 and a BM25 score
of 14.3 are not comparable quantities - any attempt to weight them directly
means inventing a scale factor and re-tuning it whenever either side changes.
RRF needs no such constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunk import Chunk
from .store import Index


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None
    keyword_rank: int | None

    @property
    def found_by(self) -> str:
        if self.dense_rank is not None and self.keyword_rank is not None:
            return "both"
        return "vector" if self.dense_rank is not None else "keyword"


def _top_ranks(scores: np.ndarray, limit: int) -> dict[int, int]:
    """Map chunk index -> 0-based rank, best first, for the top `limit`."""
    if scores.size == 0:
        return {}
    limit = min(limit, scores.size)
    # argpartition finds the top-k without sorting the whole array; the slice
    # is then sorted so ranks are exact.
    candidates = np.argpartition(-scores, limit - 1)[:limit]
    ordered = candidates[np.argsort(-scores[candidates])]
    return {int(index): rank for rank, index in enumerate(ordered) if scores[index] > 0}


def search(
    index: Index,
    query: str,
    query_vector: np.ndarray,
    *,
    top_k: int = 5,
    candidates: int = 25,
    rrf_k: int = 60,
) -> list[Hit]:
    dense = _top_ranks(index.dense_scores(query_vector), candidates)
    keyword = _top_ranks(index.keyword_scores(query), candidates)

    fused: dict[int, float] = {}
    for ranks in (dense, keyword):
        for position, rank in ranks.items():
            fused[position] = fused.get(position, 0.0) + 1.0 / (rrf_k + rank + 1)

    ordered = sorted(fused.items(), key=lambda item: -item[1])[:top_k]
    return [
        Hit(
            chunk=index.chunks[position],
            score=score,
            dense_rank=dense.get(position),
            keyword_rank=keyword.get(position),
        )
        for position, score in ordered
    ]
