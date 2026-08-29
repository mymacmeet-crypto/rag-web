"""The evaluation harness - the thing that makes this production rather than a demo.

It measures one number above all: **recall@k**, the share of questions where
the correct page appears in the retrieved chunks.

That number is chosen deliberately. Almost every bad RAG answer is a retrieval
failure, not a generation failure - if the right passage never reaches the
prompt, no amount of prompt tuning can save the answer. Recall@k is also cheap
and deterministic to measure: it needs the embedding model but not the chat
model, so it runs in seconds and gives the same result every time.

`--refusal` additionally checks the opposite failure: questions the corpus
cannot answer, where a small local model's habit is to invent something
plausible. That one does need the chat model and is correspondingly slower.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .answer import build_prompt, stream_answer
from .grounding import looks_like_refusal
from .config import Config
from .ollama import OllamaClient
from .retrieve import search
from .store import Index
from .store import normalize_rows

import numpy as np


@dataclass
class Question:
    question: str
    # Any one of these URLs counts as correct. A question can legitimately be
    # answered by more than one page.
    expect_urls: list[str] = field(default_factory=list)
    # True for questions the corpus deliberately cannot answer.
    unanswerable: bool = False
    note: str = ""


@dataclass
class Result:
    question: Question
    hit_rank: int | None
    retrieved: list[str]

    @property
    def found(self) -> bool:
        return self.hit_rank is not None


def load_questions(path: Path) -> list[Question]:
    """Read the golden set.

    YAML if PyYAML is available, otherwise JSON - the file is a list of
    objects either way, and the tool must not gain a dependency for it.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"No evaluation set at {path}.\n"
            f"  Write 20-30 real questions with the URL that answers each."
        )
    raw = path.read_text(encoding="utf-8")

    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is needed for .yaml question sets; use .json instead."
            ) from error
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of questions")

    questions: list[Question] = []
    for entry in data:
        questions.append(
            Question(
                question=entry["question"],
                expect_urls=list(entry.get("expect_urls") or []),
                unanswerable=bool(entry.get("unanswerable")),
                note=entry.get("note", ""),
            )
        )
    return questions


def _matches(retrieved_url: str, expected: str) -> bool:
    """Compare URLs ignoring the anchor.

    Recall asks whether the right *page* was found. Whether the anchor is the
    ideal one is a separate, finer question - conflating them would make a
    correct retrieval look like a failure.
    """
    return retrieved_url.split("#")[0].rstrip("/") == expected.split("#")[0].rstrip("/")


def evaluate_retrieval(
    index: Index,
    client: OllamaClient,
    config: Config,
    questions: Sequence[Question],
) -> list[Result]:
    answerable = [q for q in questions if not q.unanswerable]
    if not answerable:
        return []

    vectors = normalize_rows(
        np.array(
            client.embed([q.question for q in answerable], config.embed.model),
            dtype=np.float32,
        )
    )

    results: list[Result] = []
    for question, vector in zip(answerable, vectors):
        hits = search(
            index,
            question.question,
            vector,
            top_k=config.retrieve.top_k,
            candidates=config.retrieve.candidates,
            rrf_k=config.retrieve.rrf_k,
        )
        urls = [hit.chunk.url for hit in hits]
        rank = next(
            (
                position
                for position, url in enumerate(urls)
                if any(_matches(url, expected) for expected in question.expect_urls)
            ),
            None,
        )
        results.append(Result(question=question, hit_rank=rank, retrieved=urls))
    return results


def recall(results: Sequence[Result]) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result.found) / len(results)


def mean_reciprocal_rank(results: Sequence[Result]) -> float:
    """How high up the correct page lands, not just whether it appears.

    Recall alone cannot distinguish 'first result' from 'fifth result', but
    that difference is most of the answer quality.
    """
    if not results:
        return 0.0
    total = sum(1 / (r.hit_rank + 1) for r in results if r.hit_rank is not None)
    return total / len(results)


@dataclass
class RefusalResult:
    question: Question
    answer: str
    refused: bool


def evaluate_refusal(
    index: Index,
    client: OllamaClient,
    config: Config,
    questions: Sequence[Question],
) -> list[RefusalResult]:
    """Check the model declines when the corpus has no answer."""
    unanswerable = [q for q in questions if q.unanswerable]
    if not unanswerable:
        return []

    client.require_model(config.generate.model)
    vectors = normalize_rows(
        np.array(
            client.embed([q.question for q in unanswerable], config.embed.model),
            dtype=np.float32,
        )
    )

    results: list[RefusalResult] = []
    for question, vector in zip(unanswerable, vectors):
        hits = search(
            index,
            question.question,
            vector,
            top_k=config.retrieve.top_k,
            candidates=config.retrieve.candidates,
            rrf_k=config.retrieve.rrf_k,
        )
        prompt = build_prompt(question.question, hits)
        answer = "".join(
            stream_answer(
                client,
                prompt,
                model=config.generate.model,
                temperature=config.generate.temperature,
                max_tokens=config.generate.max_tokens,
            )
        )
        results.append(
            RefusalResult(
                question=question,
                answer=answer.strip(),
                refused=looks_like_refusal(answer),
            )
        )
    return results
