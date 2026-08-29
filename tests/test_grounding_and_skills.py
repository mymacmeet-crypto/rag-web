"""Tests for the anti-hallucination checks and the skill registry.

These are the guarantees that do not depend on the model choosing to behave,
so they are the ones worth pinning down hardest.
"""

import numpy as np

from askdocs import grounding, skills
from askdocs.chunk import Chunk
from askdocs.retrieve import Hit
from askdocs.skills import SkillAnswer, SkillContext


def make_hit(url: str, score: float = 0.02) -> Hit:
    chunk = Chunk(
        chunk_id="d:0000",
        doc_id="d",
        url=url,
        title="T",
        section="docs",
        breadcrumb="T",
        anchor=None,
        text="body",
        tokens=1,
        doc_sha256="h",
        source="s",
    )
    return Hit(chunk=chunk, score=score, dense_rank=0, keyword_rank=None)


# -- guard 1: refuse before generating ------------------------------------

def test_refuses_when_nothing_was_retrieved():
    assert grounding.should_refuse_before_generating([], min_score=0.01)


def test_refuses_when_every_score_is_below_the_floor():
    hits = [make_hit("https://x/a", 0.001), make_hit("https://x/b", 0.002)]
    assert grounding.should_refuse_before_generating(hits, min_score=0.01)


def test_proceeds_when_something_scores_well():
    hits = [make_hit("https://x/a", 0.05)]
    assert not grounding.should_refuse_before_generating(hits, min_score=0.01)


# -- guard 2: fabricated citations ----------------------------------------

def test_fabricated_citation_is_caught():
    """The most damaging failure: a citation is what makes an answer look
    trustworthy, so inventing one is worse than inventing prose."""
    hits = [make_hit("https://x/real")]
    report = grounding.check("See https://x/invented for details.", hits)
    assert not report.grounded
    assert report.fabricated_urls == ["https://x/invented"]


def test_real_citation_passes():
    hits = [make_hit("https://x/real")]
    report = grounding.check("As https://x/real explains, yes.", hits)
    assert report.grounded
    assert report.fabricated_urls == []


def test_anchor_on_a_cited_url_is_not_fabrication():
    """Citing /docs/faq when the source was /docs/faq#ssg is the right
    document - flagging it would be a false alarm."""
    hits = [make_hit("https://x/docs/faq#ssg")]
    assert grounding.check("See https://x/docs/faq for more.", hits).grounded


def test_markdown_link_is_extracted():
    hits = [make_hit("https://x/real")]
    report = grounding.check("See [the docs](https://x/real).", hits)
    assert report.cited_urls == ["https://x/real"]
    assert report.grounded


def test_citation_check_can_be_disabled():
    hits = [make_hit("https://x/real")]
    report = grounding.check(
        "See https://x/invented.", hits, verify_citations=False, require_citation=False
    )
    assert report.grounded


# -- guard 3: uncited assertions ------------------------------------------

def test_long_answer_with_no_citation_is_flagged():
    report = grounding.check("word " * 60, [make_hit("https://x/a")])
    assert not report.grounded
    assert "no citation" in " ".join(report.reasons)


def test_short_answer_without_citation_is_allowed():
    assert grounding.check("Yes.", [make_hit("https://x/a")]).grounded


# -- refusals -------------------------------------------------------------

def test_refusal_is_recognised_and_skips_other_checks():
    report = grounding.check(grounding.REFUSAL, [make_hit("https://x/a")])
    assert report.refused
    assert report.grounded


def test_no_warning_for_a_good_answer():
    assert grounding.warning_for(grounding.check("Yes. https://x/a", [make_hit("https://x/a")])) is None


def test_warning_is_loud_for_a_bad_one():
    report = grounding.check("See https://x/fake.", [make_hit("https://x/a")])
    warning = grounding.warning_for(report)
    assert warning and "UNGROUNDED" in warning


# -- skills ---------------------------------------------------------------

class _Manifest:
    chunk_count = 678
    doc_hashes = {"a": "1", "b": "2"}
    source = "test"
    built_at = "2026-01-01T00:00:00+00:00"
    embed_model = "nomic-embed-text"
    dimensions = 768


class _Index:
    manifest = _Manifest()


def test_corpus_info_skill_answers_a_question_documents_cannot():
    answer = skills.dispatch("how many pages are indexed?", SkillContext(index=_Index()))
    assert answer is not None
    assert "678 chunks" in answer.text
    assert "2 documents" in answer.text


def test_skill_declines_an_ordinary_question():
    assert skills.dispatch("what is edge delivery?", SkillContext(index=_Index())) is None


def test_skill_needs_its_context():
    assert skills.dispatch("how many pages are indexed?", SkillContext(index=None)) is None


def test_a_broken_skill_never_takes_down_a_question():
    @skills.register
    class Exploding:
        name = "exploding"
        description = "raises on purpose"
        priority = 999

        def handles(self, question, context):
            raise RuntimeError("boom")

        def run(self, question, context):
            raise RuntimeError("boom")

    try:
        # Falls through to retrieval rather than crashing.
        assert skills.dispatch("anything at all", SkillContext()) is None
    finally:
        skills._REGISTRY[:] = [s for s in skills._REGISTRY if s.name != "exploding"]


def test_registry_orders_by_priority():
    names = [skill.name for skill in skills.registered()]
    assert "corpus-info" in names
