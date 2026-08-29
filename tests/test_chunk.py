"""Tests for chunking."""

from askdocs.chunk import chunk_document, estimate_tokens, split_sections


def _chunk(text, **kwargs):
    return chunk_document(
        doc_id="abc123",
        url="https://example.com/docs/page",
        title="A Page",
        section="docs",
        text=text,
        doc_sha256="deadbeef",
        source="test",
        **kwargs,
    )


def test_splits_on_headings():
    text = "intro\n\n## One\n\nalpha\n\n## Two\n\nbeta"
    sections = split_sections(text)
    assert [s.heading for s in sections] == [None, "One", "Two"]


def test_heading_inside_code_fence_is_not_a_heading():
    text = "## Real\n\n```sh\n## not a heading\necho hi\n```\n\nmore"
    sections = split_sections(text)
    assert [s.heading for s in sections] == ["Real"]


def test_small_sections_are_packed_together():
    text = "## A\n\nshort\n\n## B\n\nalso short\n\n## C\n\ntiny"
    chunks = _chunk(text, target_tokens=700)
    assert len(chunks) == 1


def test_oversized_section_is_split():
    body = "word " * 2000  # far beyond the target
    chunks = _chunk(f"## Big\n\n{body}", target_tokens=200)
    assert len(chunks) > 1
    assert all(chunk.text.strip() for chunk in chunks)


def test_chunk_ids_are_unique_and_stable():
    text = "## A\n\n" + "alpha " * 500 + "\n\n## B\n\n" + "beta " * 500
    first = _chunk(text, target_tokens=200)
    second = _chunk(text, target_tokens=200)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


def test_breadcrumb_carries_page_and_section():
    chunks = _chunk("## Getting Started\n\nsome text here")
    assert chunks[0].breadcrumb == "A Page > Getting Started"


def test_embedding_text_includes_the_breadcrumb():
    """Two chunks about 'limits' from different pages are otherwise near
    identical to an embedding model."""
    chunk = _chunk("## Limits\n\nthe limit is ten")[0]
    assert chunk.embedding_text().startswith("A Page > Limits")


def test_anchored_section_stands_alone_and_gets_its_anchor():
    """An author-assigned anchor marks an addressable unit - one FAQ question
    and its answer. Packing several together can only cite one of them."""
    text = "## Q one\n\nanswer one\n\n## Q two\n\nanswer two"
    chunks = _chunk(text, anchors={"Q one": "one", "Q two": "two"})
    assert len(chunks) == 2
    assert chunks[0].url.endswith("#one")
    assert chunks[1].url.endswith("#two")


def test_chunk_without_an_anchored_heading_gets_no_anchor():
    """Citing the page top is correct; citing the wrong section is not."""
    chunks = _chunk("## Unmarked\n\ntext", anchors={"Something Else": "x"})
    assert chunks[0].anchor is None
    assert "#" not in chunks[0].url


def test_no_empty_chunks():
    chunks = _chunk("## A\n\n\n\n## B\n\nreal content")
    assert all(chunk.text.strip() for chunk in chunks)


def test_token_estimate_is_positive():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) == 100


def test_overlong_heading_is_trimmed_in_the_breadcrumb():
    """A hero block flattened into something heading-shaped must not put 300
    characters of marketing copy in front of every vector on the page."""
    long_heading = "## " + "Create a website seriously fast " * 12
    chunk = _chunk(f"{long_heading}\n\nbody text")[0]
    assert len(chunk.breadcrumb) < 160
    assert chunk.breadcrumb.endswith("...")


def test_heading_markup_is_stripped_from_the_breadcrumb():
    chunk = _chunk("## **Large Sites**<br>and more\n\nbody")[0]
    assert "**" not in chunk.breadcrumb
    assert "<br>" not in chunk.breadcrumb
