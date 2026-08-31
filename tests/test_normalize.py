"""Tests for block normalization.

The fixtures are real shapes taken from www.aem.live, not invented ones - the
bugs that matter here are all in the awkward cases the real corpus contains:
nested tables, cells that wrap across lines, and grid tables that appear inside
code fences as examples.
"""

from askdocs.normalize import normalize


def test_metadata_block_is_dropped():
    text = """# Authoring

Some real prose here.

+------------------------------------------------------------+
| Metadata                                                   |
+-------------+----------------------------------------------+
| Title       | Authoring and Publishing Content             |
+-------------+----------------------------------------------+
"""
    result = normalize(text)
    assert "Metadata" not in result.text
    assert "Some real prose here." in result.text
    assert result.dropped_blocks == 1


def test_section_metadata_anchor_is_harvested_under_its_heading():
    text = """## Is Edge Delivery Services a Static Site Generator?

No, it renders at the edge.

+------------------+
| Section Metadata |
+---------+--------+
| id      | ssg    |
+---------+--------+
"""
    result = normalize(text)
    assert result.anchors == {
        "Is Edge Delivery Services a Static Site Generator?": "ssg"
    }
    assert "Section Metadata" not in result.text


def test_anchor_binds_to_nearest_preceding_heading():
    """Two anchored sections must not have their ids swapped."""
    text = """## First question

Answer one.

+------------------+
| Section Metadata |
+---------+--------+
| id      | one    |
+---------+--------+

## Second question

Answer two.

+------------------+
| Section Metadata |
+---------+--------+
| id      | two    |
+---------+--------+
"""
    result = normalize(text)
    assert result.anchors == {"First question": "one", "Second question": "two"}


def test_table_row_becomes_a_self_contained_sentence():
    """A row must carry its own headings, so it survives any chunk boundary."""
    text = """+--------------------------------------------------+
| Table                                            |
+--------------------------------------------------+
| +----------------------+-----------------------+ |
| | **Code**             | **Likely Root Cause** | |
| +----------------------+-----------------------+ |
| | `AEM_BACKEND_        | Backend fetch failed  | |
| | FETCH_FAILED`        | (network).            | |
| +----------------------+-----------------------+ |
"""
    result = normalize(text)
    assert "AEM_BACKEND_FETCH_FAILED" in result.text.replace("\n", "")
    # The heading is repeated inline, which is the whole point.
    assert "Likely Root Cause" in result.text
    assert "+--" not in result.text


def test_wrapped_cell_is_rejoined():
    text = """+--------------------------------+
| Table                          |
+--------------------------------+
| | **Term** | **Meaning**     | |
| | edge     | close to the    | |
| |          | visitor         | |
+--------------------------------+
"""
    result = normalize(text)
    assert "close to the visitor" in result.text


def test_list_items_in_a_cell_keep_their_own_lines():
    """A bullet opens with the same character as a wrapped identifier.

    Real shape from /developer/ue-trial, where the whole sign-up pitch lives in
    one cell. Joining these produced `how you can:- Build and deploy`.
    """
    text = """+------------------------------------------------+
| Columns                                        |
+------------------------------------------------+
| Learn in the tutorial how you can:              |
| - Build and deploy a website                    |
| - Use AEM to manage your content                |
+------------------------------------------------+
"""
    result = normalize(text)
    assert "can:- Build" not in result.text
    assert "- Build and deploy a website" in result.text
    assert "- Use AEM to manage your content" in result.text


def test_heading_in_a_cell_keeps_its_own_line():
    """A heading absorbed into the previous line is a section chunk.py never sees.

    chunk.py cuts at h2/h3, so this is not only a readability fix - it restores
    boundaries the chunker was blind to.
    """
    text = """+------------------------------------------------+
| Columns                                        |
+------------------------------------------------+
| ### Sign up for the tutorial                    |
| It takes about a minute.                        |
| ### What you will build                         |
| A working site.                                 |
+------------------------------------------------+
"""
    result = normalize(text)
    assert "tutorial ### What" not in result.text
    for line in ("### Sign up for the tutorial", "### What you will build"):
        assert any(l.strip() == line for l in result.text.splitlines()), line


def test_wrapped_identifier_still_rejoins_without_a_space():
    """The list fix must not break the case the identifier rule exists for."""
    text = """+------------------------------------------------+
| Columns                                        |
+------------------------------------------------+
| the config lives in helix-                      |
| query.yaml at the root                          |
+------------------------------------------------+
"""
    result = normalize(text)
    assert "helix-query.yaml" in result.text


def test_grid_table_inside_a_code_fence_is_untouched():
    """Developer docs show grid tables as examples. Flattening one corrupts
    exactly the content the reader came for."""
    text = """# Blocks

And it will be turned into this table representation:

```
+------------------+
| Our Partners     |
+------------------+
| https://foo.com  |
+------------------+
```

More prose.
"""
    result = normalize(text)
    assert "+------------------+" in result.text
    assert "| Our Partners     |" in result.text
    assert result.dropped_blocks == 0
    assert result.flattened_blocks == 0


def test_unknown_block_is_flattened_never_dropped():
    """A site can add a block type next month. Untidy text beats vanished
    content."""
    text = """+--------------------------------------+
| Some Block Nobody Has Seen Before    |
+--------------------------------------+
| Genuinely important new content.     |
+--------------------------------------+
"""
    result = normalize(text)
    assert "Genuinely important new content." in result.text
    assert result.dropped_blocks == 0
    assert result.flattened_blocks == 1


def test_block_name_variants_are_the_same_block():
    text = """+------------------------------+
| Section Metadata (contained) |
+---------+--------------------+
| style   | dark               |
+---------+--------------------+
"""
    assert normalize(text).dropped_blocks == 1


def test_pointer_block_keeps_one_honest_line():
    text = """+----------------------------------------------+
| Fragment                                     |
+----------------------------------------------+
| [https://x/docs/special](/docs/special)      |
+----------------------------------------------+
"""
    result = normalize(text)
    assert "See also: /docs/special" in result.text


def test_empty_image_references_are_removed():
    text = "![][image0]\n\n# Title\n\nProse.\n\n[image0]: https://example.com/a.png?width=1200"
    result = normalize(text)
    assert "image0" not in result.text
    assert "# Title" in result.text


def test_normalize_is_deterministic():
    text = "## H\n\ntext\n\n+------------------+\n| Section Metadata |\n+------+-----------+\n| id   | x         |\n+------+-----------+\n"
    assert normalize(text) == normalize(text)
