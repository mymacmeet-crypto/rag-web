"""Tests for env overrides, connectors, and the drop folder."""

import json

import pytest

from askdocs import config as config_module
from askdocs.config import ConfigError, load
from askdocs.sources import Document, connector_for, discover, load_all


def write_config(tmp_path, body: str = ""):
    (tmp_path / "askdocs.toml").write_text(body, encoding="utf-8")
    return tmp_path


# -- configuration --------------------------------------------------------

def test_defaults_apply_with_no_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load(start=tmp_path)
    assert config.embed.model == "nomic-embed-text"
    assert config.generate.model == "llama3.2:3b"


def test_toml_overrides_default(tmp_path):
    root = write_config(tmp_path, '[generate]\nmodel = "qwen2.5:7b"\n')
    config = load(str(root / "askdocs.toml"))
    assert config.generate.model == "qwen2.5:7b"
    assert config.origins["generate.model"] == "askdocs.toml"


def test_env_overrides_toml(tmp_path, monkeypatch):
    root = write_config(tmp_path, '[generate]\nmodel = "from-file"\n')
    monkeypatch.setenv("ASKDOCS_GENERATE_MODEL", "from-env")
    config = load(str(root / "askdocs.toml"))
    assert config.generate.model == "from-env"
    assert config.origins["generate.model"] == "env ASKDOCS_GENERATE_MODEL"


def test_env_sets_embed_model_and_host(tmp_path, monkeypatch):
    root = write_config(tmp_path)
    monkeypatch.setenv("ASKDOCS_EMBED_MODEL", "mxbai-embed-large")
    monkeypatch.setenv("ASKDOCS_OLLAMA_HOST", "http://gpu-box:11434")
    config = load(str(root / "askdocs.toml"))
    assert config.embed.model == "mxbai-embed-large"
    assert config.ollama.host == "http://gpu-box:11434"


def test_env_coerces_numbers_and_booleans(tmp_path, monkeypatch):
    root = write_config(tmp_path)
    monkeypatch.setenv("ASKDOCS_RETRIEVE_TOP_K", "9")
    monkeypatch.setenv("ASKDOCS_GROUNDING_VERIFY_CITATIONS", "false")
    config = load(str(root / "askdocs.toml"))
    assert config.retrieve.top_k == 9
    assert config.grounding.verify_citations is False


def test_bad_env_value_is_rejected_loudly(tmp_path, monkeypatch):
    root = write_config(tmp_path)
    monkeypatch.setenv("ASKDOCS_RETRIEVE_TOP_K", "not-a-number")
    with pytest.raises(ConfigError, match="ASKDOCS_RETRIEVE_TOP_K"):
        load(str(root / "askdocs.toml"))


def test_misspelled_key_is_an_error_not_a_silent_no_op(tmp_path):
    root = write_config(tmp_path, '[generate]\nmodle = "typo"\n')
    with pytest.raises(ConfigError, match="unknown key"):
        load(str(root / "askdocs.toml"))


def test_paths_resolve_against_the_config_file(tmp_path):
    root = write_config(tmp_path, '[corpus]\ndir = "./docs"\n')
    config = load(str(root / "askdocs.toml"))
    assert config.corpus.dir == (tmp_path / "docs").resolve()


# -- connectors -----------------------------------------------------------

def test_jsonl_connector_claims_jsonl(tmp_path):
    assert connector_for(tmp_path / "a.jsonl").name == "jsonl"
    assert connector_for(tmp_path / "a.md").name == "markdown"
    assert connector_for(tmp_path / "a.pdf") is None


def test_jsonl_loads_records(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps({"id": "1", "url": "https://x/a", "text": "hello", "title": "A"}) + "\n",
        encoding="utf-8",
    )
    documents = list(load_all([path], "test"))
    assert len(documents) == 1
    assert documents[0].title == "A"
    # A missing hash is filled in, or refreshes re-embed everything.
    assert documents[0].sha256


def test_jsonl_missing_required_field_names_it(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"id": "1", "text": "hello"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="url"):
        list(load_all([path], "test"))


def test_markdown_reads_front_matter(tmp_path):
    path = tmp_path / "page.md"
    path.write_text(
        "---\nurl: https://example.com/p\ntitle: The Page\n---\n\n# Heading\n\nBody.",
        encoding="utf-8",
    )
    document = next(iter(load_all([path], "test")))
    assert document.url == "https://example.com/p"
    assert document.title == "The Page"
    assert "front matter" not in document.text
    assert document.text.strip().startswith("# Heading")


def test_markdown_without_front_matter_still_cites_somewhere_real(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("# My Note\n\nSome content.", encoding="utf-8")
    document = next(iter(load_all([path], "test")))
    assert document.title == "My Note"
    assert document.url.startswith("file:///")


def test_discover_finds_supported_files_in_subfolders(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("# b", encoding="utf-8")
    (tmp_path / "ignore.pdf").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("x", encoding="utf-8")
    (tmp_path / "_notes.md").write_text("x", encoding="utf-8")

    names = sorted(path.name for path in discover(tmp_path))
    assert names == ["a.jsonl", "b.md"]


def test_document_hash_is_content_addressed():
    first = Document(id="1", url="u", text="same").with_hash()
    second = Document(id="2", url="v", text="same").with_hash()
    assert first.sha256 == second.sha256
