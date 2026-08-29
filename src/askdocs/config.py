"""Configuration.

Precedence, highest first:

    CLI flag  ->  environment variable  ->  askdocs.toml  ->  built-in default

Environment variables are named ASKDOCS_<SECTION>_<KEY>, e.g.

    ASKDOCS_EMBED_MODEL=mxbai-embed-large
    ASKDOCS_GENERATE_MODEL=qwen2.5:7b
    ASKDOCS_OLLAMA_HOST=http://gpu-box:11434
    ASKDOCS_RETRIEVE_TOP_K=8

so a deployment can swap models without editing a checked-in file. `askdocs
status` prints where each value came from, because a setting that silently
does not apply is worse than one that is obviously wrong.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, get_type_hints

DEFAULT_FILENAME = "askdocs.toml"
ENV_PREFIX = "ASKDOCS"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class CorpusConfig:
    # The drop folder. Anything supported placed in here is indexed.
    dir: Path = Path("./corpus")
    # Optional single explicit file, for pointing straight at a dataset that
    # lives elsewhere. When set it is indexed in addition to the folder.
    dataset: Path | None = None
    source: str = "corpus"


@dataclass(frozen=True)
class ChunkConfig:
    target_tokens: int = 700
    overlap_tokens: int = 80
    oversize_factor: float = 1.4


@dataclass(frozen=True)
class EmbedConfig:
    model: str = "nomic-embed-text"
    batch_size: int = 16


@dataclass(frozen=True)
class GenerateConfig:
    model: str = "llama3.2:3b"
    temperature: float = 0.0
    max_tokens: int = 800
    # "ollama" or "mlxlm". When "mlxlm", generation is routed to the mlx_lm
    # OpenAI-compatible server instead of Ollama.
    backend: str = "ollama"


@dataclass(frozen=True)
class RetrieveConfig:
    top_k: int = 5
    candidates: int = 25
    rrf_k: int = 60


@dataclass(frozen=True)
class GroundingConfig:
    """Guards against the failure that matters most: a confident invention.

    A small local model does not go quiet when the sources are irrelevant, it
    writes something plausible. These are the checks that catch it.
    """
    # Below this fused retrieval score, refuse without calling the model at
    # all. Nothing good comes of handing five irrelevant chunks to a 3B model.
    min_score: float = 0.0155
    # Reject answers citing a URL that was not in the retrieved set.
    verify_citations: bool = True
    # Require at least one citation in any non-refusal answer.
    require_citation: bool = True


@dataclass(frozen=True)
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    timeout: int = 300


@dataclass(frozen=True)
class MlxLmConfig:
    host: str = "http://127.0.0.1:8080"
    timeout: int = 300


@dataclass(frozen=True)
class EvalConfig:
    questions: Path = Path("./eval/questions.yaml")
    min_recall: float = 0.80


@dataclass(frozen=True)
class IndexConfig:
    path: Path = Path("./index")
    # Storage backend. "flat" keeps chunks and vectors in plain files, which is
    # exact and sub-millisecond well past 10k chunks. See store.py for when
    # that stops being true.
    backend: str = "flat"


@dataclass(frozen=True)
class LogConfig:
    path: Path = Path("./logs/queries.jsonl")


@dataclass(frozen=True)
class Config:
    root: Path
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    mlxlm: MlxLmConfig = field(default_factory=MlxLmConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    log: LogConfig = field(default_factory=LogConfig)
    # Where each overridden value came from, for `askdocs status`.
    origins: dict[str, str] = field(default_factory=dict)

    @property
    def index_path(self) -> Path:
        return self.index.path

    @property
    def log_path(self) -> Path:
        return self.log.path


_SECTIONS: dict[str, type] = {
    "corpus": CorpusConfig,
    "index": IndexConfig,
    "chunk": ChunkConfig,
    "embed": EmbedConfig,
    "generate": GenerateConfig,
    "retrieve": RetrieveConfig,
    "grounding": GroundingConfig,
    "ollama": OllamaConfig,
    "mlxlm": MlxLmConfig,
    "eval": EvalConfig,
    "log": LogConfig,
}

_PATH_FIELDS = {
    ("corpus", "dir"),
    ("corpus", "dataset"),
    ("index", "path"),
    ("eval", "questions"),
    ("log", "path"),
}


def _coerce(value: Any, annotation: Any, where: str) -> Any:
    """Turn a string (from env) or TOML scalar into the declared type."""
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    try:
        if "Path" in text:
            return Path(str(value)).expanduser()
        if "bool" in text:
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "on"}
        if "int" in text:
            return int(value)
        if "float" in text:
            return float(value)
        return str(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{where}: cannot read {value!r} as {text} ({error})") from error


def _env_name(section: str, key: str) -> str:
    return f"{ENV_PREFIX}_{section.upper()}_{key.upper()}"


def _build_section(
    cls: type,
    table: dict[str, Any],
    section: str,
    origins: dict[str, str],
):
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}

    unknown = set(table) - known
    if unknown:
        raise ConfigError(
            f"[{section}] has unknown key(s): {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )

    values: dict[str, Any] = {}
    for key in known:
        env_key = _env_name(section, key)
        if env_key in os.environ:
            values[key] = _coerce(os.environ[env_key], hints.get(key, str), env_key)
            origins[f"{section}.{key}"] = f"env {env_key}"
        elif key in table:
            values[key] = _coerce(table[key], hints.get(key, str), f"[{section}].{key}")
            origins[f"{section}.{key}"] = "askdocs.toml"

    return cls(**values)


def find_config(explicit: str | None = None, start: Path | None = None) -> Path | None:
    """Locate the config file. Returns None when there is none - defaults plus
    environment variables are a complete configuration on their own."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        return path

    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / DEFAULT_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load(explicit: str | None = None, start: Path | None = None) -> Config:
    path = find_config(explicit, start)
    root = path.parent if path else Path.cwd()

    raw: dict[str, Any] = {}
    if path:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)

    unknown_sections = set(raw) - set(_SECTIONS)
    if unknown_sections:
        raise ConfigError(
            f"Unknown section(s) in {path}: {', '.join(sorted(unknown_sections))}"
        )

    origins: dict[str, str] = {}
    built: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        table = raw.get(name, {})
        if not isinstance(table, dict):
            raise ConfigError(f"[{name}] must be a table")
        built[name] = _build_section(cls, dict(table), name, origins)

    config = Config(root=root, origins=origins, **built)
    return _resolve_paths(config)


def _resolve_paths(config: Config) -> Config:
    """Make every path absolute, relative to the config file's directory."""
    updates: dict[str, Any] = {}
    for section, key in _PATH_FIELDS:
        holder = getattr(config, section)
        value = getattr(holder, key)
        if value is None:
            continue
        resolved = value if Path(value).is_absolute() else (config.root / value).resolve()
        updates.setdefault(section, {})[key] = Path(resolved)

    for section, changes in updates.items():
        setattr_target = replace(getattr(config, section), **changes)
        config = replace(config, **{section: setattr_target})
    return config
