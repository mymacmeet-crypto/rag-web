"""Command line interface.

    askdocs index          build or refresh the index from corpus/
    askdocs ask "..."      answer a question, streaming, with grounding checks
    askdocs eval           measure recall@k against the golden set
    askdocs status         config, index, models, connectors, skills
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import __version__, grounding, skills
from .answer import build_prompt, stream_answer
from .config import Config, ConfigError, load
from .evaluate import (
    evaluate_refusal,
    evaluate_retrieval,
    load_questions,
    mean_reciprocal_rank,
    recall,
)
from .logs import QueryRecord, append, record_hits
from .mlxlm import MlxLmClient, MlxLmError
from .ollama import OllamaClient, OllamaError
from .pipeline import EmptyCorpusError, build_index, changed_documents, corpus_files
from .retrieve import search
from .sources import registered as registered_connectors
from .store import Index, normalize_rows


def _client(config: Config) -> OllamaClient:
    return OllamaClient(host=config.ollama.host, timeout=config.ollama.timeout)


def _generate_client(config: Config) -> OllamaClient | MlxLmClient:
    if config.generate.backend == "mlxlm":
        return MlxLmClient(host=config.mlxlm.host, timeout=config.mlxlm.timeout)
    return OllamaClient(host=config.ollama.host, timeout=config.ollama.timeout)


def _note(message: str) -> None:
    print(message, file=sys.stderr)


def _apply_overrides(args: argparse.Namespace, config: Config) -> Config:
    """CLI flags win over env, which wins over the file."""
    from dataclasses import replace

    origins = dict(config.origins)
    if getattr(args, "embed_model", None):
        config = replace(config, embed=replace(config.embed, model=args.embed_model))
        origins["embed.model"] = "--embed-model"
    if getattr(args, "model", None):
        config = replace(
            config, generate=replace(config.generate, model=args.model)
        )
        origins["generate.model"] = "--model"
    if getattr(args, "top_k", None):
        config = replace(
            config, retrieve=replace(config.retrieve, top_k=args.top_k)
        )
        origins["retrieve.top_k"] = "--top-k"
    return replace(config, origins=origins)


# --------------------------------------------------------------------------

def cmd_index(args: argparse.Namespace, config: Config) -> int:
    client = _client(config)

    if not args.rebuild and config.index_path.exists():
        try:
            existing = Index.load(config.index_path)
        except (OSError, ValueError):
            existing = None
        if existing is not None:
            changed = changed_documents(config, existing)
            if not changed:
                _note(
                    f"Index is current: {existing.manifest.chunk_count} chunks, "
                    f"built {existing.manifest.built_at}. Nothing changed."
                )
                _note("Force a full rebuild with --rebuild.")
                return 0
            _note(f"{len(changed)} document(s) changed since the last build.")

    started = time.perf_counter()
    index = build_index(config, client, progress=_note)
    index.save(config.index_path)

    _note("")
    _note(
        f"Indexed {index.manifest.chunk_count} chunks "
        f"({index.manifest.dimensions} dims) in {time.perf_counter() - started:.1f}s"
    )
    _note(f"Written to {config.index_path}")
    return 0


def cmd_ask(args: argparse.Namespace, config: Config) -> int:
    question = " ".join(args.question).strip()
    if not question:
        _note('Ask something:  askdocs ask "how do I publish a page?"')
        return 2

    index = Index.load(config.index_path)
    client = _client(config)
    gen_client = _generate_client(config)

    # Vectors from two different models are not comparable. Search would still
    # return five chunks - simply the wrong five, with no error anywhere.
    if index.manifest.embed_model != config.embed.model:
        _note(
            f"Index was built with embedding model "
            f"{index.manifest.embed_model!r} but the current setting is "
            f"{config.embed.model!r}.\n"
            f"  Vectors from different models are not comparable.\n"
            f"  Rebuild:  askdocs index --rebuild"
        )
        return 2

    # A skill that claims the question answers it directly - no retrieval, no
    # generation, and no opportunity to invent anything.
    skill_answer = skills.dispatch(
        question, skills.SkillContext(index=index, config=config)
    )
    if skill_answer is not None:
        print(skill_answer.text)
        return 0

    started = time.perf_counter()
    query_vector = normalize_rows(
        np.array(client.embed([question], config.embed.model), dtype=np.float32)
    )[0]
    hits = search(
        index,
        question,
        query_vector,
        top_k=config.retrieve.top_k,
        candidates=config.retrieve.candidates,
        rrf_k=config.retrieve.rrf_k,
    )
    retrieve_ms = int((time.perf_counter() - started) * 1000)

    if args.show_sources or args.dry_run:
        _note(f"\nRetrieved {len(hits)} chunks in {retrieve_ms}ms:")
        for number, hit in enumerate(hits, 1):
            _note(f"  {number}. [{hit.found_by:>8}] score {hit.score:.4f}  {hit.chunk.breadcrumb}")
            _note(f"     {hit.chunk.url}")

    # Guard one: nothing plausible was retrieved, so do not ask a small model
    # to work with irrelevant context. It will oblige, and that is the problem.
    if grounding.should_refuse_before_generating(hits, config.grounding.min_score):
        best = max((hit.score for hit in hits), default=0.0)
        print(grounding.LOW_CONFIDENCE)
        _note(f"\n[best score {best:.4f} < floor {config.grounding.min_score}]")
        append(
            config.log_path,
            QueryRecord(
                question=question,
                retrieved=record_hits(hits),
                refused=True,
                grounded=True,
                retrieve_ms=retrieve_ms,
                embed_model=config.embed.model,
                generate_model=config.generate.model,
            ),
        )
        return 0

    prompt = build_prompt(question, hits)
    if args.show_sources or args.dry_run:
        _note(f"\nPrompt is ~{prompt.approx_tokens} tokens.\n")
    if args.dry_run:
        return 0

    gen_client.require_model(config.generate.model)
    _note(f"[{config.generate.model} thinking - this takes a while on CPU]\n")

    started = time.perf_counter()
    pieces: list[str] = []
    for piece in stream_answer(
        gen_client,
        prompt,
        model=config.generate.model,
        temperature=config.generate.temperature,
        max_tokens=config.generate.max_tokens,
    ):
        pieces.append(piece)
        print(piece, end="", flush=True)
    print()
    generate_ms = int((time.perf_counter() - started) * 1000)
    answer = "".join(pieces)

    # Guards two and three: every cited URL must have been retrieved, and a
    # substantive answer must cite something.
    report = grounding.check(
        answer,
        hits,
        verify_citations=config.grounding.verify_citations,
        require_citation=config.grounding.require_citation,
    )

    if not args.no_cite and not report.refused:
        _note("\nSources:")
        for number, hit in enumerate(hits, 1):
            _note(f"  {number}. {hit.chunk.url}")

    warning = grounding.warning_for(report)
    if warning:
        _note(warning)

    _note(f"\n[retrieved {retrieve_ms}ms, generated {generate_ms / 1000:.1f}s]")

    append(
        config.log_path,
        QueryRecord(
            question=question,
            retrieved=record_hits(hits),
            prompt_tokens=prompt.approx_tokens,
            answer_chars=len(answer),
            refused=report.refused,
            grounded=report.grounded,
            grounding_reasons=report.reasons,
            fabricated_urls=report.fabricated_urls,
            retrieve_ms=retrieve_ms,
            generate_ms=generate_ms,
            embed_model=config.embed.model,
            generate_model=config.generate.model,
        ),
    )
    # A non-zero exit on an ungrounded answer makes this scriptable.
    return 0 if report.grounded else 3


def cmd_eval(args: argparse.Namespace, config: Config) -> int:
    index = Index.load(config.index_path)
    client = _client(config)
    gen_client = _generate_client(config)
    questions = load_questions(config.eval.questions)

    results = evaluate_retrieval(index, client, config, questions)
    if not results:
        _note("No answerable questions in the evaluation set.")
        return 2

    score = recall(results)
    print(f"Retrieval over {len(results)} questions, top-{config.retrieve.top_k}:")
    print(f"  recall@{config.retrieve.top_k}  {score:.0%}")
    print(f"  MRR          {mean_reciprocal_rank(results):.3f}")

    misses = [result for result in results if not result.found]
    if misses:
        print(f"\n{len(misses)} miss(es) - the right page never reached the prompt:")
        for result in misses:
            print(f"\n  Q: {result.question.question}")
            print(f"  expected: {', '.join(result.question.expect_urls)}")
            for url in result.retrieved[:3]:
                print(f"  got:      {url}")

    if args.refusal:
        print("\nRefusal check (needs the chat model, slow on CPU)...")
        refusals = evaluate_refusal(index, client, config, questions, generate_client=gen_client)
        if refusals:
            held = sum(1 for result in refusals if result.refused)
            print(f"  declined correctly: {held}/{len(refusals)}")
            for result in refusals:
                if not result.refused:
                    print(f"\n  INVENTED an answer to: {result.question.question}")
                    print(f"    {result.answer[:200]}")
            if held < len(refusals):
                return 1
        else:
            print("  no unanswerable questions in the set - add some")

    if score < config.eval.min_recall:
        print(
            f"\nFAIL: recall {score:.0%} is below the "
            f"{config.eval.min_recall:.0%} threshold.",
            file=sys.stderr,
        )
        return 1

    print(f"\nPASS: recall {score:.0%} meets the {config.eval.min_recall:.0%} threshold.")
    return 0


def cmd_status(args: argparse.Namespace, config: Config) -> int:
    print(f"config     {config.root}")
    print(f"corpus     {config.corpus.dir}")
    try:
        files = corpus_files(config)
        if files:
            for path in files[:8]:
                print(f"           {path.name}")
            if len(files) > 8:
                print(f"           ... and {len(files) - 8} more")
        else:
            print("           (empty - drop .jsonl or .md files in here)")
    except OSError as error:
        print(f"           unreadable: {error}")

    try:
        index = Index.load(config.index_path)
        print(
            f"index      {index.manifest.chunk_count} chunks from "
            f"{len(index.manifest.doc_hashes)} documents, built {index.manifest.built_at}"
        )
        if index.manifest.embed_model != config.embed.model:
            print(
                f"           MISMATCH: index built with "
                f"{index.manifest.embed_model}, config says {config.embed.model}"
            )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"index      not built ({error})")

    print(f"embed      {config.embed.model}    [{config.origins.get('embed.model', 'default')}]")
    print(f"generate   {config.generate.model}    [{config.origins.get('generate.model', 'default')}]")
    print(f"backend    {config.generate.backend}    [{config.origins.get('generate.backend', 'default')}]")
    print(f"ollama     {config.ollama.host}    [{config.origins.get('ollama.host', 'default')}]")
    if config.generate.backend == "mlxlm":
        print(f"mlxlm      {config.mlxlm.host}    [{config.origins.get('mlxlm.host', 'default')}]")

    try:
        models = _client(config).available_models()
        ok = config.embed.model in models or any(m.split(":")[0] == config.embed.model for m in models)
        print(f"           embed     {'pulled' if ok else 'NOT PULLED - ollama pull ' + config.embed.model}")
    except OllamaError as error:
        print(f"           {error}")

    if config.generate.backend == "mlxlm":
        try:
            _generate_client(config).require_model(config.generate.model)
            print(f"           generate  mlxlm server reachable at {config.mlxlm.host}")
        except MlxLmError as error:
            print(f"           {error}")
    else:
        try:
            models = _client(config).available_models()
            ok = config.generate.model in models or any(m.split(":")[0] == config.generate.model for m in models)
            print(f"           generate  {'pulled' if ok else 'NOT PULLED - ollama pull ' + config.generate.model}")
        except OllamaError as error:
            print(f"           {error}")

    print(f"connectors {', '.join(sorted(c.name for c in registered_connectors()))}")
    print(f"skills     {', '.join(skill.name for skill in skills.registered()) or '(none)'}")
    return 0


def cmd_env(args: argparse.Namespace, config: Config) -> int:
    """Print every environment variable this tool reads."""
    from .config import _SECTIONS, _env_name
    from dataclasses import fields as dc_fields

    print("Environment variables (override askdocs.toml, overridden by CLI flags):\n")
    for section, cls in _SECTIONS.items():
        for entry in dc_fields(cls):
            name = _env_name(section, entry.name)
            current = getattr(getattr(config, section), entry.name)
            origin = config.origins.get(f"{section}.{entry.name}", "default")
            print(f"  {name:<34} = {str(current):<40} [{origin}]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="askdocs",
        description="Offline question answering over a local documentation corpus.",
    )
    parser.add_argument("--version", action="version", version=f"askdocs {__version__}")
    parser.add_argument("--config", help="path to askdocs.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    index_parser = sub.add_parser("index", help="build or refresh the index")
    index_parser.add_argument("--rebuild", action="store_true", help="rebuild everything")
    index_parser.add_argument("--embed-model", help="override the embedding model")
    index_parser.set_defaults(func=cmd_index)

    ask_parser = sub.add_parser("ask", help="answer a question")
    ask_parser.add_argument("question", nargs="+")
    ask_parser.add_argument("--model", help="override the chat model")
    ask_parser.add_argument("--embed-model", help="override the embedding model")
    ask_parser.add_argument("--top-k", type=int, help="chunks to retrieve")
    ask_parser.add_argument("--show-sources", action="store_true", help="print what was retrieved")
    ask_parser.add_argument("--dry-run", action="store_true", help="retrieve only, do not generate")
    ask_parser.add_argument("--no-cite", action="store_true", help="omit the source list")
    ask_parser.set_defaults(func=cmd_ask)

    eval_parser = sub.add_parser("eval", help="measure retrieval quality")
    eval_parser.add_argument("--refusal", action="store_true", help="also check it declines (slow)")
    eval_parser.add_argument("--model", help="override the chat model")
    eval_parser.set_defaults(func=cmd_eval)

    sub.add_parser("status", help="show config, index, models, extensions").set_defaults(
        func=cmd_status
    )
    sub.add_parser("env", help="list every environment variable read").set_defaults(
        func=cmd_env
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _apply_overrides(args, load(args.config))
        return args.func(args, config)
    except (ConfigError, OllamaError, MlxLmError, EmptyCorpusError) as error:
        print(f"{error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        print(f"{error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
