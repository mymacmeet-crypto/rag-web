# askdocs — command reference

Shell examples are **zsh / bash** (macOS).

---

## One-time setup

Needs internet for the embed model only. The Devstral model is already
downloaded. Everything after setup is offline.

```bash
# 1. Install Ollama (embed model only — no chat model needed from Ollama)
#    https://ollama.com/download  OR:
brew install ollama

# 2. Pull the embed model (~275 MB — turns text into numbers)
ollama pull nomic-embed-text

# 3. Install mlx-lm (runs the Devstral chat model on Apple Silicon)
pip install mlx-lm

# 4. Install askdocs
cd ~/LocalOllamaLLM/rag-web
pip install -e .

# 5. Confirm everything lines up
askdocs status
```

`askdocs status` should show:
- `pulled` beside `nomic-embed-text` (Ollama)
- `mlxlm server reachable` beside `generate` (only after the server is started — see below)

---

## Start the model server (required before asking questions)

The Devstral model runs via `mlx_lm.server`. Start it in a **separate terminal**
and leave it running.

```bash
python -m mlx_lm.server \
  --model mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit \
  --port 8080
```

The model is already cached at `~/.cache/huggingface/hub/` — it will not
re-download. First load takes ~30 s while weights are read from disk.

To stop it: `Ctrl-C` in that terminal.

---

## Add documents

Put files in `corpus/`. Subfolders are fine; names starting with `_` or `.` are
skipped.

```bash
# Drop in a JSONL dataset
cp ~/web-archiver/archive/www.aem.live/dataset.jsonl corpus/aem-live.jsonl

# Or plain Markdown
cp runbook.md corpus/

# See what will be indexed
askdocs status
```

---

## Build the index

```bash
askdocs index              # build, or re-do only what changed
askdocs index --rebuild    # force a full rebuild
```

Indexing only calls the **embed model** (Ollama, fast). The chat model is not
needed and the server does not need to be running for this step.

Run this after adding, changing, or removing anything in `corpus/`.

---

## Ask questions

The mlx_lm server must be running first (see above).

```bash
askdocs ask "does it rebuild the whole site when I change one page?"
askdocs ask "what does AEM_BACKEND_FETCH_FAILED mean?"
askdocs ask "how do I add an image to a page?"
```

A 24 B model on Apple Silicon with 4-bit weights: expect roughly 15–40 s per
answer depending on your chip. Words stream out as they are written.

### The flags that matter

```bash
# Retrieve only — no model call, comes back instantly.
# Use this while tuning: it shows whether the right chunks are found.
askdocs ask "..." --dry-run

# Show what was retrieved AND answer
askdocs ask "..." --show-sources

# More or fewer chunks in the prompt (more context = slower generation)
askdocs ask "..." --top-k 8

# Skip the trailing source list
askdocs ask "..." --no-cite
```

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Answered, and the grounding checks passed |
| `2` | Setup problem — no index, server down, model mismatch |
| `3` | **Answered, but a grounding check failed** — fabricated citation or unsupported claim |
| `130` | You pressed Ctrl-C |

Code `3` is the one to watch for in scripts.

---

## Measure quality

```bash
askdocs eval               # recall@5 and MRR — fast, embed model only, server not needed
askdocs eval --refusal     # also checks it declines unanswerable questions (server required, slow)
```

`eval` exits non-zero if recall falls below the threshold in `askdocs.toml`.

Edit the question set at `eval/questions.yaml`. Paraphrase questions rather than
copying headings, and include some deliberately unanswerable ones.

---

## Inspect

```bash
askdocs status             # corpus, index, models, backend, connectors, skills
askdocs env                # every env var, its value, and where it came from
askdocs --version
askdocs --help
askdocs ask --help
```

`askdocs env` is the one to reach for when a setting seems not to apply.

---

## Settings

Precedence: **CLI flag → environment variable → `askdocs.toml` → default.**

```bash
# For one command
ASKDOCS_MLXLM_HOST=http://gpu-mac:8080 askdocs ask "..."

# For this whole terminal session
export ASKDOCS_MLXLM_HOST=http://gpu-mac:8080
export ASKDOCS_GENERATE_MODEL=mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit

# Clear it
unset ASKDOCS_MLXLM_HOST
```

### The variables worth knowing

| Variable | Does |
|---|---|
| `ASKDOCS_GENERATE_BACKEND` | `ollama` or `mlxlm` — which backend writes answers |
| `ASKDOCS_GENERATE_MODEL` | HF repo name or Ollama model name for the chat model |
| `ASKDOCS_MLXLM_HOST` | mlx_lm server address (default `http://127.0.0.1:8080`) |
| `ASKDOCS_EMBED_MODEL` | Which model makes the vectors — **changing this invalidates the index** |
| `ASKDOCS_OLLAMA_HOST` | Point the embed model at another machine |
| `ASKDOCS_RETRIEVE_TOP_K` | Chunks per answer |
| `ASKDOCS_CORPUS_DIR` | Where documents live |
| `ASKDOCS_INDEX_PATH` | Where the index is written |
| `ASKDOCS_GROUNDING_MIN_SCORE` | How strict the refuse-early gate is |
| `ASKDOCS_GENERATE_TEMPERATURE` | Raise above 0 for more variation — and more invention |

Run `askdocs env` for the full list.

---

## Switching back to Ollama

To use a smaller GGUF model via Ollama instead of Devstral, edit `askdocs.toml`:

```toml
[generate]
backend = "ollama"
model   = "qwen2.5:7b"
```

Then pull it:

```bash
ollama pull qwen2.5:7b
```

No code changes. Embedding always stays on Ollama regardless of the generate backend.

---

## Refreshing after the site changes

```bash
# 1. Refetch changed pages — needs internet
cd ~/LocalOllamaLLM/web-archiver
npm run dataset

# 2. Copy the updated dataset across
cp archive/www.aem.live/dataset.jsonl ~/LocalOllamaLLM/rag-web/corpus/aem-live.jsonl

# 3. Re-embed only what changed — offline, server not needed
cd ~/LocalOllamaLLM/rag-web
askdocs index
```

---

## Development

```bash
python -m pytest tests -q              # all tests
python -m pytest tests -q -k normalize # just the cleaning tests
python -m pytest tests -v              # verbose

# Run without installing
PYTHONPATH=src python -m askdocs.cli status
```

---

## When something is wrong

| Symptom | Try |
|---|---|
| `Could not reach mlx_lm server` | Start `python -m mlx_lm.server --model mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit --port 8080` |
| `Could not reach Ollama` | `ollama serve` in another terminal |
| `Model ... is not available` | `ollama pull nomic-embed-text` |
| `No index at ...` | `askdocs index` |
| `Index was built with embedding model X but config says Y` | `askdocs index --rebuild` |
| `Nothing to index` | Put a `.jsonl` or `.md` file in `corpus/` |
| Answers are wrong or irrelevant | `askdocs ask "..." --dry-run` — check whether the right chunks are found. If not, that is a retrieval problem, not a prompt problem |
| Answers are too slow | Lower `--top-k`, or run the mlx_lm server on a Mac with more RAM |
| A setting seems ignored | `askdocs env` — it names the source of every value |

---

## Reading the query log

Every question is appended to `logs/queries.jsonl`.

```bash
# Last 5 questions
tail -5 logs/queries.jsonl | python -m json.tool

# Every answer that failed a grounding check
grep '"grounded": false' logs/queries.jsonl
```
