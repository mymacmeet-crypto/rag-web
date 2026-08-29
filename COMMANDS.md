# askdocs — command reference

Shell examples are **PowerShell**, since that is the shell on this machine.
Bash equivalents are given where the syntax differs (mainly environment
variables).

---

## One-time setup

Needs internet. Everything after this is offline.

```powershell
# 1. Install Ollama — download and run the installer
#    https://ollama.com/download
#    Then check it is running:
ollama --version

# 2. Pull the two models (about 2.5 GB total)
ollama pull nomic-embed-text     # turns text into numbers  (~275 MB)
ollama pull llama3.2:3b          # writes the answers       (~2 GB)

# 3. Install askdocs
cd D:\MCP\askdocs
pip install -e .

# 4. Confirm everything lines up before building
askdocs status
```

`askdocs status` should show `pulled` beside both models. If it says Ollama is
unreachable, start it with `ollama serve` in another terminal.

---

## Add documents

Put files in `corpus/`. Subfolders are fine; names starting with `_` or `.` are
skipped.

```powershell
# Copy a web-archiver dataset in
Copy-Item D:\MCP\web-archiver\archive\www.aem.live\dataset.jsonl D:\MCP\askdocs\corpus\aem-live.jsonl

# Or drop in plain Markdown
Copy-Item .\runbook.md D:\MCP\askdocs\corpus\

# See what will be indexed
askdocs status
```

---

## Build the index

```powershell
askdocs index              # build, or re-do only what changed
askdocs index --rebuild    # force a full rebuild
```

Run this after adding, changing or removing anything in `corpus/`. First build
on this machine takes a few minutes; later runs only re-embed documents whose
content actually changed.

---

## Ask questions

```powershell
askdocs ask "does it rebuild the whole site when I change one page?"
askdocs ask "what does AEM_BACKEND_FETCH_FAILED mean?"
askdocs ask "how do I add an image to a page?"
```

Expect roughly 30–50 seconds per answer on this CPU. Words stream out as they
are written.

### The flags that matter

```powershell
# Retrieve only — no model call, comes back instantly.
# This is the one to use while tuning: it shows whether the right
# chunks are being found, which is where most problems actually are.
askdocs ask "..." --dry-run

# Show what was retrieved AND answer
askdocs ask "..." --show-sources

# More or fewer chunks in the prompt (more context = slower on CPU)
askdocs ask "..." --top-k 8

# Try a different chat model for one question
askdocs ask "..." --model qwen2.5:7b

# Skip the trailing source list
askdocs ask "..." --no-cite
```

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Answered, and the grounding checks passed |
| `2` | Setup problem — no index, Ollama down, model mismatch |
| `3` | **Answered, but a grounding check failed** — a fabricated citation, or an unsupported claim |
| `130` | You pressed Ctrl-C |

Code `3` is the one to watch for in scripts.

---

## Measure quality

```powershell
askdocs eval               # recall@5 and MRR — fast, embedding model only
askdocs eval --refusal     # also checks it declines unanswerable questions (slow)
```

`eval` exits non-zero if recall falls below the threshold in `askdocs.toml`, so
it can gate a build the way a test suite does.

Edit the question set at `eval/questions.yaml`. Two rules: paraphrase the
questions rather than copying headings, and keep some deliberately unanswerable
ones.

---

## Inspect

```powershell
askdocs status             # corpus, index, models, connectors, skills
askdocs env                # every env var, its value, and where it came from
askdocs --version
askdocs --help
askdocs ask --help
```

`askdocs env` is the one to reach for when a setting seems not to apply — it
shows the source of every value.

---

## Settings

Precedence: **CLI flag → environment variable → `askdocs.toml` → default.**

### PowerShell

```powershell
# For one command
$env:ASKDOCS_GENERATE_MODEL = "qwen2.5:7b"; askdocs ask "..."

# For this whole terminal session
$env:ASKDOCS_GENERATE_MODEL = "qwen2.5:7b"
$env:ASKDOCS_OLLAMA_HOST    = "http://gpu-box:11434"
askdocs ask "..."

# Clear it again
Remove-Item Env:\ASKDOCS_GENERATE_MODEL
```

### Bash

```bash
ASKDOCS_GENERATE_MODEL=qwen2.5:7b askdocs ask "..."
export ASKDOCS_OLLAMA_HOST=http://gpu-box:11434
```

### The variables worth knowing

| Variable | Does |
|---|---|
| `ASKDOCS_GENERATE_MODEL` | Which model writes answers |
| `ASKDOCS_EMBED_MODEL` | Which model makes the numbers — **changing this invalidates the index** |
| `ASKDOCS_OLLAMA_HOST` | Point at another machine, e.g. one with a GPU |
| `ASKDOCS_RETRIEVE_TOP_K` | Chunks per answer |
| `ASKDOCS_CORPUS_DIR` | Where documents live |
| `ASKDOCS_INDEX_PATH` | Where the index is written |
| `ASKDOCS_GROUNDING_MIN_SCORE` | How strict the refuse-early gate is |
| `ASKDOCS_GENERATE_TEMPERATURE` | Raise above 0 for more variation — and more invention |

Run `askdocs env` for the full list of 24.

---

## Refreshing after the site changes

```powershell
# 1. Refetch changed pages — needs internet
cd D:\MCP\web-archiver
npm run dataset

# 2. Copy the updated dataset across
Copy-Item .\archive\www.aem.live\dataset.jsonl D:\MCP\askdocs\corpus\aem-live.jsonl

# 3. Re-embed only what changed — offline
cd D:\MCP\askdocs
askdocs index
```

---

## Development

```powershell
python -m pytest tests -q            # all 70 tests
python -m pytest tests -q -k normalize   # just the cleaning tests
python -m pytest tests -v            # verbose

# Run without installing
$env:PYTHONPATH = "src"; python -m askdocs.cli status
```

---

## When something is wrong

| Symptom | Try |
|---|---|
| `Could not reach Ollama` | `ollama serve` in another terminal |
| `Model ... is not available` | `ollama pull <name>` — the message gives the exact command |
| `No index at ...` | `askdocs index` |
| `Index was built with embedding model X but config says Y` | `askdocs index --rebuild` |
| `Nothing to index` | Put a `.jsonl` or `.md` file in `corpus/` |
| Answers are wrong or irrelevant | `askdocs ask "..." --dry-run` — check whether the right chunks are even being found. They usually are not, and that is a retrieval problem, not a prompt problem |
| Answers are too slow | Smaller model, or lower `--top-k`. A working GPU is the real fix |
| A setting seems ignored | `askdocs env` — it names the source of every value |

---

## Reading the query log

Every question is appended to `logs/queries.jsonl`.

```powershell
# Last 5 questions
Get-Content logs\queries.jsonl -Tail 5 | ForEach-Object { $_ | ConvertFrom-Json } | Format-List question, refused, grounded

# Every answer that failed a grounding check — the ones to investigate
Select-String -Path logs\queries.jsonl -Pattern '"grounded": false'
```
