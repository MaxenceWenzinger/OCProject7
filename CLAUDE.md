# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

OpenClassrooms Project 7 — **POC RAG chatbot for the fictional client Puls-Events**, answering user questions about cultural events ingested from the Open Agenda API. Epics 1 (env setup) and 2 (Open Agenda ingestion + cleaning) are **complete**. Current focus is Epic 3 (FAISS vectorization).

See `documentation/plan-de-travail.md` for the full epic / task breakdown and what's done vs pending.

## Stack and key decisions

The brief from "Jérémy, Responsable technique" mandates a specific stack. Some choices were tightened or deviated from during planning — those decisions are recorded here.

### Brief-mandated

- **LangChain** to orchestrate retrieval + generation
- **FAISS (`faiss-cpu`)** as the vector store (CPU-only for portability, explicit instruction in the brief)
- **FastAPI** for the REST API (recommended in the brief for its auto-generated Swagger at `/docs`)
- **Docker** — the final API must run from a built image for the live demo
- **Data source** : Open Agenda via `public.opendatasoft.com`

### Project-specific decisions (deviating from or refining the brief)

- **Python 3.12** (the brief said ≥ 3.8 — we pinned higher for modern typing). Managed via **uv** (`pyproject.toml` + `uv.lock` are the source of truth ; `requirements.txt` will only be generated at the end as a deliverable).
- **LLM**: Mistral local via **Ollama** (`langchain-ollama`), not the Mistral cloud API. Goal: offline demo, no API key, no internet dependency.
- **Embeddings**: HuggingFace local (a multilingual sentence-transformers model, French data).
- **Geographic scope**: **France entière**, not Île-de-France as the brief suggested. Validated with the professor.
- **Temporal scope**: **no date filter, ever**. Neither at ingestion nor at retrieval. Brief suggested ±1 year ; user explicitly chose to handle the full ~1 M event dataset because the engineering problem is more interesting.
- **CI**: GitHub Actions in scope (tests + Ragas eval), not punted to "future work".

## Repository layout

```
api/                 FastAPI app (Epic 5)
src/
  data/              ingestion + cleaning (Epic 2)
    clean.py         pure cleaning functions, no I/O
  indexing/          FAISS index building (Epic 3)
  rag/               retrieval-augmented generation chain (Epic 4)
scripts/             entry-point scripts; thin I/O wrappers around src/
  explore_openagenda.py     Epic 2.1 — API exploration / smoke tests
  fetch_openagenda.py       Epic 2.2 — streaming download to data/raw/
  measure_duplicates.py     Epic 2.4 — one-off dedup measurement
  clean_events.py           Epic 2.4 — full cleaning pipeline
tests/               pytest test suite
data/
  raw/               raw JSONL from Open Agenda (gitignored, ~2 GB)
  processed/         cleaned JSONL (gitignored, ~1.7 GB)
  interim/           intermediate artifacts (gitignored)
  index/             FAISS index (gitignored, regeneratable)
documentation/
  enonce.txt                authoritative spec, in French — "the brief"
  plan-de-travail.md        epic + task breakdown, decisions log
  data.md                   dataset reference: source, schema, cleaning
  Template+de+rapport+technique.docx   technical report template
evaluation/          Q/A dataset + Ragas evaluation (Epic 6)
```

**Conventions** : business logic in `src/`, runnable scripts in `scripts/`. The split lets `clean.py` and friends be tested in isolation with no filesystem dependency. Don't put logic in `scripts/` beyond argparse + log + I/O orchestration.

## Required endpoints (Epic 5)

- `POST /ask` — accepts a question, returns a generated answer grounded in the FAISS index. **Stateless** (no conversation history — brief explicitly says so for the POC).
- `POST /rebuild` — rebuilds the vector index on demand. Brief warns to treat this as sensitive ; not for public exposure.

## Required deliverables beyond code

Graded artifacts, not nice-to-haves:

1. **Vector-index build script** runnable standalone (separate from the API runtime)
2. **Annotated Q/A test set** (reference questions + human-annotated answers) used to evaluate response quality
3. **Unit tests** covering data indexing and system performance, plus **automated evaluation metrics** via **Ragas** (`evaluation/evaluate_rag.py`, wired into GitHub Actions)
4. **Technical report** (PDF or README) following `documentation/Template+de+rapport+technique.docx`
5. **PowerPoint, 10–15 slides** for the oral defense

## Environment

- `uv sync` is enough to install everything. Run commands via `uv run …`.
- `.env` is gitignored. No Mistral cloud key is needed (Ollama is local) — `.env` only matters if a future feature needs secrets.
- **Ollama is a separate prerequisite** (not in `uv sync`). The README documents `ollama pull mistral`.
- Smoke check: `uv run python scripts/check_env.py` (Epic 1.6).

## Working notes

### Open Agenda dataset specifics (learned the hard way)

These are non-obvious things about the data — saving you the discovery time:

- The full dataset is **~1.13 M events**, frozen by Opendatasoft in April 2024 (but contains events declared up to 2027).
- Use the **`/exports/jsonl`** endpoint (one streamable JSON object per line), not `/exports/json` which returns a single giant array.
- `country_fr` and `category` are dropped at the fetch stage : `country_fr` is constant ("France (Métropole)") by construction of the `where=` filter, and `category` is null on 100 % of the dataset.
- `keywords_fr` and `accessibility_label_fr` come as **`list`** types, not `str`. `clean.normalize_whitespace` handles both.
- Some descriptions contain **lone Unicode surrogates** (`\ud835` etc.) that crash UTF-8 writes. `clean.normalize_whitespace` strips them.
- `longdescription_fr` contains **HTML** ; `description_fr` is plain text.
- `attendancemode` is a **JSON string** of a nested object with multilingual labels, not a simple enum. `clean.parse_attendance_mode` extracts the `id` and maps to `"sur_place" | "en_ligne" | "mixte"`.
- ~98.5 % of events are `attendance_mode == "sur_place"`. The field is meant to be kept in metadata only (not embedded), to avoid diluting embeddings with a near-constant token.

### Glossary

- "the brief" / "l'énoncé" / "étape N" → `documentation/enonce.txt`. 6 étapes: (1) env setup, (2) Open Agenda ingestion + filtering, (3) chunking + FAISS, (4) LangChain + Mistral RAG, (5) FastAPI, (6) Docker + demo + slides.
- "the plan" → `documentation/plan-de-travail.md` (the user's epic/task breakdown — finer-grained than the 6 étapes).

### Misc

- The `readme.md` is itself a graded deliverable. It currently bootstraps install/usage ; it'll be finalized in Epic 8.7.
- Verify FAISS / LangChain version compatibility carefully (brief warns about this). Versions are pinned in `pyproject.toml`.
