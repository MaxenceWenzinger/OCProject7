# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

OpenClassrooms Project 7 — **POC RAG chatbot for the fictional client Puls-Events**, answering user questions about cultural events ingested from the Open Agenda API. Epics 1 (env setup), 2 (Open Agenda ingestion + cleaning), 3 (FAISS vectorization with parent-child chunking) and 4 (LangChain + Mistral-via-Ollama RAG chain with self-querying pre-filter) are **complete**. Current focus is Epic 5 (FastAPI exposing `/ask` and `/rebuild`).

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
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dim, 128-token window), loaded locally via `langchain-huggingface`. Picked over `intfloat/multilingual-e5-base` (~9× slower on CPU) because parent-child chunking makes the short window a non-issue. See `documentation/data.md` for the benchmark.
- **Chunking strategy**: **parent-child**. Each event's `page_content` is split into N chunks of ≤120 MiniLM tokens with 24-token overlap. Chunks get embedded into FAISS; at retrieval time we dedup by `parent_uid` and serve the **full parent Document** to the LLM. Mentor's recommendation, calibrated against the dataset's token distribution.
- **Geographic scope**: **France entière**, not Île-de-France as the brief suggested. Validated with the professor.
- **Temporal scope**: keep only events whose **last occurrence ends in 2025 or later** — purely past events (ending in 2024 or earlier) are dropped. Filter applied at the cleaning stage, on `lastdate_end` with fallback to `firstdate_end`. No filter at retrieval. This trims the ~1 M raw events down to ~253 k. Brief suggested ±1 year ; first decision was "no temporal filter ever", revised on 2026-05-21 to focus on current/upcoming events while keeping the France-wide geographic scope.
- **CI**: GitHub Actions in scope (tests + Ragas eval), not punted to "future work".

## Repository layout

```
api/                 FastAPI app (Epic 5, not started yet)
src/
  data/
    clean.py                       pure cleaning functions, no I/O
  indexing/
    build_documents.py             event → Document(s) (parent + chunks)
  rag/
    llm.py                         get_llm() → ChatOllama (mistral-small:latest)
    chain.py                       LCEL chain {context, question} → str (prompt + LLM + parser)
    query_parser.py                self-querying LLM extractor → QueryFilters (Pydantic)
    retrieval.py                   load index/parent_store/LUT + retrieve_parents (pre-filter)
    service.py                     RAGService orchestrating extract → retrieve → generate
scripts/             thin I/O wrappers around src/
  fetch_openagenda.py              streaming download to data/raw/
  clean_events.py                  full cleaning pipeline
  build_index.py                   FAISS index build with parent-child chunking
  profile_lengths.py               one-off text-length profiling (exploration)
  benchmark_embeddings.py          one-off embedding model benchmark
tests/
  conftest.py                      shared fixtures (events + built_index, session-scoped)
  test_clean.py                    48 tests (unit)
  test_build_documents.py          24 tests (unit, fake tokenizer)
  test_indexing.py                 9 tests (integration, real MiniLM + FAISS, marked `slow`)
  test_rag.py                      5 tests (E2E, real Ollama, marked `slow`, auto-skip if Ollama down)
data/
  raw/                             raw JSONL from Open Agenda (gitignored, ~2 GB)
  processed/                       cleaned JSONL (gitignored, ~400 MB)
  interim/                         intermediate artifacts (gitignored)
  index/                           FAISS index + parent_store + uid→faiss_ids LUT (gitignored, ~1.5 GB, regeneratable)
documentation/
  enonce.txt                       authoritative spec, in French — "the brief"
  plan-de-travail.md               epic + task breakdown, decisions log
  data.md                          dataset + index reference: source, schema, cleaning, FAISS build
  Template+de+rapport+technique.docx   technical report template
evaluation/                        Q/A dataset + Ragas evaluation (Epic 6, not started yet)
```

**Conventions** : business logic in `src/`, runnable scripts in `scripts/`. The split lets `clean.py`, `build_documents.py` and the `rag/` modules be tested in isolation with no filesystem dependency. Don't put logic in `scripts/` beyond argparse + log + I/O orchestration.

**Two kinds of scripts**:
1. **Pipeline scripts** (`fetch_openagenda.py`, `clean_events.py`, `build_index.py`) — the three steps to rebuild the index from scratch. Run them in order; each picks up the output of the previous via `data/`.
2. **Exploration scripts** (`profile_lengths.py`, `benchmark_embeddings.py`) — one-off measurements that guided design decisions. Kept in the repo for traceability but not part of the production pipeline.

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
- **Ollama is a separate prerequisite** (not in `uv sync`). The README documents `ollama pull mistral-small:latest`.

## Working notes

### Open Agenda dataset specifics (learned the hard way)

These are non-obvious things about the data — saving you the discovery time:

- The full dataset is **~1.13 M events** and is actively maintained — ~161k events were updated in 2025 and ~148k in 2026. The `modified` field of the Opendatasoft catalog says April 2024 but that's misleading: the `updatedat` field of individual records tells the real story.
- Use the **`/exports/jsonl`** endpoint (one streamable JSON object per line), not `/exports/json` which returns a single giant array.
- `country_fr` and `category` are dropped at the fetch stage : `country_fr` is constant ("France (Métropole)") by construction of the `where=` filter, and `category` is null on 100 % of the dataset.
- `keywords_fr` and `accessibility_label_fr` come as **`list`** types, not `str`. `clean.normalize_whitespace` handles both.
- Some descriptions contain **lone Unicode surrogates** (`\ud835` etc.) that crash UTF-8 writes. `clean.normalize_whitespace` strips them.
- `longdescription_fr` contains **HTML** ; `description_fr` is plain text.
- `attendancemode` is a **JSON string** of a nested object with multilingual labels, not a simple enum. `clean.parse_attendance_mode` extracts the `id` and maps to `"sur_place" | "en_ligne" | "mixte"`.
- ~98.5 % of events are `attendance_mode == "sur_place"`. The field is meant to be kept in metadata only (not embedded), to avoid diluting embeddings with a near-constant token.

### Indexing specifics

- **Index layout on disk** (in `data/index/`): three files, always together — `index.faiss` (the FAISS binary, ~890 MB), `index.pkl` (LangChain mapping vector_id → chunk Document, ~395 MB), and `parent_store.pkl` (our addition: dict `uid → parent Document`, ~289 MB). The first two are written by `FAISS.save_local()` and reloaded together via `FAISS.load_local()`. The third is our additional pickle, loaded separately at API startup for the parent-child join.
- **Parent-child invariant**: each chunk Document carries `parent_uid` (duplicated from `uid` for clarity) and `chunk_index` (debug). At retrieval, dedup the FAISS hits by `parent_uid` and pull the full text from `parent_store` before passing to the LLM. Top-K of ~15 chunks is needed to consistently yield ~5 distinct parents (ratio is ~2.3 chunks/event).
- **MiniLM token-budget tightness**: chunks are sized at 120 tokens to leave room for the 2 special tokens `[CLS]`/`[SEP]` the tokenizer adds automatically — the model's hard limit is 128. Going to 128 would silently truncate. The chunk size is enforced via the **real MiniLM tokenizer** (not character estimation) in `event_to_chunks`.
- **`build_index.py` is not unit-tested**; integration coverage lives in `tests/test_indexing.py` which spins up the real MiniLM + FAISS on a 5-event fixture (~40 s). It's marked `@pytest.mark.slow` so `pytest -m "not slow"` skips it for tight dev loops.
- **Don't try to chunk-and-embed in a single pass without batching**: the API rate limit on HF Hub is fine, but loading the model + holding 580k chunks + their embeddings in RAM all at once would blow memory. `build_index.py` batches by 5000 chunks and adds them incrementally via `db.add_documents()`.

### RAG chain specifics

- **LCEL, not `RetrievalQA`**: the legacy class is deprecated since LangChain 0.2 and can't host our parent-child dedup. The chain in `chain.py` is intentionally minimal (`{context, question} | prompt | llm | StrOutputParser`) — retrieval lives in Python (`retrieval.py`), invoked by `RAGService` before the chain. Tradeoff is documented in `plan-de-travail.md` (Epic 4 decisions).
- **Self-querying via LLM**: `query_parser.py` uses `ChatOllama.with_structured_output(QueryFilters)` to extract `{city, region, year, date_after, date_before}` from the user question. Costs an extra LLM call (~6-8s on mistral-small JSON-strict, slower than free-text generation per-token). If extraction throws, `RAGService.answer` degrades gracefully to empty filters — extraction is best-effort, not a hard contract.
- **Pre-filter on FAISS**: FAISS doesn't support metadata filtering natively — LangChain's `filter=` parameter is post-filter, which broke for rare cities (e.g. `city="Reims"` returned 0 because the 200 most-similar chunks to "jazz" are all Paris/Lyon). We do real pre-filtering via a `uid → list[faiss_id]` LUT built once (~1.6s) and disk-cached as `data/index/uid_to_faiss_ids.pkl` with mtime invalidation against `index.faiss`. At query time: select allowed uids from parent_store, expand to faiss_ids, `reconstruct_batch` the vectors, compute L2 distance in numpy. ~370ms/query, beats post-filter for rare filters. Date filters stay post-filter (per-date LUT would be disproportionate for a POC).
- **Fail-open retrieval**: if the extracted filter returns 0 parents, we re-run the similarity search with no filter and set `filter_relaxed=True` in the returned dict. A bad filter (typo, hallucinated city) shouldn't blackhole the response.
- **Region aliases**: the dataset has both French and English region names (`"Bretagne"` vs `"Brittany"`, `"Normandie"` vs `"Normandy"`). `retrieval.REGION_ALIASES` normalizes English to French; combined with accent-insensitive matching this absorbs the variants.
- **`RAGService.answer(q)` returns rich dict**: `{answer, sources, filters_used, filter_relaxed, timings}`. `timings` breaks down extract/retrieve/generate/total in ms — useful for the technical report and for the upcoming FastAPI endpoint to expose.
- **Two LLM calls per `/ask`**: extract (~7s) + generate (~11s) = ~18s warm on the dev machine. Acceptable for POC, documented in `plan-de-travail.md`. If we ever want to cut latency: stream the generation (perceived latency ~2s), or use a smaller model for extraction only.
- **Empty `__init__.py` files in `src/rag/`** — don't re-export anything from there. Each module is imported directly (`from src.rag.service import RAGService`).

### Test layout specifics

- **Shared session-scoped fixtures in `tests/conftest.py`**: `fixture_events` (5 thematically distinct events) and `built_index` (FAISS index built once, ~10-15s). `test_indexing.py` and `test_rag.py` both consume them — the index build is paid once per pytest session, not once per file.
- **`tests/test_rag.py` auto-skips if Ollama is down**: `ollama_available` fixture pings `localhost:11434/api/tags` with a 2s timeout and calls `pytest.skip(...)` on failure. CI without Ollama shows SKIPPED, not FAILED. The test is the only thing in the repo with a hard runtime dependency on Ollama, and it's `slow`-marked.

### Glossary

- "the brief" / "l'énoncé" / "étape N" → `documentation/enonce.txt`. 6 étapes: (1) env setup, (2) Open Agenda ingestion + filtering, (3) chunking + FAISS, (4) LangChain + Mistral RAG, (5) FastAPI, (6) Docker + demo + slides.
- "the plan" → `documentation/plan-de-travail.md` (the user's epic/task breakdown — finer-grained than the 6 étapes).

### Misc

- The `readme.md` is itself a graded deliverable. It currently bootstraps install/usage ; it'll be finalized in Epic 8.7.
- Verify FAISS / LangChain version compatibility carefully (brief warns about this). Versions are pinned in `pyproject.toml`.
