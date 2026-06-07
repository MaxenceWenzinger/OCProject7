# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

OpenClassrooms Project 7 — **POC RAG chatbot for the fictional client Puls-Events**, answering user questions about cultural events ingested from the Open Agenda API. Epics 1 (env setup), 2 (Open Agenda ingestion + cleaning), 3 (FAISS vectorization with parent-child chunking), 4 (LangChain + Mistral RAG chain with self-querying pre-filter) and 5 (FastAPI exposing `/ask` and `/rebuild`) are **complete**. Epic 6 is partially complete: 6.1 (annotated Q/A dataset, 30 questions in 5 categories), 6.2 (schema enrichment), 6.3 (`evaluate_rag.py` script with Ragas), 6.4 (`documentation/evaluation.md` + baseline run) are all **done**. Current focus is Epic 6.5 (quality iterations on the 5 findings identified by the baseline).

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
- **LLM**: **Mistral via Cloud API** (`langchain-mistralai`, default `mistral-medium-3.5`). Initial choice was Mistral-small local via Ollama for an offline POC ; switched to API on 2026-06-02 after Epic 6.3 revealed that mistral-small produced (a) hallucinated dates in self-querying (q01 « Quand a lieu X » → fabricated date_after/date_before), (b) JSON-malformed outputs that Ragas couldn't parse (~40 % failure on internal prompts). Free tier rate limits actually checked via API headers — `mistral-medium-3.5` retained over `mistral-large-latest` (4 req/min) and `mistral-medium-latest` (2508, 23 req/min) because it offers 50 req/min and lower /ask latency. Ollama remains supported via `LLM_PROVIDER=ollama` for offline experiments. Demo Docker is now online-only — documented in the report as a tradeoff.
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dim, 128-token window), loaded locally via `langchain-huggingface`. Picked over `intfloat/multilingual-e5-base` (~9× slower on CPU) because parent-child chunking makes the short window a non-issue. See `documentation/data.md` for the benchmark.
- **Chunking strategy**: **parent-child**. Each event's `page_content` is split into N chunks of ≤120 MiniLM tokens with 24-token overlap. Chunks get embedded into FAISS; at retrieval time we dedup by `parent_uid` and serve the **full parent Document** to the LLM. Mentor's recommendation, calibrated against the dataset's token distribution.
- **Geographic scope**: **France entière**, not Île-de-France as the brief suggested. Validated with the professor.
- **Temporal scope**: keep only events whose **last occurrence ends in 2025 or later** — purely past events (ending in 2024 or earlier) are dropped. Filter applied at the cleaning stage, on `lastdate_end` with fallback to `firstdate_end`. No filter at retrieval. This trims the ~1 M raw events down to ~253 k. Brief suggested ±1 year ; first decision was "no temporal filter ever", revised on 2026-05-21 to focus on current/upcoming events while keeping the France-wide geographic scope.
- **CI**: GitHub Actions in scope (tests + Ragas eval), not punted to "future work".

## Repository layout

```
api/
  main.py                          FastAPI app : lifespan + routes /, /ask, /rebuild, /rebuild/status
  schemas.py                       Pydantic : AskRequest/Response, Source, RebuildResponse, HealthResponse
  rebuild.py                       Bearer auth dependency + run_rebuild() job + RebuildState (save/load JSON)
src/
  data/
    clean.py                       pure cleaning functions, no I/O
  indexing/
    build_documents.py             event → Document(s) (parent + chunks)
  rag/
    llm.py                         get_llm() → ChatMistralAI (default) or ChatOllama, dispatched by LLM_PROVIDER
    chain.py                       LCEL chain {context, question} → str (prompt + LLM + parser)
    query_parser.py                self-querying LLM extractor → QueryFilters (Pydantic, today-aware)
    retrieval.py                   load index/parent_store/LUT + retrieve_parents (pre-filter incl. dates)
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
  test_api.py                      14 tests (functional, TestClient, mocked RAGService + job)
data/
  raw/                             raw JSONL from Open Agenda (gitignored, ~2 GB)
  processed/                       cleaned JSONL (gitignored, ~400 MB)
  interim/                         intermediate artifacts (gitignored)
  index/                           FAISS index + parent_store + uid→faiss_ids LUT + rebuild_state.json (gitignored, ~1.5 GB, regeneratable)
documentation/
  enonce.txt                       authoritative spec, in French — "the brief"
  plan-de-travail.md               epic + task breakdown, decisions log
  data.md                          dataset + index reference: source, schema, cleaning, FAISS build
  evaluation.md                    Ragas methodology, implementation choices, baseline run, findings
  Template+de+rapport+technique.docx   technical report template
evaluation/
  qa_dataset.jsonl                 30 hand-annotated Q/R in 5 categories
  evaluate_rag.py                  RAGService + Ragas runner; CSV+JSON outputs; `--sample N`, `--skip-ragas`
  results/                         dated subfolders `run_<ts>/` with per_question.csv + summary.json (gitignored)
.env / .env.example                ADMIN_TOKEN, MISTRAL_API_KEY (.env gitignored, .env.example committed)
```

**Conventions** : business logic in `src/`, runnable scripts in `scripts/`. The split lets `clean.py`, `build_documents.py` and the `rag/` modules be tested in isolation with no filesystem dependency. Don't put logic in `scripts/` beyond argparse + log + I/O orchestration.

**Two kinds of scripts**:
1. **Pipeline scripts** (`fetch_openagenda.py`, `clean_events.py`, `build_index.py`) — the three steps to rebuild the index from scratch. Run them in order; each picks up the output of the previous via `data/`.
2. **Exploration scripts** (`profile_lengths.py`, `benchmark_embeddings.py`) — one-off measurements that guided design decisions. Kept in the repo for traceability but not part of the production pipeline.

## API endpoints (Epic 5, live)

- `GET /` — health-check (`HealthResponse { status, rag_ready }`).
- `POST /ask` — `AskRequest { question }` → `AskResponse { answer, sources[], filters_used, filter_relaxed }`. **Stateless** (no conversation history — brief explicitly says so for the POC). Returns `503` if a rebuild is running, `422` on empty question.
- `POST /rebuild` — Bearer-protected, fires `fetch + clean + build + hot-swap` as a `BackgroundTask`, returns `202` immediately. `409` if already running, `401` on bad/missing token, `503` if `ADMIN_TOKEN` env var is unset (fail-secure).
- `GET /rebuild/status` — unprotected, returns `in_progress / started_at / finished_at / last_error`. State persists across restarts via `data/index/rebuild_state.json`.

## Required deliverables beyond code

Graded artifacts, not nice-to-haves:

1. **Vector-index build script** runnable standalone (separate from the API runtime)
2. **Annotated Q/A test set** (reference questions + human-annotated answers) used to evaluate response quality
3. **Unit tests** covering data indexing and system performance, plus **automated evaluation metrics** via **Ragas** (`evaluation/evaluate_rag.py`, wired into GitHub Actions)
4. **Technical report** (PDF or README) following `documentation/Template+de+rapport+technique.docx`
5. **PowerPoint, 10–15 slides** for the oral defense

## Environment

- `uv sync` is enough to install everything. Run commands via `uv run …`.
- **`.env` is required** (gitignored). Copy `.env.example` and fill:
  - `ADMIN_TOKEN` (Bearer token for `POST /rebuild`). Generate via `uv run python -c "import secrets; print(secrets.token_urlsafe(32))"`. If unset at startup, the lifespan logs a loud warning and `/rebuild` answers `503` (fail-secure).
  - `MISTRAL_API_KEY` (obligatory when `LLM_PROVIDER=mistral`, the default). Create at https://console.mistral.ai/. If unset and provider is mistral, `get_llm()` raises a clear `RuntimeError` at first call.
- **Ollama is an optional prerequisite** for `LLM_PROVIDER=ollama` (offline fallback, not in `uv sync`). Install via `ollama pull mistral-small:latest`.

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
- **Parent-child invariant**: each chunk Document carries `parent_uid` (duplicated from `uid` for clarity) and `chunk_index` (debug). At retrieval, dedup the FAISS hits by `parent_uid` and pull the full text from `parent_store` before passing to the LLM. `RAGService` defaults are `k_chunks=30 / k_parents=10` (ratio 3, dataset chunks/event median 2.3 — comfortable margin before dedup).
- **MiniLM token-budget tightness**: chunks are sized at 120 tokens to leave room for the 2 special tokens `[CLS]`/`[SEP]` the tokenizer adds automatically — the model's hard limit is 128. Going to 128 would silently truncate. The chunk size is enforced via the **real MiniLM tokenizer** (not character estimation) in `event_to_chunks`.
- **`build_index.py` is not unit-tested**; integration coverage lives in `tests/test_indexing.py` which spins up the real MiniLM + FAISS on a 5-event fixture (~40 s). It's marked `@pytest.mark.slow` so `pytest -m "not slow"` skips it for tight dev loops.
- **Don't try to chunk-and-embed in a single pass without batching**: the API rate limit on HF Hub is fine, but loading the model + holding 580k chunks + their embeddings in RAM all at once would blow memory. `build_index.py` batches by 5000 chunks and adds them incrementally via `db.add_documents()`.

### RAG chain specifics

- **LCEL, not `RetrievalQA`**: the legacy class is deprecated since LangChain 0.2 and can't host our parent-child dedup. The chain in `chain.py` is intentionally minimal (`{context, question} | prompt | llm | StrOutputParser`) — retrieval lives in Python (`retrieval.py`), invoked by `RAGService` before the chain. Tradeoff is documented in `plan-de-travail.md` (Epic 4 decisions).
- **Self-querying via LLM, today-aware**: `query_parser.py` uses `get_llm().with_structured_output(QueryFilters)` to extract `{city, region, date_after, date_before}` — the underlying chat model is whatever `LLM_PROVIDER` resolves to. A mentioned year has **no dedicated field** (the old `year` filter was removed on 2026-06-07): it is resolved by the LLM into `date_after=YYYY-01-01` / `date_before=YYYY-12-31` bounds, so year filtering uses the same `first_date`/`last_date` overlap as any date window instead of an exact-equality match on a `event_year` derived from the start date alone. The system prompt receives the current date and weekday via `RunnablePassthrough.assign` on each invocation (not frozen at startup — the API may run for days). Date resolution lookup order: env var `EVAL_FROZEN_DATE` (set by `evaluate_rag.py` for reproducibility, unset in production) → `date.today()`. The LLM is told to resolve relative dates (« ce dimanche », « cet été ») into ISO dates, to infer `date_after = today` when the main clause is present/future and no date is given, and to inspect the main clause's grammatical tense — not subordinate context clauses (e.g. « j'ai entendu qu'il y avait... pourrais-tu m'en lister » is a future request). « N'importe quand » is the explicit escape hatch for nulling both date bounds. If extraction throws, `RAGService.answer` degrades gracefully to empty filters.
- **Pre-filter on FAISS, all four fields**: FAISS doesn't support metadata filtering natively — LangChain's `filter=` parameter is post-filter, which broke for rare cities (e.g. `city="Reims"` returned 0 because the 200 most-similar chunks to "jazz" are all Paris/Lyon). We do real pre-filtering via a `uid → list[faiss_id]` LUT built once (~1.6s) and disk-cached as `data/index/uid_to_faiss_ids.pkl` with mtime invalidation against `index.faiss`. At query time: select allowed uids from parent_store, expand to faiss_ids, `reconstruct_batch` the vectors, compute L2 distance in numpy. ~370ms/query. **All four filters (city / region / date_after / date_before) are evaluated as pre-filter** against the parent metadata (`first_date` / `last_date` overlap with the requested window). Date post-filtering was tried first but was inadequate: on Paris × June-Oct 2026, the top-15 most-similar chunks contained ~1 in the date window, so the user got 1 source instead of 10. Iterating the parent_store on dates is cheap (~same complexity as city/region).
- **Fail-open retrieval**: if the extracted filter returns 0 parents, we re-run the similarity search with no filter and set `filter_relaxed=True` in the returned dict. A bad filter (typo, hallucinated city) shouldn't blackhole the response.
- **Region aliases**: the dataset has both French and English region names (`"Bretagne"` vs `"Brittany"`, `"Normandie"` vs `"Normandy"`). `retrieval.REGION_ALIASES` normalizes English to French; combined with accent-insensitive matching this absorbs the variants.
- **`RAGService.answer(q)` returns rich dict**: `{answer, sources, filters_used, filter_relaxed, timings}`. `timings` breaks down extract/retrieve/generate/total in ms — useful for the technical report and for the upcoming FastAPI endpoint to expose.
- **Two LLM calls per `/ask`**: with Mistral API (`mistral-medium-3.5`, default) ~1s extract + ~2-3s generate = ~3-5s warm in isolation. Under Ragas load (concurrent quota pressure), observed up to ~13s/question on the baseline run. With Ollama (`mistral-small`, opt-in) ~7s + ~11s = ~18s warm on the dev machine.
- **LLM provider dispatch in `llm.py`**: `get_llm()` reads `LLM_PROVIDER` (`mistral` default, `ollama` fallback) and returns the right `BaseChatModel`. All chains (`chain.py`, `query_parser.py`) and the Ragas judge use it — switching the env var changes the whole stack at once. Imports of `ChatMistralAI` and `ChatOllama` are lazy inside the dispatch functions so a missing prereq doesn't break import time. When provider is `mistral`, `_build_mistral` also installs three idempotent monkey-patches required for Ragas + Mistral free-tier interop: retry on HTTP 429/5xx, strip of markdown ` ```json ``` ` fences around outputs, and recursive aggregation of nested `token_usage` dicts (the latter is a real `langchain-mistralai` bug surfaced by `answer_relevancy`'s batch generation). All three live in `src/rag/llm.py` and are detailed in `documentation/evaluation.md`.
- **Empty `__init__.py` files in `src/rag/`** — don't re-export anything from there. Each module is imported directly (`from src.rag.service import RAGService`).

### API specifics

- **Lifespan + `app.state`**: the FastAPI lifespan instantiates the `RAGService` singleton, loads `RebuildState.load()` from disk, and warns if `ADMIN_TOKEN` is missing. All routes pull from `request.app.state` — no globals, mock-friendly in tests.
- **`/ask` 503 conditions**: returns 503 if `rebuild_in_progress=True` (an index swap is in flight) or if `rag_service is None` (lifespan didn't finish — typically only seen when bypassing lifespan in tests).
- **`/rebuild` runs the full pipeline**: not just `build_index`. Interprets "reconstruire la base vectorielle" as `fetch + clean + build + hot-swap`. Calls business functions directly from the scripts (`scripts.fetch_openagenda.fetch`, `scripts.clean_events.clean_stream`, `scripts.build_index.build`) — no `rebuild_index.py` wrapper script. Imports are lazy inside `run_rebuild` so the API startup doesn't pay sentence-transformers loading cost twice.
- **Hot-swap safety**: at the end of the job, a new `RAGService()` is instantiated, **then** `app.state.rag_service` is replaced. If the new instance throws (e.g. corrupted index after a partial write), the old service stays in place and the error is stored in `state.last_error`. The flag is always cleared in `finally`.
- **`rebuild_state.json` persistence**: lives in `data/index/rebuild_state.json` (gitignored). Contains `started_at / finished_at / last_error`. `in_progress` is intentionally NOT persisted — a crash mid-rebuild should not leave the API stuck answering 503 forever. Loaded once at lifespan startup. The current file was hand-crafted: `finished_at = mtime(index.faiss)`, `started_at = finished_at - 1h58` (~12 min fetch + 4 min clean + 1h42 build). A real `/rebuild` run rewrites it automatically.
- **Bearer auth**: `verify_admin_token` dependency uses `HTTPBearer(auto_error=False)` so we control the 401 response shape (default is 403 when header is missing). Comparison via `secrets.compare_digest` (constant-time). Swagger's Authorize lock accepts the raw token (no `Bearer ` prefix — it's added automatically).
- **Two response views on the same parent Document**: the LLM sees the full `page_content` (title + description + longdescription + keywords + conditions) via `format_docs` in `chain.py`. The API user sees a metadata-only `Source` projection + a short description extracted on-the-fly from `page_content` (2nd block after splitting on `\n\n`, skipping the title and the `Mots-clés :`/`Conditions :` blocks, capped at 300 chars). `description_fr` is not in the FAISS metadata to save disk space, but it IS recoverable from `page_content`.
- **Test layer mocks**: `tests/test_api.py` uses `MagicMock()` for `RAGService` and `monkeypatch.setattr("api.main.run_rebuild", ...)` to stub the 2h job. `monkeypatch.setenv("ADMIN_TOKEN", ...)` rather than touching `.env`. 14 tests in ~3.5s, marked `not slow`, runs in CI without Ollama/FAISS/MiniLM loaded.

### Test layout specifics

- **Shared session-scoped fixtures in `tests/conftest.py`**: `fixture_events` (5 thematically distinct events) and `built_index` (FAISS index built once, ~10-15s). `test_indexing.py` and `test_rag.py` both consume them — the index build is paid once per pytest session, not once per file.
- **`tests/test_rag.py` auto-skips if Ollama is down**: `ollama_available` fixture pings `localhost:11434/api/tags` with a 2s timeout and calls `pytest.skip(...)` on failure. CI without Ollama shows SKIPPED, not FAILED. The test is the only thing in the repo with a hard runtime dependency on Ollama, and it's `slow`-marked.

### Glossary

- "the brief" / "l'énoncé" / "étape N" → `documentation/enonce.txt`. 6 étapes: (1) env setup, (2) Open Agenda ingestion + filtering, (3) chunking + FAISS, (4) LangChain + Mistral RAG, (5) FastAPI, (6) Docker + demo + slides.
- "the plan" → `documentation/plan-de-travail.md` (the user's epic/task breakdown — finer-grained than the 6 étapes).

### Misc

- The `readme.md` is itself a graded deliverable. It currently bootstraps install/usage ; it'll be finalized in Epic 8.7.
- Verify FAISS / LangChain version compatibility carefully (brief warns about this). Versions are pinned in `pyproject.toml`.
