# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is **empty** — only the project brief (`documentation/enonce.txt`, in French) and a report template (`documentation/Template+de+rapport+technique.docx`) exist. There is no code, no `requirements.txt`, no `Dockerfile`, and `readme.md` / `.gitignore` are both empty. Future Claude instances are expected to scaffold the project from scratch following the brief summarized below.

## What the project must deliver

OpenClassrooms Project 7 — a freelance data-science mission for fictional client **Puls-Events**. Deliverable is a **Proof of Concept RAG (Retrieval-Augmented Generation) chatbot** that answers user questions about upcoming cultural events. The brief from "Jérémy, Responsable technique" mandates a specific stack — these are not free choices:

- **LangChain** to orchestrate retrieval + generation
- **FAISS** (use `faiss-cpu`, not `faiss-gpu`, for portability — explicit instruction in the brief) as the vector store
- **Mistral** for both embeddings and the generation LLM (API key via env var / `.env`, never committed)
- **FastAPI or Flask** for the REST API (FastAPI is recommended in the brief for its auto-generated Swagger at `/docs`)
- **Docker** — the final API must run from a built image for the live demo
- **Python ≥ 3.8**

Data source is the **Open Agenda API** via `public.opendatasoft.com`. Scope: pick one geographic zone, restrict to events less than a year old (1 year of history + upcoming events).

## Required endpoints

- `POST /ask` — accepts a question, returns a generated answer grounded in the FAISS index
- `GET` or `POST` `/rebuild` — rebuilds the vector index on demand (treat as sensitive; the brief warns against exposing it publicly)

The brief explicitly states **no conversation history** is needed in the POC — keep `/ask` stateless.

## Required artifacts beyond code

These are graded deliverables, not nice-to-haves:

1. **Vector-index build script** runnable standalone (separate from the API runtime)
2. **Annotated Q/A test set** (reference questions + human-annotated answers) used to evaluate response quality
3. **Unit tests** covering data indexing and system performance, plus **automated evaluation metrics** — the brief suggests **Ragas** for similarity / faithfulness / context-recall scoring, runnable from a script like `evaluate_rag.py` and wireable into GitHub Actions
4. **Technical report** (PDF or README) following `documentation/Template+de+rapport+technique.docx`
5. **PowerPoint, 10–15 slides** for the oral defense

## Suggested repo layout (per the brief)

The brief asks for clear separation: folders for `scripts/` (preprocessing, vectorization, index build), `api/`, `tests/`, `documentation/`. Keep RAG business logic decoupled from the API layer so it can be imported as a function/class — the API is a thin transport wrapper.

## Environment & secrets

- Use `venv`, `conda`, or `poetry` — environment must be reproducible from `requirements.txt` or `environment.yml`
- Never commit `.env` or the Mistral API key
- The brief warns to verify FAISS/LangChain version compatibility — pin versions
- Test imports early as a smoke check: `faiss`, `langchain.vectorstores.FAISS`, `langchain.embeddings.HuggingFaceEmbeddings`, `mistral.MistralClient`

## Notes for working in this repo

- The authoritative spec is `documentation/enonce.txt` (French). When the user references "the brief", "l'énoncé", or a step number ("étape 3"), this is what they mean. There are 6 numbered étapes: (1) env setup, (2) Open Agenda ingestion + filtering, (3) chunking + FAISS indexing, (4) LangChain + Mistral RAG chain, (5) FastAPI/Flask API, (6) Docker + demo + slides.
- The `readme.md` to be authored is itself a deliverable — the brief calls for objectives, project structure, and reproduction instructions in it.
