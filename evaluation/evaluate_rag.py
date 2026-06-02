"""Évaluation automatisée du RAG sur le jeu de test annoté.

Charge `evaluation/qa_dataset.jsonl`, fait tourner `RAGService` sur chaque
question, calcule les métriques Ragas (faithfulness, answer_relevancy,
context_precision, context_recall) sur les questions in-domain, et un
check booléen séparé sur les questions out_of_domain. Exporte un CSV
détaillé (une ligne par question) + un JSON agrégé dans
`evaluation/results/run_<timestamp>/`.

Date système figée via `EVAL_FROZEN_DATE` (défaut `2026-06-02`,
date d'annotation du dataset) pour que les questions à expressions
relatives (« cet été », « ce week-end ») restent reproductibles.

Le LLM est celui choisi par `LLM_PROVIDER` (cf. `src/rag/llm.py`) —
même provider/modèle pour le RAG et pour le Ragas judge. Recommandé :
`mistral` (API cloud, défaut) pour la stabilité du JSON-strict
attendu par Ragas. `ollama` reste disponible pour des runs offline
mais produit des scores NaN sur les prompts Ragas que mistral-small
ne sait pas parser fiablement.

Utiliser `--sample N` pour itérer rapidement sur N questions tirées
au hasard (seed=42).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Date d'annotation du dataset. Fixe la date système du RAG quand la var
# n'est pas déjà définie par l'utilisateur — garantit que les questions à
# expressions temporelles relatives donnent les mêmes filtres extraits.
DEFAULT_FROZEN_DATE = "2026-06-02"
os.environ.setdefault("EVAL_FROZEN_DATE", DEFAULT_FROZEN_DATE)

# Ragas 0.4.x émet des DeprecationWarnings pour l'import depuis
# `ragas.metrics` au profit de `ragas.metrics.collections`. Le collections-API
# nécessite un `InstructorBaseRagasLLM` non disponible avec Ollama en 0.4.3 ;
# on reste sur l'API stable, ces warnings seront muets pendant le run.
warnings.filterwarnings("ignore", category=DeprecationWarning)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Charge `.env` (MISTRAL_API_KEY, LLM_PROVIDER, etc.) avant tout import LLM.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from datasets import Dataset  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.rag.llm import get_llm  # noqa: E402
from src.rag.service import RAGService  # noqa: E402

log = logging.getLogger(__name__)

QA_DATASET_PATH = PROJECT_ROOT / "evaluation" / "qa_dataset.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

# Pattern de refus attendu sur les questions out_of_domain (cf. règle 1 du
# prompt système dans src/rag/chain.py). Match insensible à la casse et
# tolérant aux variations d'apostrophe/espace.
OOD_REFUSAL_PATTERN = re.compile(
    r"je ne peux r[ée]pondre qu['’]?\s*[àa]\s*des questions sur les "
    r"[ée]v[ée]nements culturels",
    re.IGNORECASE,
)


@dataclass
class QAEntry:
    """Une ligne du `qa_dataset.jsonl` parsée."""

    id: str
    category: str
    question: str
    ground_truth: str
    expected_contexts: list[str]
    expected_filter: dict
    notes: str

    @classmethod
    def from_dict(cls, d: dict) -> QAEntry:
        return cls(
            id=d["id"],
            category=d["category"],
            question=d["question"],
            ground_truth=d["ground_truth"],
            expected_contexts=d["expected_contexts"],
            expected_filter=d["expected_filter"],
            notes=d.get("notes", ""),
        )


def load_qa_dataset(path: Path = QA_DATASET_PATH) -> list[QAEntry]:
    """Charge le `qa_dataset.jsonl` en mémoire."""
    entries = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entries.append(QAEntry.from_dict(json.loads(line)))
    return entries


def run_rag_on_entries(
    service: RAGService,
    entries: list[QAEntry],
) -> list[dict]:
    """Exécute le RAG sur chaque question, renvoie une liste de records bruts.

    Chaque record contient les champs nécessaires pour Ragas
    (`question`, `answer`, `retrieved_contexts`, `reference`) plus les
    métadonnées d'analyse (UIDs des sources, filtres extraits, timings)."""
    records = []
    for i, entry in enumerate(entries, 1):
        log.info("[%d/%d] %s (%s) — %s", i, len(entries), entry.id,
                 entry.category, entry.question[:80])
        t0 = time.perf_counter()
        result = service.answer(entry.question)
        elapsed = time.perf_counter() - t0

        retrieved_contexts = [s.page_content for s in result["sources"]]
        retrieved_uids = [s.metadata.get("uid", "") for s in result["sources"]]

        records.append({
            "id": entry.id,
            "category": entry.category,
            "question": entry.question,
            "ground_truth": entry.ground_truth,
            "expected_contexts": entry.expected_contexts,
            "expected_filter": entry.expected_filter,
            "notes": entry.notes,
            "answer": result["answer"],
            "retrieved_contexts": retrieved_contexts,
            "retrieved_uids": retrieved_uids,
            "filters_used": result["filters_used"],
            "filter_relaxed": result["filter_relaxed"],
            "timings": result["timings"],
            "elapsed_s": elapsed,
        })
        log.info("  → %d sources, filters=%s, %.1fs",
                 len(retrieved_contexts), result["filters_used"], elapsed)
    return records


def score_ood(records: list[dict]) -> list[dict]:
    """Score les réponses out_of_domain : la réponse contient-elle le refus ?

    Renvoie une liste de dicts `{id, passed, answer_excerpt}` pour les
    seuls records OOD. Le pattern est défini en haut du module."""
    results = []
    for r in records:
        if r["category"] != "out_of_domain":
            continue
        passed = bool(OOD_REFUSAL_PATTERN.search(r["answer"]))
        results.append({
            "id": r["id"],
            "passed": passed,
            "answer_excerpt": r["answer"][:200],
        })
    return results


def build_ragas_dataset(in_domain_records: list[dict]) -> Dataset:
    """Construit un HF Dataset au format attendu par Ragas.

    Champs requis : `question`, `answer`, `contexts` (list[str]),
    `ground_truth` (string). Ragas 0.4.x accepte ces noms historiques ;
    `EvaluationDataset.from_list` ferait l'équivalent avec les noms
    modernes (`user_input`, `response`, `retrieved_contexts`, `reference`)."""
    return Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["retrieved_contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in in_domain_records
    ])


def run_ragas_eval(in_domain_records: list[dict]) -> dict:
    """Lance Ragas avec le même LLM que le RAG (cohérence via `LLM_PROVIDER`).

    Renvoie un dict `{question_id → {metric → score}}`. Le LLM et les
    embeddings sont enveloppés par les wrappers Ragas pour compatibilité
    avec LangChain."""
    provider = os.getenv("LLM_PROVIDER", "mistral")
    log.info("Préparation des wrappers Ragas (provider=%s)...", provider)
    llm_wrapper = LangchainLLMWrapper(get_llm())
    embeddings_wrapper = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
        )
    )

    dataset = build_ragas_dataset(in_domain_records)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # Sérialisation (max_workers=1) pour les deux providers :
    # - Ollama est mono-process par nature.
    # - Mistral API : avec 4 workers en parallèle, les 429 « rate limit » et
    #   « service tier capacity exceeded » (code 3505, transient, côté infra
    #   Mistral) se cumulent et provoquent des TimeoutError. En série on
    #   absorbe naturellement les pauses sans faire grimper l'horloge Ragas.
    # Timeout 300s laisse à un job lent (jusqu'à 8 retries httpx avec backoff
    #   exponentiel jusqu'à 16s) le temps de terminer avant que Ragas n'expire.
    run_config = RunConfig(timeout=300, max_workers=1, max_retries=3)

    log.info("Lancement Ragas sur %d questions × %d métriques...",
             len(in_domain_records), len(metrics))
    t0 = time.perf_counter()
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm_wrapper,
        embeddings=embeddings_wrapper,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )
    log.info("Ragas terminé en %.1fs", time.perf_counter() - t0)

    # `result` est un EvaluationResult ; le `.scores` est une liste
    # alignée sur les lignes du dataset. On reconstruit un dict id → scores.
    scores_by_id: dict[str, dict[str, float]] = {}
    for record, scores in zip(in_domain_records, result.scores):
        scores_by_id[record["id"]] = {k: _safe_float(v) for k, v in scores.items()}
    return scores_by_id


def _safe_float(v) -> float | None:
    """Convertit en float ou renvoie None pour NaN/None (sérialisable JSON)."""
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def write_per_question_csv(
    records: list[dict],
    scores_by_id: dict[str, dict[str, float]],
    ood_results: list[dict],
    out_path: Path,
) -> None:
    """Écrit le détail par question dans un CSV.

    Une ligne par question avec : id, category, question, answer (tronquée),
    retrieved_uids, filters_used, filter_relaxed, elapsed_s, et les scores
    Ragas (NaN pour OOD) + ood_passed (NaN pour in-domain)."""
    import csv

    ood_passed_by_id = {r["id"]: r["passed"] for r in ood_results}
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "id", "category", "question", "answer",
            "retrieved_uids", "expected_contexts",
            "filters_used", "expected_filter", "filter_relaxed",
            *metric_names, "ood_passed",
            "elapsed_s",
        ])
        for r in records:
            scores = scores_by_id.get(r["id"], {})
            writer.writerow([
                r["id"],
                r["category"],
                r["question"],
                r["answer"][:500],
                "|".join(r["retrieved_uids"]),
                "|".join(r["expected_contexts"]),
                json.dumps(r["filters_used"], ensure_ascii=False),
                json.dumps(r["expected_filter"], ensure_ascii=False),
                r["filter_relaxed"],
                *[_csv_score(scores.get(m)) for m in metric_names],
                ood_passed_by_id.get(r["id"], ""),
                f"{r['elapsed_s']:.2f}",
            ])


def _csv_score(v) -> str:
    return "" if v is None else f"{v:.4f}"


def write_summary_json(
    records: list[dict],
    scores_by_id: dict[str, dict[str, float]],
    ood_results: list[dict],
    out_path: Path,
    frozen_date: str,
    sample_n: int | None,
) -> None:
    """Écrit l'agrégat (moyennes globales + par catégorie + OOD) en JSON."""
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # Agrégats globaux (in-domain uniquement)
    in_domain = [r for r in records if r["category"] != "out_of_domain"]
    global_means = {
        m: _mean([scores_by_id.get(r["id"], {}).get(m) for r in in_domain])
        for m in metric_names
    }

    # Agrégats par catégorie
    by_category: dict[str, dict[str, float | int]] = {}
    cats = sorted({r["category"] for r in records})
    for cat in cats:
        cat_records = [r for r in records if r["category"] == cat]
        if cat == "out_of_domain":
            n_passed = sum(1 for o in ood_results if o["passed"])
            by_category[cat] = {
                "n": len(cat_records),
                "n_passed": n_passed,
                "pass_rate": n_passed / len(cat_records) if cat_records else None,
            }
        else:
            by_category[cat] = {"n": len(cat_records)}
            for m in metric_names:
                by_category[cat][m] = _mean(
                    [scores_by_id.get(r["id"], {}).get(m) for r in cat_records]
                )

    # Métadonnées du run (commit, date système figée, etc.). Les défauts
    # affichés ici doivent rester en phase avec ceux de `src/rag/llm.py`.
    from src.rag.llm import DEFAULT_MISTRAL_MODEL, DEFAULT_OLLAMA_MODEL
    provider = os.environ.get("LLM_PROVIDER", "mistral")
    if provider == "mistral":
        llm_model = os.environ.get("MISTRAL_MODEL", DEFAULT_MISTRAL_MODEL)
    elif provider == "ollama":
        llm_model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    else:
        llm_model = "?"
    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "frozen_date": frozen_date,
        "sample_n": sample_n,
        "llm_provider": provider,
        "llm_model": llm_model,
        "git_commit": _git_commit(),
        "n_questions_total": len(records),
        "n_questions_in_domain": len(in_domain),
        "n_questions_out_of_domain": len(ood_results),
        "global_means_in_domain": global_means,
        "by_category": by_category,
        "timings": {
            "mean_elapsed_s": _mean([r["elapsed_s"] for r in records]),
            "total_rag_s": sum(r["elapsed_s"] for r in records),
        },
        "filter_relaxed_count": sum(1 for r in records if r["filter_relaxed"]),
    }

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)


def _mean(values) -> float | None:
    """Moyenne en ignorant les None ; renvoie None si tout est None."""
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _git_commit() -> str | None:
    """Renvoie le SHA court du HEAD ou None si pas dans un repo git."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sample", type=int, default=None,
        help="N'évaluer que N questions tirées au hasard (seed=42). Sans flag, run complet.",
    )
    parser.add_argument(
        "--dataset", type=Path, default=QA_DATASET_PATH,
        help="Chemin du qa_dataset.jsonl à utiliser (défaut: evaluation/qa_dataset.jsonl).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Dossier de sortie. Par défaut : evaluation/results/run_<timestamp>/.",
    )
    parser.add_argument(
        "--skip-ragas", action="store_true",
        help="Lance le RAG mais saute les métriques Ragas (utile pour debug).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    frozen_date = os.environ["EVAL_FROZEN_DATE"]
    log.info("EVAL_FROZEN_DATE=%s (date système figée pour le RAG)", frozen_date)
    log.info("LLM_PROVIDER=%s (LLM utilisé partout : RAG + Ragas judge)",
             os.getenv("LLM_PROVIDER", "mistral"))

    entries = load_qa_dataset(args.dataset)
    log.info("Chargé %d questions depuis %s", len(entries), args.dataset)

    if args.sample is not None:
        random.seed(42)
        entries = random.sample(entries, min(args.sample, len(entries)))
        entries.sort(key=lambda e: e.id)
        log.info("Mode --sample %d : %d questions retenues (ids=%s)",
                 args.sample, len(entries), [e.id for e in entries])

    out_dir = args.out_dir or (
        RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Sortie : %s", out_dir)

    log.info("Initialisation RAGService (~20s)...")
    service = RAGService()

    records = run_rag_on_entries(service, entries)
    ood_results = score_ood(records)
    log.info("OOD : %d/%d refus correctement formulés",
             sum(1 for r in ood_results if r["passed"]), len(ood_results))

    in_domain_records = [r for r in records if r["category"] != "out_of_domain"]
    if args.skip_ragas or not in_domain_records:
        scores_by_id = {}
        log.info("Ragas sauté (--skip-ragas ou 0 question in-domain)")
    else:
        scores_by_id = run_ragas_eval(in_domain_records)

    csv_path = out_dir / "per_question.csv"
    json_path = out_dir / "summary.json"
    write_per_question_csv(records, scores_by_id, ood_results, csv_path)
    write_summary_json(records, scores_by_id, ood_results, json_path,
                       frozen_date=frozen_date, sample_n=args.sample)

    log.info("Écrit %s", csv_path)
    log.info("Écrit %s", json_path)

    # Récap console
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    with json_path.open(encoding="utf-8") as fh:
        summary = json.load(fh)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
