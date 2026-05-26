"""Construction de l'index FAISS parent-child à partir des events clean.

Pipeline :

  data/processed/events_clean_*.jsonl
        │
        ▼ (streaming, ligne par ligne)
  event_to_chunks (tokenizer MiniLM, 120 tokens, overlap 24)
        │
        ▼
  accumulation par batchs de N chunks
        │
        ▼
  HuggingFaceEmbeddings (paraphrase-multilingual-MiniLM-L12-v2)
        │
        ▼
  FAISS : 1er batch → from_documents ; suivants → add_documents
        │
        ▼ (à la fin)
  save_local → data/index/index.faiss + data/index/index.pkl
  pickle    → data/index/parent_store.pkl  ({uid → Document parent})

Le `parent_store` est nécessaire à l'inférence (Epic 4) : la similarity
search FAISS retourne des chunks, on dédoublonne par parent_uid et on
récupère le Document parent complet pour le passer au LLM.

Écriture atomique : on construit dans `data/index/.tmp/` et on swap à la
fin pour ne jamais laisser un index partiel sous le nom canonique.

Exécution :
    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --limit 1000   # test rapide
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.build_documents import event_to_chunks, event_to_document  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_index")

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "events_clean_2026-05-21.jsonl"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE_CHUNKS = 5000  # ~52 s par batch attendu sur CPU MiniLM


def latest_clean_file() -> Path:
    candidates = sorted((PROJECT_ROOT / "data" / "processed").glob("events_clean_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            "Aucun fichier events_clean_*.jsonl trouvé. "
            "Lance d'abord `uv run python scripts/clean_events.py`."
        )
    return candidates[-1]


def stream_chunks_and_parents(
    input_path: Path,
    tokenizer,
    limit: int | None,
):
    """Générateur : pour chaque event, yield (list[chunks], parent_doc).
    Le caller accumule les chunks par batch, et conserve les parents dans
    un dict pour le parent_store final."""
    with input_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            event = json.loads(line)
            chunks = event_to_chunks(event, tokenizer)
            if not chunks:
                continue
            parent_doc = event_to_document(event)
            yield chunks, parent_doc


def build(input_path: Path, index_dir: Path, batch_size: int, limit: int | None) -> dict:
    # Import paresseux : sentence-transformers prend ~5s à charger
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import SentenceTransformer

    log.info("Chargement du modèle %s ...", MODEL_NAME)
    t0 = time.perf_counter()
    # On charge SentenceTransformer pour exposer le tokenizer au chunker,
    # et HuggingFaceEmbeddings pour l'API LangChain (mêmes poids partagés
    # via le cache HF, donc pas de double download).
    st_model = SentenceTransformer(MODEL_NAME, device="cpu")
    tokenizer = st_model.tokenizer
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 32},
    )
    log.info("  → modèle chargé en %.1f s", time.perf_counter() - t0)

    # Préparation du répertoire temporaire (écriture atomique)
    tmp_dir = index_dir / ".tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    parent_store: dict[str, "Document"] = {}  # noqa: F821 (forward-ref Document)
    db: "FAISS | None" = None
    chunk_buffer: list = []
    n_events_seen = 0
    n_chunks_total = 0
    n_batches = 0
    t_build_start = time.perf_counter()

    def flush_batch(force: bool = False) -> None:
        nonlocal db, chunk_buffer, n_chunks_total, n_batches
        if not chunk_buffer:
            return
        if not force and len(chunk_buffer) < batch_size:
            return

        n_batches += 1
        t_batch = time.perf_counter()
        if db is None:
            db = FAISS.from_documents(chunk_buffer, embeddings)
        else:
            db.add_documents(chunk_buffer)
        elapsed = time.perf_counter() - t_batch
        n_chunks_total += len(chunk_buffer)

        # ETA basée sur le rythme cumulé depuis le début
        total_elapsed = time.perf_counter() - t_build_start
        rate = n_chunks_total / total_elapsed if total_elapsed > 0 else 0
        log.info(
            "Batch %d : +%d chunks en %.1fs (total %d chunks, %d events, rate %.0f ch/s)",
            n_batches, len(chunk_buffer), elapsed, n_chunks_total, n_events_seen, rate,
        )
        chunk_buffer = []

    log.info("Streaming %s ...", _safe_relative(input_path))
    for chunks, parent_doc in stream_chunks_and_parents(input_path, tokenizer, limit):
        n_events_seen += 1
        chunk_buffer.extend(chunks)
        uid = parent_doc.metadata["uid"]
        parent_store[uid] = parent_doc
        flush_batch(force=False)

    # Vide le dernier batch partiel
    flush_batch(force=True)

    if db is None:
        log.error("Aucun chunk produit, index non créé.")
        shutil.rmtree(tmp_dir)
        return {"n_events": 0, "n_chunks": 0}

    # Sauvegarde FAISS
    log.info("Sauvegarde de l'index FAISS dans %s ...", _safe_relative(tmp_dir))
    db.save_local(str(tmp_dir))

    # Sauvegarde du parent_store
    parent_path = tmp_dir / "parent_store.pkl"
    log.info("Sauvegarde du parent_store (%d events) ...", len(parent_store))
    with parent_path.open("wb") as fh:
        pickle.dump(parent_store, fh)

    # Swap atomique : si un index canonique existe déjà, on le remplace
    for child in tmp_dir.iterdir():
        target = index_dir / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(child), str(target))
    tmp_dir.rmdir()

    total = time.perf_counter() - t_build_start
    log.info("=== Terminé ===")
    log.info("Events lus       : %d", n_events_seen)
    log.info("Chunks indexés   : %d", n_chunks_total)
    log.info("Ratio chunks/ev  : %.2f", n_chunks_total / max(1, n_events_seen))
    log.info("Durée build      : %s", _fmt_duration(total))
    log.info("Index sauvegardé : %s", _safe_relative(index_dir))

    return {
        "n_events": n_events_seen,
        "n_chunks": n_chunks_total,
        "n_parents": len(parent_store),
        "duration_s": total,
    }


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=None,
                        help="Fichier clean d'entrée (défaut : dernier events_clean_*.jsonl)")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR,
                        help="Répertoire de sortie pour l'index (défaut : data/index/)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_CHUNKS,
                        help=f"Taille de batch en chunks (défaut : {BATCH_SIZE_CHUNKS})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limite du nombre d'events à indexer (pour tests rapides)")
    return parser.parse_args()


def _safe_relative(path: Path) -> str:
    """Affiche un chemin relatif au PROJECT_ROOT si possible, absolu sinon."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    input_path = args.input or latest_clean_file()
    if not input_path.exists():
        log.error("Fichier d'entrée introuvable : %s", input_path)
        return 1
    args.index_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Build FAISS index parent-child ===")
    log.info("Entrée    : %s", _safe_relative(input_path))
    log.info("Sortie    : %s", _safe_relative(args.index_dir))
    log.info("Modèle    : %s", MODEL_NAME)
    log.info("Batch     : %d chunks", args.batch_size)
    if args.limit:
        log.info("Limit     : %d events (mode test)", args.limit)

    build(input_path, args.index_dir, args.batch_size, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
