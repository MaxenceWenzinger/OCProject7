"""Benchmark du modèle d'embedding sur des events réels.

Charge le modèle HuggingFace (par défaut `intfloat/multilingual-e5-base`),
embed N events du dataset clean en batch, mesure le débit et extrapole
la durée pour le dataset complet.

Sert à :
  - vérifier que le modèle se télécharge et se charge sans erreur ;
  - valider la dimension du vecteur retourné ;
  - dimensionner le temps total du build d'index ;
  - comparer rapidement plusieurs modèles via `--model`.

Exécution :
    uv run python scripts/benchmark_embeddings.py
    uv run python scripts/benchmark_embeddings.py --model intfloat/multilingual-e5-base --n 200
    uv run python scripts/benchmark_embeddings.py --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --n 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.build_documents import event_to_document  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark_embeddings")

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "events_clean_2026-05-21.jsonl"
DATASET_FULL_SIZE = 252_901


def load_sample(input_path: Path, n: int) -> list[str]:
    """Lit les N premiers events du clean et retourne la liste des
    `page_content` à embedder."""
    texts: list[str] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            ev = json.loads(line)
            doc = event_to_document(ev)
            texts.append(doc.page_content)
    return texts


def benchmark(model_name: str, texts: list[str], batch_size: int) -> dict:
    """Charge le modèle puis embed la liste, retourne stats."""
    # Import paresseux : charger HuggingFaceEmbeddings prend ~10 s
    from langchain_huggingface import HuggingFaceEmbeddings

    log.info("Chargement du modèle : %s", model_name)
    t0 = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": batch_size},
    )
    load_time = time.perf_counter() - t0
    log.info("  → chargé en %.1f s", load_time)

    # Smoke : une seule requête pour valider la dimension
    log.info("Smoke test (1 requête)...")
    vec = embeddings.embed_query("Concert de jazz à Paris en juin 2025")
    dim = len(vec)
    log.info("  → vecteur de dimension %d", dim)

    # Benchmark sur N events en batch
    log.info("Embedding de %d events en batch...", len(texts))
    t0 = time.perf_counter()
    vectors = embeddings.embed_documents(texts)
    embed_time = time.perf_counter() - t0

    n = len(texts)
    rate = n / embed_time
    log.info("  → %d events en %.1f s (%.1f events/s)", n, embed_time, rate)

    # Extrapolation
    extrapolated = DATASET_FULL_SIZE / rate
    h = int(extrapolated // 3600)
    m = int((extrapolated % 3600) // 60)
    s = int(extrapolated % 60)
    log.info(
        "Extrapolation pour les %d events du dataset : ~%dh %02dm %02ds",
        DATASET_FULL_SIZE, h, m, s,
    )

    return {
        "model": model_name,
        "load_time_s": load_time,
        "dim": dim,
        "n_embedded": n,
        "embed_time_s": embed_time,
        "events_per_s": rate,
        "extrapolated_full_s": extrapolated,
        "vectors_sample_size": len(vectors),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Nom HuggingFace du modèle")
    parser.add_argument("--n", type=int, default=100, help="Nombre d'events à embedder")
    parser.add_argument("--batch", type=int, default=32, help="Taille de batch")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        log.error("Fichier introuvable : %s", args.input)
        return 1

    log.info("=== Benchmark embeddings ===")
    log.info("Modèle : %s", args.model)
    log.info("Sample : %d events depuis %s", args.n, args.input.relative_to(PROJECT_ROOT))
    log.info("Batch  : %d", args.batch)
    log.info("")

    texts = load_sample(args.input, args.n)
    log.info("Longueurs de texte (chars) : min=%d  moy=%d  max=%d",
             min(len(t) for t in texts),
             sum(len(t) for t in texts) // len(texts),
             max(len(t) for t in texts))

    benchmark(args.model, texts, args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
