"""Pipeline de nettoyage du dataset Open Agenda.

Lit `data/raw/events_<date>.jsonl` ligne par ligne, applique
`src.data.clean.clean_event` puis `is_valid`, déduplique via la clé
`(title, date, location_name)`, et écrit la sortie
dans `data/processed/events_clean_<date>.jsonl`.

Ne charge jamais tout le dataset en mémoire (1 M d'événements, 2,16 GB de
raw) : streaming JSONL en entrée, JSONL en sortie, set de clés vues comme
seul état global.

Stats finales : nombre d'events lus, rejetés (par règle), dédupliqués,
gardés. Écriture atomique via `.tmp` → rename.

Exécution : `uv run python scripts/clean_events.py`
Options : `--input <path>`, `--output <path>`
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import date
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clean import clean_event, dedup_key, is_valid  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clean_events")

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"


def latest_raw_file() -> Path:
    """Trouve le fichier raw le plus récent (events_*.jsonl) si --input absent."""
    candidates = sorted(DEFAULT_RAW_DIR.glob("events_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"Aucun fichier events_*.jsonl trouvé dans {DEFAULT_RAW_DIR}. "
            "Lance d'abord `uv run python scripts/fetch_openagenda.py`."
        )
    return candidates[-1]


def _invalid_reason(event: dict) -> str:
    """Pourquoi un event a été rejeté (pour les stats)."""
    if not event.get("title_fr"):
        return "no_title"
    if not event.get("description_fr"):
        return "no_description"
    year = event.get("event_year")
    if year is None:
        return "no_year"
    return "year_out_of_range"


def clean_stream(input_path: Path, output_path: Path) -> dict[str, int]:
    """Streame input → output en appliquant clean_event + is_valid + dédup.
    Retourne un dict de stats."""
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    seen_keys: set[tuple[str, str, str]] = set()
    rejection_reasons: Counter[str] = Counter()
    n_read = 0
    n_kept = 0
    n_dup = 0
    n_invalid = 0
    n_bytes_out = 0

    with input_path.open("r", encoding="utf-8") as fin, tmp_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            n_read += 1
            raw = json.loads(line)
            cleaned = clean_event(raw)

            if not is_valid(cleaned):
                n_invalid += 1
                rejection_reasons[_invalid_reason(cleaned)] += 1
                continue

            key = dedup_key(cleaned)
            if key is not None:
                if key in seen_keys:
                    n_dup += 1
                    continue
                seen_keys.add(key)

            out_line = json.dumps(cleaned, ensure_ascii=False) + "\n"
            fout.write(out_line)
            n_bytes_out += len(out_line.encode("utf-8"))
            n_kept += 1

            if n_read % 100_000 == 0:
                log.info(
                    "  ... %d lus / %d gardés / %d invalides / %d doublons",
                    n_read,
                    n_kept,
                    n_invalid,
                    n_dup,
                )

    tmp_path.replace(output_path)
    return {
        "read": n_read,
        "kept": n_kept,
        "invalid": n_invalid,
        "duplicate": n_dup,
        "bytes_out": n_bytes_out,
        **{f"invalid_{reason}": count for reason, count in rejection_reasons.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Fichier JSONL d'entrée (défaut : dernier events_*.jsonl de data/raw/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Fichier JSONL de sortie (défaut : data/processed/events_clean_<date>.jsonl)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input or latest_raw_file()
    if not input_path.exists():
        log.error("Fichier d'entrée introuvable : %s", input_path)
        return 1

    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or DEFAULT_OUT_DIR / f"events_clean_{date.today().isoformat()}.jsonl"

    log.info("=== Cleaning Open Agenda ===")
    log.info("Entrée : %s", input_path.relative_to(PROJECT_ROOT))
    log.info("Sortie : %s", output_path.relative_to(PROJECT_ROOT))

    stats = clean_stream(input_path, output_path)

    log.info("=== Terminé ===")
    log.info("Lus       : %d", stats["read"])
    log.info("Gardés    : %d (%.1f %%)", stats["kept"], 100 * stats["kept"] / stats["read"])
    log.info(
        "Invalides : %d (%.1f %%)",
        stats["invalid"],
        100 * stats["invalid"] / stats["read"],
    )
    for key, value in sorted(stats.items()):
        if key.startswith("invalid_"):
            log.info("  - %-30s %d", key.removeprefix("invalid_"), value)
    log.info("Doublons  : %d (%.1f %%)", stats["duplicate"], 100 * stats["duplicate"] / stats["read"])
    log.info("Taille fichier sortie : %.1f MB", stats["bytes_out"] / (1024 * 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
