"""Profilage des longueurs des champs textuels du dataset clean.

Mesure la distribution des longueurs (caractères + estimation en tokens) des
champs qui iront dans le `page_content` indexé par FAISS :

  - title_fr        (toujours présent)
  - description_fr  (toujours présent)
  - longdescription_fr (souvent présent, peut être long)
  - keywords_fr     (parfois présent)
  - conditions_fr   (parfois présent)
  - concaténation des 5 (ce qu'on embeddera en pratique)

L'enjeu : les modèles d'embedding sentence-transformers tronquent
silencieusement au-delà de leur fenêtre (souvent 512 tokens). Si beaucoup
d'événements dépassent, il faut soit chunker, soit choisir un modèle à
fenêtre plus large.

Estimation tokens : on prend caractères/4 (heuristique standard pour du
français — sentence-transformers utilise un sentencepiece ~équivalent au
BPE de GPT).

Script d'exploration jetable.

Exécution : `uv run python scripts/profile_lengths.py`
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("profile_lengths")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "events_clean_2026-05-21.jsonl"

# Champs qui composeront le page_content embeddé
CONTENT_FIELDS = (
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "keywords_fr",
    "conditions_fr",
)

# Seuils de tokens à vérifier (limites typiques des modèles HF multilingues)
TOKEN_THRESHOLDS = (256, 384, 512, 768, 1024, 2048)


def percentiles(sorted_values: list[int], qs: tuple[float, ...]) -> dict[float, int]:
    """Renvoie les percentiles à partir d'une liste DÉJÀ triée."""
    out = {}
    n = len(sorted_values)
    for q in qs:
        idx = min(int(q * n), n - 1)
        out[q] = sorted_values[idx]
    return out


def char_to_tokens(n_chars: int) -> int:
    """Estimation grossière : ~4 chars par token pour du français en BPE."""
    return n_chars // 4


def profile(input_path: Path) -> None:
    per_field_chars: dict[str, list[int]] = {f: [] for f in CONTENT_FIELDS}
    per_field_present: dict[str, int] = {f: 0 for f in CONTENT_FIELDS}
    concat_chars: list[int] = []
    n_total = 0
    longest_concat = (0, None)
    shortest_nontrivial_concat = (10**9, None)

    log.info("Lecture de %s", input_path.relative_to(PROJECT_ROOT))
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            ev = json.loads(line)
            n_total += 1

            parts: list[str] = []
            for field in CONTENT_FIELDS:
                value = ev.get(field)
                if value:
                    per_field_present[field] += 1
                    per_field_chars[field].append(len(value))
                    parts.append(value)

            concat = " ".join(parts)
            concat_chars.append(len(concat))
            if len(concat) > longest_concat[0]:
                longest_concat = (len(concat), ev.get("uid"))
            if 50 < len(concat) < shortest_nontrivial_concat[0]:
                shortest_nontrivial_concat = (len(concat), ev.get("uid"))

            if n_total % 200_000 == 0:
                log.info("  ... %d lignes", n_total)

    log.info("=== %d événements profilés ===", n_total)

    # Présence des champs
    log.info("")
    log.info("--- Présence des champs ---")
    for field, count in per_field_present.items():
        pct = 100 * count / n_total
        log.info("  %-22s %d (%.1f %%)", field, count, pct)

    # Distribution par champ (sur les events où le champ est présent)
    log.info("")
    log.info("--- Longueur en caractères par champ (non-null uniquement) ---")
    log.info(
        "  %-22s %8s %8s %8s %8s %8s %8s %10s",
        "champ", "P50", "P75", "P90", "P95", "P99", "max", "moy",
    )
    for field, values in per_field_chars.items():
        if not values:
            continue
        values.sort()
        p = percentiles(values, (0.5, 0.75, 0.9, 0.95, 0.99))
        avg = sum(values) // len(values)
        log.info(
            "  %-22s %8d %8d %8d %8d %8d %8d %10d",
            field, p[0.5], p[0.75], p[0.9], p[0.95], p[0.99], values[-1], avg,
        )

    # Distribution du concat
    log.info("")
    log.info("--- Concaténation (ce qu'on embeddera) ---")
    concat_chars.sort()
    p = percentiles(concat_chars, (0.5, 0.75, 0.9, 0.95, 0.99))
    avg = sum(concat_chars) // len(concat_chars)
    log.info(
        "  caractères : P50=%d  P75=%d  P90=%d  P95=%d  P99=%d  max=%d  moy=%d",
        p[0.5], p[0.75], p[0.9], p[0.95], p[0.99], concat_chars[-1], avg,
    )
    log.info(
        "  tokens (~) : P50=%d  P75=%d  P90=%d  P95=%d  P99=%d  max=%d  moy=%d",
        char_to_tokens(p[0.5]),
        char_to_tokens(p[0.75]),
        char_to_tokens(p[0.9]),
        char_to_tokens(p[0.95]),
        char_to_tokens(p[0.99]),
        char_to_tokens(concat_chars[-1]),
        char_to_tokens(avg),
    )

    # Combien d'events dépassent chaque seuil de tokens
    log.info("")
    log.info("--- Events dépassant un seuil de tokens (estimation) ---")
    for threshold in TOKEN_THRESHOLDS:
        max_chars = threshold * 4
        n_above = sum(1 for c in concat_chars if c > max_chars)
        pct = 100 * n_above / n_total
        log.info(
            "  > %4d tokens (~%5d chars) : %7d events (%.2f %%)",
            threshold, max_chars, n_above, pct,
        )

    # Exemples
    log.info("")
    log.info("--- Exemples ---")
    log.info(
        "  Plus long  : uid=%s  %d caractères (~%d tokens)",
        longest_concat[1], longest_concat[0], char_to_tokens(longest_concat[0]),
    )
    log.info(
        "  Plus court (>50 char)  : uid=%s  %d caractères",
        shortest_nontrivial_concat[1], shortest_nontrivial_concat[0],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Fichier JSONL d'entrée (défaut : {DEFAULT_INPUT.relative_to(PROJECT_ROOT)})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        log.error("Fichier introuvable : %s", args.input)
        return 1
    profile(args.input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
