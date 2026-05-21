"""Téléchargement du dataset Open Agenda.

Streame l'export JSONL d'Opendatasoft pour le dataset
`evenements-publics-openagenda`, applique les filtres
(France métropolitaine, description non-null), et écrit la sortie ligne par
ligne dans `data/raw/events_<YYYY-MM-DD>.jsonl`.

Endpoint utilisé : `/exports/jsonl` plutôt que `/exports/json` — chaque ligne
est un objet JSON indépendant, ce qui permet un vrai streaming (pas besoin de
charger 1 M d'événements en mémoire).

Robustesse :
  - retries automatiques avec backoff exponentiel sur les erreurs réseau /
    5xx via `urllib3.Retry` (couvre coupure connexion + erreurs serveur) ;
  - timeout connect/read séparés ;
  - écriture atomique : on télécharge dans un fichier `.tmp` puis on renomme
    une fois le stream terminé, pour ne jamais laisser un fichier partiel
    sous un nom « propre ».

Exécution : `uv run python scripts/fetch_openagenda.py`
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_openagenda")

EXPORT_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "evenements-publics-openagenda/exports/jsonl"
)

# Colonnes retenues pour le RAG (cf. documentation/data.md).
# `country_fr` n'est PAS dans le select : il est constant ("France (Métropole)")
# par construction du filtre where=, inutile de l'embarquer dans chaque ligne.
# `category` n'est PAS dans le select : champ vide à 100 % dans le snapshot
# Opendatasoft (vérifié sur 1 051 298 events).
SELECT_FIELDS = [
    # Identité & traçabilité
    "uid",
    "slug",
    "canonicalurl",
    # Contenu textuel
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "keywords_fr",
    "conditions_fr",
    # Dates
    "firstdate_begin",
    "firstdate_end",
    "lastdate_begin",
    "lastdate_end",
    # Localisation
    "location_name",
    "location_address",
    "location_city",
    "location_postalcode",
    "location_department",
    "location_region",
    "location_coordinates",
    # Public
    "age_min",
    "age_max",
    "accessibility_label_fr",
    "attendancemode",
]

# Filtres appliqués côté API (-7,4 % de hors-périmètre éliminés).
WHERE_CLAUSE = (
    'country_fr="France (Métropole)" '
    "AND description_fr IS NOT NULL"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Timeouts : (connect, read). On laisse confortable car le serveur peut être
# lent à initialiser un gros export.
TIMEOUT = (10, 60)
CHUNK_SIZE = 64 * 1024  # 64 KB — équilibre RAM / nombre de syscalls


def build_session() -> requests.Session:
    """Session avec retries + backoff exponentiel sur 429/5xx et erreurs réseau."""
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,  # 0s, 2s, 4s, 8s, 16s
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch(output_path: Path, session: requests.Session) -> tuple[int, int]:
    """Télécharge l'export et l'écrit ligne par ligne. Retourne (n_lignes, n_octets)."""
    params = {
        "select": ",".join(SELECT_FIELDS),
        "where": WHERE_CLAUSE,
    }
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    log.info("GET %s", EXPORT_URL)
    log.info("  select  : %d champs", len(SELECT_FIELDS))
    log.info("  where   : %s", WHERE_CLAUSE)

    n_lines = 0
    n_bytes = 0
    progress_step = 50_000
    next_progress = progress_step
    with session.get(EXPORT_URL, params=params, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining:
            log.info("  rate-limit restant : %s appels", remaining)
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                fh.write(chunk)
                n_bytes += len(chunk)
                n_lines += chunk.count(b"\n")
                if n_lines >= next_progress:
                    log.info(
                        "  ... %d lignes / %.1f MB",
                        n_lines,
                        n_bytes / (1024 * 1024),
                    )
                    next_progress = ((n_lines // progress_step) + 1) * progress_step

    tmp_path.replace(output_path)
    return n_lines, n_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RAW_DIR,
        help="Répertoire de sortie (défaut : data/raw/)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nom de fichier (défaut : events_<YYYY-MM-DD>.jsonl)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    filename = args.name or f"events_{date.today().isoformat()}.jsonl"
    output_path = args.out_dir / filename

    session = build_session()
    log.info("=== Téléchargement Open Agenda ===")
    log.info("Sortie : %s", output_path.relative_to(PROJECT_ROOT))

    n_lines, n_bytes = fetch(output_path, session)

    log.info("=== Terminé ===")
    log.info("Lignes écrites : %d", n_lines)
    log.info("Taille fichier : %.1f MB", n_bytes / (1024 * 1024))
    log.info("Chemin         : %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
