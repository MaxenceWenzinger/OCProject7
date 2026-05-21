"""Nettoyage des événements Open Agenda.

Fonctions pures sans I/O — le wrapper qui lit/écrit les fichiers est
`scripts/clean_events.py`. Cette séparation permet de tester `clean_event` et
`is_valid` directement sur le fixture `data/raw/sample_events.json` (cf. `tests/test_clean.py`).

Pipeline appliqué à chaque événement :
  1. Strip HTML sur `longdescription_fr` et `conditions_fr` (BeautifulSoup
     + `html.parser` natif → aucune dépendance ajoutée).
  2. Décodage des entités HTML (`&amp;`, `&nbsp;`, ...) via `html.unescape`.
  3. Normalisation des espaces : `\\s+` → un seul espace, trim.
  4. Parsing du champ `attendancemode` (JSON imbriqué) → champ dérivé
     `attendance_mode: "sur_place" | "en_ligne" | "mixte" | None`.
  5. Ajout du champ dérivé `event_year` extrait de `firstdate_begin`.
  6. Validation finale (`is_valid`) : titre non vide, date plausible,
     description non vide.

La déduplication (clé `(title_fr, firstdate_begin, location_name)`, `scripts/measure_duplicates.py`) est appliquée dans le wrapper
streaming car elle nécessite un état global (set de clés vues).
"""

from __future__ import annotations

import html
import json
import re
import warnings
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

# Certains champs Open Agenda ressemblent à des URLs ; BeautifulSoup émet alors
# un avertissement non pertinent dans notre contexte (on ne fetch rien).
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

HTML_FIELDS: tuple[str, ...] = ("longdescription_fr", "conditions_fr")
TEXT_FIELDS: tuple[str, ...] = (
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "keywords_fr",
    "conditions_fr",
    "location_name",
    "location_address",
    "location_city",
    "location_department",
    "location_region",
    "accessibility_label_fr",
)

# Champs qu'on retire activement à l'entrée si le raw les contient encore
# (ancien raw d'avant qu'on les sorte du select=). country_fr est constant
# ("France (Métropole)") par construction du filtre where=, et category est
# null à 100 % dans le snapshot Opendatasoft.
DROPPED_FIELDS: tuple[str, ...] = ("country_fr", "category")

ATTENDANCE_MAP: dict[int, str] = {
    1: "sur_place",
    2: "en_ligne",
    3: "mixte",
}

# Fenêtre temporelle plausible : on filtre les années aberrantes vues en P7-2.1
# (1900, 23, 2503, 2032...). Les events 2027–2030 sont rares mais légitimes.
VALID_YEAR_MIN = 2010
VALID_YEAR_MAX = 2030

_WHITESPACE_RE = re.compile(r"\s+")

# Certains organisateurs collent des caractères Unicode mal encodés (surrogates
# isolés U+D800–U+DFFF) dans leurs descriptions, ce qui plante l'écriture
# UTF-8. On les supprime — c'est de la donnée corrompue, pas du texte légitime.
_LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def strip_html(text: str | None) -> str | None:
    """Supprime les balises HTML et décode les entités. Préserve None."""
    if text is None:
        return None
    soup = BeautifulSoup(text, "html.parser")
    # `separator=" "` évite que "<p>foo</p><p>bar</p>" devienne "foobar".
    plain = soup.get_text(separator=" ")
    return html.unescape(plain)


def normalize_whitespace(text: str | list | None) -> str | None:
    """Collapse les espaces multiples et trim. Préserve None mais transforme
    une chaîne devenue vide après nettoyage en None (cohérent avec le choix
    'garder null' validé en planning).

    Certains champs Open Agenda (`keywords_fr`, `accessibility_label_fr`)
    arrivent sous forme de liste de strings — on les joint avec ", "."""
    if text is None:
        return None
    if isinstance(text, list):
        text = ", ".join(str(item) for item in text if item)
    text = _LONE_SURROGATE_RE.sub("", text)
    cleaned = _WHITESPACE_RE.sub(" ", text).strip()
    return cleaned or None


def parse_attendance_mode(raw: str | None) -> str | None:
    """Le champ `attendancemode` brut est une chaîne JSON imbriquée du type
    `{"id": 1, "label": {"fr": "Sur place", ...}}`. On en extrait juste
    l'`id` qu'on mappe vers une enum simple."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return ATTENDANCE_MAP.get(data.get("id"))


def extract_year(iso_datetime: str | None) -> int | None:
    """Renvoie l'année à partir d'une date ISO 8601, ou None si non parsable."""
    if not iso_datetime:
        return None
    try:
        return datetime.fromisoformat(iso_datetime).year
    except (ValueError, TypeError):
        return None


def is_valid(event: dict[str, Any]) -> bool:
    """Garde-fou final : on rejette les events sans titre, sans description
    utilisable, ou avec une date aberrante.

    Note : on ne rejette PAS sur l'absence de longdescription_fr — beaucoup
    d'events n'ont qu'une description courte et c'est OK (~10,7 % des events
    n'ont pas de longdescription)."""
    if not event.get("title_fr"):
        return False
    if not event.get("description_fr"):
        return False
    year = event.get("event_year")
    if year is None:
        return False
    if not (VALID_YEAR_MIN <= year <= VALID_YEAR_MAX):
        return False
    return True


def clean_event(raw: dict[str, Any]) -> dict[str, Any]:
    """Applique le pipeline complet à un événement brut. Retourne un nouveau
    dict (pas de mutation in-place). Le résultat peut ensuite passer
    `is_valid` pour décider s'il est conservé."""
    cleaned = dict(raw)

    for field in DROPPED_FIELDS:
        cleaned.pop(field, None)

    for field in HTML_FIELDS:
        if field in cleaned:
            cleaned[field] = strip_html(cleaned[field])

    for field in TEXT_FIELDS:
        if field in cleaned:
            cleaned[field] = normalize_whitespace(cleaned[field])

    cleaned["attendance_mode"] = parse_attendance_mode(cleaned.pop("attendancemode", None))
    cleaned["event_year"] = extract_year(cleaned.get("firstdate_begin"))

    return cleaned


def dedup_key(event: dict[str, Any]) -> tuple[str, str, str] | None:
    """Clé de déduplication : (title, date, location_name).
    Renvoie None si l'un des trois champs manque — dans ce cas l'événement
    est conservé sans être comparé."""
    title = event.get("title_fr")
    date_ = event.get("firstdate_begin")
    loc = event.get("location_name")
    if not (title and date_ and loc):
        return None
    return (title.lower(), date_, loc.lower())
