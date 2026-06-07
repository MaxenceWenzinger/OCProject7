"""Fixtures pytest partagées entre les tests d'intégration RAG.

Centralise :
- Le mini-dataset d'événements (`fixture_events`) — 5 events réalistes
  thématiquement distincts (jazz, peinture, théâtre, randonnée, cuisine)
  avec metadata complète (ville, région, dates) pour exercer le filtrage.
- L'index FAISS construit une fois par session (`built_index`) puisque
  le build prend ~10-40s même sur 5 events à cause du chargement MiniLM.
  Les tests d'indexation et de RAG partagent ce coût.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


@pytest.fixture(scope="session", autouse=True)
def _load_env() -> None:
    """Charge `.env` pour toute la session de tests.

    Les tests E2E (`test_rag.py`) ont besoin de `MISTRAL_API_KEY` (provider
    par défaut) ou `LLM_PROVIDER=ollama` ; `os.getenv` ne lit pas `.env`
    automatiquement, contrairement à l'API et au runner d'éval qui appellent
    `load_dotenv` explicitement. On reproduit ce comportement ici pour que
    `pytest` local voie la clé. En CI, la clé vient des secrets (env réel),
    `.env` est absent et `load_dotenv` est silencieux."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")


# 5 events réalistes avec thématiques distinctes pour valider que le retrieval
# discrimine. Chaque event a un mot-clé unique qu'aucun autre ne contient,
# et des metadata variées (ville, région, dates) pour exercer le pre-filter.
_FIXTURE_EVENTS: list[dict] = [
    {
        "uid": "ev-jazz",
        "slug": "concert-jazz-bataclan",
        "canonicalurl": "https://example.org/jazz",
        "title_fr": "Concert de jazz manouche au Bataclan",
        "description_fr": "Soirée jazz manouche exceptionnelle.",
        "longdescription_fr": "Trois musiciens reconnus revisitent les standards du jazz manouche dans une ambiance feutrée. Trompette, contrebasse et guitare au programme.",
        "keywords_fr": "jazz, manouche, concert, swing",
        "conditions_fr": "Tarif unique 15 €, gratuit pour les moins de 12 ans.",
        "firstdate_begin": "2025-06-15T20:00:00+00:00",
        "lastdate_end": "2025-06-15T23:00:00+00:00",
        "location_name": "Le Bataclan",
        "location_city": "Paris",
        "location_region": "Île-de-France",
        "attendance_mode": "sur_place",
        "event_year": 2025,
    },
    {
        "uid": "ev-peinture",
        "slug": "expo-peinture-orsay",
        "canonicalurl": "https://example.org/peinture",
        "title_fr": "Exposition de peinture impressionniste",
        "description_fr": "Rétrospective des grands maîtres de l'impressionnisme français.",
        "longdescription_fr": "Plus de cinquante toiles de Monet, Renoir et Pissarro réunies au Musée d'Orsay pour célébrer le mouvement impressionniste.",
        "keywords_fr": "peinture, impressionnisme, monet, renoir",
        "conditions_fr": "Billet 14 €, gratuit le premier dimanche du mois.",
        "firstdate_begin": "2025-09-10T10:00:00+00:00",
        "lastdate_end": "2026-01-15T18:00:00+00:00",
        "location_name": "Musée d'Orsay",
        "location_city": "Paris",
        "location_region": "Île-de-France",
        "attendance_mode": "sur_place",
        "event_year": 2025,
    },
    {
        "uid": "ev-theatre",
        "slug": "moliere-comedie-francaise",
        "canonicalurl": "https://example.org/moliere",
        "title_fr": "Le Misanthrope de Molière",
        "description_fr": "Reprise du grand classique de Molière par la troupe de la Comédie-Française.",
        "longdescription_fr": "Alceste, le misanthrope, dénonce les vices de la société dans cette comédie de mœurs en vers. Mise en scène contemporaine.",
        "keywords_fr": "théâtre, molière, classique, comédie",
        "conditions_fr": "Places à partir de 5 € pour les moins de 28 ans.",
        "firstdate_begin": "2025-10-05T20:30:00+00:00",
        "lastdate_end": "2025-12-20T22:30:00+00:00",
        "location_name": "Comédie-Française",
        "location_city": "Paris",
        "location_region": "Île-de-France",
        "attendance_mode": "sur_place",
        "event_year": 2025,
    },
    {
        "uid": "ev-rando",
        "slug": "rando-vercors",
        "canonicalurl": "https://example.org/rando",
        "title_fr": "Randonnée pédestre dans le Vercors",
        "description_fr": "Découverte du plateau du Vercors en groupe accompagné.",
        "longdescription_fr": "Randonnée de 12 km à travers les alpages et les forêts du Parc Naturel Régional du Vercors, encadrée par un guide de montagne.",
        "keywords_fr": "randonnée, vercors, montagne, nature",
        "conditions_fr": "Bonnes chaussures obligatoires.",
        "firstdate_begin": "2025-07-12T08:00:00+00:00",
        "lastdate_end": "2025-07-12T17:00:00+00:00",
        "location_name": "Refuge du Vercors",
        "location_city": "Villard-de-Lans",
        "location_region": "Auvergne-Rhône-Alpes",
        "attendance_mode": "sur_place",
        "event_year": 2025,
    },
    {
        "uid": "ev-cuisine",
        "slug": "atelier-cuisine-vegetarienne",
        "canonicalurl": "https://example.org/cuisine",
        "title_fr": "Atelier de cuisine végétarienne",
        "description_fr": "Apprenez à cuisiner des plats végétariens savoureux et de saison.",
        "longdescription_fr": "Atelier pratique de trois heures animé par une cheffe spécialisée. Chaque participant repart avec ses préparations et les recettes détaillées.",
        "keywords_fr": "cuisine, végétarien, atelier, recettes",
        "conditions_fr": "Sur inscription, 45 € par personne, matériel fourni.",
        "firstdate_begin": "2025-11-08T14:00:00+00:00",
        "lastdate_end": "2025-11-08T17:00:00+00:00",
        "location_name": "Maison des associations",
        "location_city": "Lyon",
        "location_region": "Auvergne-Rhône-Alpes",
        "attendance_mode": "sur_place",
        "event_year": 2025,
    },
]


@pytest.fixture(scope="session")
def fixture_events() -> list[dict]:
    """5 events de test, distincts thématiquement et géographiquement."""
    return _FIXTURE_EVENTS


@pytest.fixture(scope="session")
def built_index(tmp_path_factory, fixture_events):
    """Construit l'index FAISS une seule fois par session pytest.

    Partagé entre `test_indexing.py` (qui valide le build) et
    `test_rag.py` (qui valide la chaîne RAG). Renvoie le couple
    `(index_dir, stats)` — `stats` est le dict retourné par `build()`."""
    from build_index import build

    tmp_dir = tmp_path_factory.mktemp("shared_index")
    input_path = tmp_dir / "events.jsonl"
    index_dir = tmp_dir / "index"

    with input_path.open("w", encoding="utf-8") as fh:
        for event in fixture_events:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    stats = build(
        input_path=input_path,
        index_dir=index_dir,
        batch_size=3,  # petit, pour exercer le flush_batch sur 2 batchs
        limit=None,
    )
    return index_dir, stats
