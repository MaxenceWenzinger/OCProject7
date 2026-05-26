"""Test d'intégration du pipeline d'indexation FAISS.

Exécute la chaîne complète sur un mini-fixture de 5 events réalistes :
chunking via tokenizer MiniLM → embeddings réels → FAISS.from_documents →
save_local → re-load → similarity_search.

C'est un test d'**intégration** (pas unitaire) — il charge le vrai modèle
MiniLM (~118 MB) et fait tourner FAISS pour de vrai. Sans mock, parce que
mocker un modèle d'embedding fragile et l'index FAISS demanderait plus de
code que la fonction testée, et ne validerait que les mocks eux-mêmes.

Lent à cause du chargement du modèle (~5-10 s). Le modèle est chargé
une seule fois par session pytest via une fixture `scope="module"`,
donc les tests successifs ne paient pas ce coût.

Marqué `@pytest.mark.slow` pour permettre `pytest -m "not slow"` qui
skip ces tests dans une boucle de dev rapide.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_index import build  # noqa: E402


pytestmark = pytest.mark.slow


# Mini-fixture de 5 events réalistes avec thématiques distinctes pour
# pouvoir valider que le retrieval discrimine bien (chaque event a un
# mot-clé unique qu'aucun autre event ne contient).
FIXTURE_EVENTS: list[dict] = [
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


@pytest.fixture(scope="module")
def built_index(tmp_path_factory):
    """Construit l'index une seule fois pour tous les tests du module.

    Écrit le mini-fixture dans un JSONL temporaire, lance `build()` du
    script d'indexation, retourne le répertoire de l'index produit."""
    tmp_dir = tmp_path_factory.mktemp("test_indexing")
    input_path = tmp_dir / "events.jsonl"
    index_dir = tmp_dir / "index"

    with input_path.open("w", encoding="utf-8") as fh:
        for event in FIXTURE_EVENTS:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    stats = build(
        input_path=input_path,
        index_dir=index_dir,
        batch_size=3,  # petit pour exercer le code de flush_batch sur 2 batchs
        limit=None,
    )
    return index_dir, stats


def test_build_returns_expected_stats(built_index):
    _, stats = built_index
    assert stats["n_events"] == 5
    assert stats["n_chunks"] >= 5  # au moins 1 chunk par event
    assert stats["n_parents"] == 5
    assert stats["duration_s"] > 0


def test_index_files_are_created(built_index):
    index_dir, _ = built_index
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "index.pkl").exists()
    assert (index_dir / "parent_store.pkl").exists()


def test_index_can_be_reloaded(built_index):
    """Vérifie que l'index peut être rechargé et que le nombre de vecteurs
    correspond au nombre de chunks rapporté par build()."""
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    index_dir, stats = built_index
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )
    db = FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )
    assert db.index.ntotal == stats["n_chunks"]
    assert db.index.d == 384  # dim MiniLM


def test_parent_store_contains_all_events(built_index):
    index_dir, _ = built_index
    with (index_dir / "parent_store.pkl").open("rb") as fh:
        parent_store = pickle.load(fh)

    expected_uids = {ev["uid"] for ev in FIXTURE_EVENTS}
    assert set(parent_store.keys()) == expected_uids
    # Chaque parent doit porter son page_content complet (titre présent)
    for event in FIXTURE_EVENTS:
        parent = parent_store[event["uid"]]
        assert event["title_fr"] in parent.page_content


@pytest.mark.parametrize(
    "query, expected_uid",
    [
        ("concert de jazz manouche", "ev-jazz"),
        ("exposition peinture impressionniste Monet", "ev-peinture"),
        ("théâtre Molière Comédie-Française", "ev-theatre"),
        ("randonnée montagne Vercors", "ev-rando"),
        ("atelier cuisine végétarienne", "ev-cuisine"),
    ],
)
def test_keyword_query_returns_expected_event_in_top1(built_index, query, expected_uid):
    """Sanity check sémantique : sur des requêtes contenant des mots-clés
    spécifiques à chaque event, le top-1 (après dédup parent) doit être
    le bon event."""
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    index_dir, _ = built_index
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )
    db = FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )

    chunks = db.similarity_search(query, k=5)
    # Dédup par parent_uid en préservant l'ordre
    seen_parents: list[str] = []
    for c in chunks:
        pid = c.metadata["parent_uid"]
        if pid not in seen_parents:
            seen_parents.append(pid)

    assert seen_parents, f"Aucun résultat pour la requête {query!r}"
    assert seen_parents[0] == expected_uid, (
        f"Pour la requête {query!r}, top-1 attendu={expected_uid!r}, "
        f"obtenu={seen_parents[0]!r}, ordre complet={seen_parents}"
    )
