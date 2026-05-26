"""Test d'intégration du pipeline d'indexation FAISS.

Exécute la chaîne complète sur un mini-fixture de 5 events réalistes :
chunking via tokenizer MiniLM → embeddings réels → FAISS.from_documents →
save_local → re-load → similarity_search.

C'est un test d'**intégration** (pas unitaire) — il charge le vrai modèle
MiniLM (~118 MB) et fait tourner FAISS pour de vrai. Sans mock, parce que
mocker un modèle d'embedding fragile et l'index FAISS demanderait plus de
code que la fonction testée, et ne validerait que les mocks eux-mêmes.

Lent à cause du chargement du modèle (~5-10 s). La fixture `built_index`
de `conftest.py` est partagée à scope session avec `test_rag.py`, donc
le build n'est payé qu'une fois pour les deux fichiers.

Marqué `@pytest.mark.slow` pour permettre `pytest -m "not slow"` qui
skip ces tests dans une boucle de dev rapide.
"""

from __future__ import annotations

import pickle

import pytest

pytestmark = pytest.mark.slow


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


def test_parent_store_contains_all_events(built_index, fixture_events):
    index_dir, _ = built_index
    with (index_dir / "parent_store.pkl").open("rb") as fh:
        parent_store = pickle.load(fh)

    expected_uids = {ev["uid"] for ev in fixture_events}
    assert set(parent_store.keys()) == expected_uids
    # Chaque parent doit porter son page_content complet (titre présent)
    for event in fixture_events:
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
