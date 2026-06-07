"""Tests d'intégration bout-en-bout de la chaîne RAG.

Exerce `RAGService.answer()` sur le mini-index partagé (`built_index`)
avec le vrai LLM configuré par `LLM_PROVIDER` (Mistral via API cloud par
défaut, Ollama en fallback offline). Si le LLM n'est pas disponible au
démarrage de la session, tous les tests E2E sont skippés proprement —
la fixture `llm_available` vérifie la disponibilité selon le provider
et appelle `pytest.skip(...)` sinon :

- provider `mistral` : skip si `MISTRAL_API_KEY` est absent (cas CI sans
  secret) ;
- provider `ollama`  : skip si le serveur local ne répond pas au ping.

Permet à la CI sans clé Mistral ni Ollama de ne pas planter.

Les assertions sont volontairement légères (mots-clés présents dans la
réponse, présence dans les sources, structure du dict retourné). La
validation rigoureuse de la qualité des réponses est le rôle de Ragas
dans l'Epic 6, pas de ces tests d'intégration.

Marqué `slow` : un appel `answer()` prend ~3-30s (extraction LLM +
retrieval + génération LLM, selon le provider). Skip via
`pytest -m "not slow"`.
"""

from __future__ import annotations

import os

import pytest
import requests

from src.rag.llm import DEFAULT_OLLAMA_HOST, DEFAULT_PROVIDER

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def llm_available() -> bool:
    """Vérifie au démarrage de session que le LLM du provider actif est joignable.

    Selon `LLM_PROVIDER` (défaut `mistral`) :
    - `mistral` : exige `MISTRAL_API_KEY` (pas d'appel réseau ici, on ne
      veut pas consommer de quota juste pour un health-check) ;
    - `ollama`  : ping `OLLAMA_HOST/api/tags` avec un timeout de 2s.

    Skip tous les tests qui consomment cette fixture si indisponible — ils
    s'affichent comme `SKIPPED` au lieu de `FAILED` (CI sans secret/serveur)."""
    provider = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()

    if provider == "mistral":
        if not os.getenv("MISTRAL_API_KEY"):
            pytest.skip("MISTRAL_API_KEY absent, tests RAG (provider mistral) skippés")
        return True

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        try:
            r = requests.get(f"{base_url}/api/tags", timeout=2)
            if r.status_code != 200:
                pytest.skip(f"Ollama répond {r.status_code}, tests RAG skippés")
            return True
        except (requests.ConnectionError, requests.Timeout) as exc:
            pytest.skip(f"Ollama injoignable ({exc}), tests RAG skippés")

    pytest.skip(f"LLM_PROVIDER={provider!r} non géré par les tests RAG")


@pytest.fixture(scope="module")
def rag_service(built_index, llm_available):
    """Instancie RAGService une seule fois pour le module.

    Coût ~15s (chargement embeddings + index + parent_store + LUT +
    init du client LLM). Réutilisé par tous les tests du module."""
    from src.rag.service import RAGService

    index_dir, _ = built_index
    return RAGService(index_dir=index_dir)


# ---------------------------------------------------------------------------
# Tests structurels rapides (n'appellent pas Ollama au moment du test)
# ---------------------------------------------------------------------------


def test_answer_empty_question_raises(rag_service):
    """Contrat : answer('') doit lever ValueError, pas appeler le LLM."""
    with pytest.raises(ValueError):
        rag_service.answer("")
    with pytest.raises(ValueError):
        rag_service.answer("   ")


# ---------------------------------------------------------------------------
# Tests E2E avec vrai LLM
# ---------------------------------------------------------------------------


def test_answer_returns_expected_shape(rag_service):
    """answer() renvoie un dict avec les clés contractuelles."""
    result = rag_service.answer("concert de jazz")
    assert set(result.keys()) >= {
        "answer", "sources", "filters_used", "filter_relaxed", "timings",
    }
    assert isinstance(result["answer"], str) and result["answer"].strip()
    assert isinstance(result["sources"], list)
    assert isinstance(result["filters_used"], dict)
    assert isinstance(result["filter_relaxed"], bool)
    assert set(result["timings"].keys()) >= {
        "extract_ms", "retrieve_ms", "generate_ms", "total_ms",
    }


def test_jazz_question_finds_jazz_event(rag_service):
    """Question thématique simple : 'jazz' doit ramener ev-jazz en source
    et le mot 'jazz' doit apparaître dans la réponse générée."""
    result = rag_service.answer("Quels concerts de jazz puis-je voir ?")

    source_uids = [s.metadata.get("uid") for s in result["sources"]]
    assert "ev-jazz" in source_uids, (
        f"ev-jazz attendu dans les sources, obtenu : {source_uids}"
    )

    # Le LLM doit citer le sujet de la question dans sa réponse
    assert "jazz" in result["answer"].lower(), (
        f"Mot 'jazz' attendu dans la réponse, obtenu : {result['answer']!r}"
    )


def test_city_filter_extracted_and_used(rag_service):
    """Question avec contrainte géographique : l'extraction self-query doit
    repérer 'Lyon', le retrieval doit privilégier les events à Lyon, et
    le LLM doit mentionner l'atelier cuisine (seul event lyonnais)."""
    result = rag_service.answer("Y a-t-il une activité à Lyon ?")

    # L'extracteur a bien repéré la ville
    assert result["filters_used"].get("city", "").lower() == "lyon", (
        f"Filtre city=Lyon attendu, obtenu : {result['filters_used']}"
    )

    # Le seul event lyonnais (atelier cuisine) doit remonter dans les sources
    source_uids = [s.metadata.get("uid") for s in result["sources"]]
    assert "ev-cuisine" in source_uids, (
        f"ev-cuisine (Lyon) attendu dans les sources, obtenu : {source_uids}"
    )


def test_out_of_domain_question_returns_fallback(rag_service):
    """Question hors-domaine : le prompt système impose le fallback exact
    'Je ne peux répondre qu'à des questions sur les événements culturels
    du catalogue.'"""
    result = rag_service.answer("Quelle est la capitale du Pérou ?")

    # Match souple : on accepte des variations mineures (ponctuation,
    # casse), mais la phrase-clé doit être présente.
    answer_lower = result["answer"].lower()
    assert "événements culturels" in answer_lower or "evenements culturels" in answer_lower, (
        f"Fallback hors-domaine attendu, obtenu : {result['answer']!r}"
    )
