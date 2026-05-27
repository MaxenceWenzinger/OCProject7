"""Tests fonctionnels de l'API FastAPI.

On exerce les endpoints `/ask`, `/rebuild` et `/rebuild/status` avec :

- Un `RAGService` **mocké** : on ne charge ni l'index FAISS réel
  (~15 s + 1,5 GB sur disque, déjà couvert par `test_indexing.py`)
  ni Ollama (déjà couvert par `test_rag.py`). Le couplage API ↔ RAG
  est validé par l'assertion : ce que `answer()` renvoie doit
  apparaître dans la réponse HTTP.
- Un job de rebuild **mocké** : le vrai rebuild prend ~2 h, on ne
  veut surtout pas qu'il s'exécute en CI. On vérifie que le job a
  été enregistré en `BackgroundTasks` et que les flags d'état sont
  correctement gérés.
- L'`ADMIN_TOKEN` injecté via `monkeypatch.setenv` au lieu d'un
  vrai `.env`. La dépendance `verify_admin_token` lit cette
  variable à chaque requête → ça suffit pour tester l'auth.

Pas de marquage `slow` : ces tests doivent tourner en CI sans Ollama
ni modèle ML chargé, en moins d'une seconde.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api.main import app
from api.rebuild import RebuildState


ADMIN_TOKEN = "test-admin-token-fixture"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service() -> MagicMock:
    """RAGService mocké : `answer()` renvoie un dict aligné sur le contrat réel."""
    service = MagicMock()
    service.answer.return_value = {
        "answer": "Voici un concert de jazz au Bataclan.",
        "sources": [
            Document(
                page_content="...",
                metadata={
                    "uid": "ev-jazz",
                    "title": "Concert de jazz manouche",
                    "url": "https://example.org/jazz",
                    "location_city": "Paris",
                    "first_date": "2025-06-15T20:00:00+00:00",
                    "last_date": "2025-06-15T23:00:00+00:00",
                },
            ),
        ],
        "filters_used": {"city": "Paris"},
        "filter_relaxed": False,
        "timings": {
            "extract_ms": 1.0,
            "retrieve_ms": 2.0,
            "generate_ms": 3.0,
            "total_ms": 6.0,
        },
    }
    return service


@pytest.fixture
def client(mock_service, monkeypatch) -> TestClient:
    """TestClient avec RAGService mocké et ADMIN_TOKEN injecté.

    On bypass le `lifespan` (qui chargerait le vrai index) en injectant
    directement les attributs dans `app.state` avant d'instancier le
    TestClient sans `with`."""
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    app.state.rag_service = mock_service
    app.state.rebuild_state = RebuildState()
    app.state.rebuild_in_progress = False
    return TestClient(app)


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------


def test_ask_returns_200_with_sources(client, mock_service):
    """Happy path : POST /ask renvoie 200 avec answer + sources + filters_used."""
    response = client.post("/ask", json={"question": "Concert jazz Paris"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Voici un concert de jazz au Bataclan."
    assert body["filters_used"] == {"city": "Paris"}
    assert body["filter_relaxed"] is False
    assert len(body["sources"]) == 1
    source = body["sources"][0]
    assert source["uid"] == "ev-jazz"
    assert source["location_city"] == "Paris"
    # Le service a bien été appelé avec la question
    mock_service.answer.assert_called_once_with("Concert jazz Paris")


def test_ask_empty_question_returns_422(client, mock_service):
    """Validation Pydantic : question vide → 422 sans appeler le service."""
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422
    mock_service.answer.assert_not_called()


def test_ask_missing_question_returns_422(client, mock_service):
    """Validation Pydantic : payload sans `question` → 422."""
    response = client.post("/ask", json={})

    assert response.status_code == 422
    mock_service.answer.assert_not_called()


def test_ask_during_rebuild_returns_503(client, mock_service):
    """Quand un rebuild est en cours, /ask refuse de servir."""
    app.state.rebuild_in_progress = True

    response = client.post("/ask", json={"question": "Concert jazz"})

    assert response.status_code == 503
    assert "rebuild" in response.json()["detail"].lower() or \
           "reconstruction" in response.json()["detail"].lower()
    mock_service.answer.assert_not_called()


# ---------------------------------------------------------------------------
# /rebuild — authentification
# ---------------------------------------------------------------------------


def test_rebuild_without_header_returns_401(client):
    """Pas de header Authorization → 401."""
    response = client.post("/rebuild")

    assert response.status_code == 401
    assert response.headers.get("www-authenticate", "").lower().startswith("bearer")


def test_rebuild_wrong_token_returns_401(client):
    """Token Bearer présent mais invalide → 401."""
    response = client.post(
        "/rebuild", headers={"Authorization": "Bearer wrong-token"}
    )

    assert response.status_code == 401


def test_rebuild_without_admin_token_env_returns_503(client, monkeypatch):
    """ADMIN_TOKEN non configuré côté serveur → 503 (fail-secure)."""
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)

    response = client.post(
        "/rebuild", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}
    )

    assert response.status_code == 503


# ---------------------------------------------------------------------------
# /rebuild — déclenchement et exécution en arrière-plan
# ---------------------------------------------------------------------------


def test_rebuild_with_valid_token_returns_202(client, monkeypatch):
    """Token valide → 202 Accepted + status=started.

    Le job réel est mocké pour ne pas lancer fetch+clean+build (~2 h)."""
    monkeypatch.setattr("api.main.run_rebuild", MagicMock())

    response = client.post(
        "/rebuild", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "started"
    assert "started_at" in body and body["started_at"]


def test_rebuild_runs_background_task(client, monkeypatch):
    """Le job est bien enregistré comme tâche d'arrière-plan et exécuté
    après la réponse HTTP (TestClient déclenche les BackgroundTasks).

    Pour valider sans lancer le vrai pipeline : on stub `run_rebuild`
    par un MagicMock qui mute `rebuild_state` comme le vrai job."""
    mock_job = MagicMock()

    def fake_job(request):
        # Mime ce que le vrai run_rebuild fait, en bref
        request.app.state.rebuild_state.in_progress = False
        request.app.state.rebuild_state.finished_at = "2026-05-27T01:00:00Z"
        request.app.state.rebuild_in_progress = False
        mock_job(request)

    monkeypatch.setattr("api.main.run_rebuild", fake_job)

    response = client.post(
        "/rebuild", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}
    )

    assert response.status_code == 202
    # TestClient exécute les BackgroundTasks après la réponse → mock_job appelé
    mock_job.assert_called_once()


def test_rebuild_conflict_when_already_in_progress(client, monkeypatch):
    """Si un rebuild est déjà en cours → 409 Conflict."""
    monkeypatch.setattr("api.main.run_rebuild", MagicMock())
    app.state.rebuild_state.in_progress = True

    response = client.post(
        "/rebuild", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}
    )

    assert response.status_code == 409


# ---------------------------------------------------------------------------
# /rebuild/status
# ---------------------------------------------------------------------------


def test_rebuild_status_idle(client):
    """État initial : aucun rebuild lancé → in_progress=False, dates None."""
    response = client.get("/rebuild/status")

    assert response.status_code == 200
    body = response.json()
    assert body["in_progress"] is False
    assert body["started_at"] is None
    assert body["finished_at"] is None
    assert body["last_error"] is None


def test_rebuild_status_in_progress(client):
    """État en cours : flag levé et started_at renseigné."""
    app.state.rebuild_state = RebuildState(
        in_progress=True,
        started_at="2026-05-27T00:00:00Z",
    )

    response = client.get("/rebuild/status")

    assert response.status_code == 200
    body = response.json()
    assert body["in_progress"] is True
    assert body["started_at"] == "2026-05-27T00:00:00Z"
    assert body["finished_at"] is None


def test_rebuild_status_no_auth_required(client):
    """`/rebuild/status` ne demande pas de token (pratique pour la démo)."""
    response = client.get("/rebuild/status")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Health-check
# ---------------------------------------------------------------------------


def test_health_check_ok(client):
    """GET / renvoie un statut OK et indique que le RAG est chargé."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "rag_ready": True}
