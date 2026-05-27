"""Application FastAPI exposant le service RAG Puls-Events.

Le cycle de vie initialise un `RAGService` unique au démarrage
(chargement du modèle d'embedding, de l'index FAISS, du parent_store
et du LUT inverse) et le partage via `app.state`. Les endpoints
exposés sont :

- `GET /` — health-check
- `POST /ask` — question/réponse RAG
- `POST /rebuild` — reconstruction de l'index (auth Bearer, async)
- `GET /rebuild/status` — état du dernier job de rebuild

Le `logging.basicConfig` au niveau module garantit que les lignes de log
déjà émises par `src.rag.service` apparaissent dans la sortie de uvicorn.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status

from api.rebuild import RebuildState, utc_now_iso, run_rebuild, verify_admin_token
from api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    RebuildResponse,
    RebuildStatusResponse,
    Source,
)
from src.rag.service import RAGService

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("Démarrage de l'API : initialisation du RAGService...")
    app.state.rag_service = RAGService()
    app.state.rebuild_state = RebuildState.load()
    app.state.rebuild_in_progress = False
    if not os.getenv("ADMIN_TOKEN"):
        log.warning(
            "ADMIN_TOKEN non défini : l'endpoint /rebuild répondra 503 "
            "tant que la variable n'est pas configurée (fail-secure)."
        )
    log.info("API prête.")
    try:
        yield
    finally:
        log.info("Arrêt de l'API.")
        app.state.rag_service = None


app = FastAPI(
    title="Puls-Events RAG API",
    description=(
        "API de questions/réponses sur les événements culturels du catalogue "
        "Open Agenda (France entière, événements en cours ou à venir)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health-check",
    tags=["health"],
)
def root() -> HealthResponse:
    """Renvoie un statut minimal indiquant si le service RAG est chargé."""
    ready = getattr(app.state, "rag_service", None) is not None
    return HealthResponse(status="ok", rag_ready=ready)


def _extract_short_description(page_content: str | None, title: str | None) -> str | None:
    """Extrait la description courte d'un Document parent depuis son `page_content`.

    Le format produit par `build_page_content` est :
        {title}\\n\\n{description_fr}\\n\\n{longdescription_fr}\\n\\n
        Mots-clés : ...\\n\\nConditions : ...

    On découpe sur `\\n\\n` et on prend le premier bloc qui n'est ni le
    titre, ni préfixé par `Mots-clés :` / `Conditions :`. Plafonné à
    300 caractères pour rester un *aperçu* (description_fr est déjà
    plafonné à 200 chars par Open Agenda, mais quelques events sans
    description courte ont leur longdescription en bloc 2)."""
    if not page_content:
        return None
    blocks = [b.strip() for b in page_content.split("\n\n") if b.strip()]
    for block in blocks:
        if title and block == title:
            continue
        if block.startswith("Mots-clés :") or block.startswith("Conditions :"):
            continue
        return block[:300] + ("…" if len(block) > 300 else "")
    return None


def _source_from_document(doc) -> Source:
    """Projette un Document parent (LangChain) vers le schéma Source de l'API."""
    md = doc.metadata or {}
    title = md.get("title")
    return Source(
        uid=md.get("uid") or "",
        title=title,
        description=_extract_short_description(doc.page_content, title),
        url=md.get("url"),
        location_city=md.get("location_city"),
        first_date=md.get("first_date"),
        last_date=md.get("last_date"),
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Poser une question sur le catalogue",
    tags=["rag"],
    responses={
        503: {"description": "Service indisponible (rebuild de l'index en cours)."},
    },
)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Pose une question au RAG et renvoie la réponse + sources + filtres extraits."""
    if getattr(request.app.state, "rebuild_in_progress", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reconstruction de l'index en cours, réessayez dans quelques minutes.",
        )

    service: RAGService | None = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service RAG non initialisé.",
        )

    result = service.answer(payload.question)
    return AskResponse(
        answer=result["answer"],
        sources=[_source_from_document(d) for d in result["sources"]],
        filters_used=result["filters_used"],
        filter_relaxed=result["filter_relaxed"],
    )


@app.post(
    "/rebuild",
    response_model=RebuildResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Reconstruire l'index FAISS (fetch + clean + build)",
    tags=["admin"],
    responses={
        202: {"description": "Job accepté et lancé en arrière-plan."},
        401: {"description": "Token admin manquant ou invalide."},
        409: {"description": "Un rebuild est déjà en cours."},
        503: {"description": "ADMIN_TOKEN non configuré côté serveur."},
    },
    dependencies=[Depends(verify_admin_token)],
)
def rebuild(request: Request, background_tasks: BackgroundTasks) -> RebuildResponse:
    """Déclenche un rebuild complet de l'index en arrière-plan.

    Pipeline : téléchargement Open Agenda → cleaning → indexation FAISS,
    puis hot-swap du `RAGService` une fois l'index reconstruit. Pendant
    l'exécution, `POST /ask` répond `503`. La progression est consultable
    via `GET /rebuild/status`."""
    state: RebuildState = request.app.state.rebuild_state
    if state.in_progress:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Un rebuild est déjà en cours.",
        )

    background_tasks.add_task(run_rebuild, request)
    return RebuildResponse(status="started", started_at=utc_now_iso())


@app.get(
    "/rebuild/status",
    response_model=RebuildStatusResponse,
    summary="État du dernier job de rebuild",
    tags=["admin"],
)
def rebuild_status(request: Request) -> RebuildStatusResponse:
    """Renvoie l'état courant du flag `rebuild_in_progress` + horodatages
    et éventuelle dernière erreur. Non protégé : pratique pour la démo."""
    state: RebuildState = request.app.state.rebuild_state
    return RebuildStatusResponse(
        in_progress=state.in_progress,
        started_at=state.started_at,
        finished_at=state.finished_at,
        last_error=state.last_error,
    )
