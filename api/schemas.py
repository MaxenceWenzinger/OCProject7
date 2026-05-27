"""Schémas Pydantic exposés par l'API.

On garde une séparation stricte entre :
- les schémas « entrée » (`AskRequest`) — validés par FastAPI avant de
  toucher le service ;
- les schémas « sortie » (`AskResponse`, `Source`, `RebuildResponse`,
  `RebuildStatusResponse`) — construits depuis le dict renvoyé par
  `RAGService.answer()` ou depuis l'état du job de rebuild.

`filters_used` est exposé tel quel (dict des filtres extraits non-nuls)
plutôt que sous forme d'un schéma typé : le contenu peut évoluer côté
self-querying sans casser le contrat API, et la valeur est de toute
façon présentée à titre informatif (debug / démo).
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Statut du service exposé par `GET /`."""

    status: str = Field(..., description="'ok' si l'API répond.")
    rag_ready: bool = Field(
        ...,
        description="True si le RAGService a été initialisé (index FAISS chargé).",
    )


class AskRequest(BaseModel):
    """Question utilisateur posée à l'API."""

    question: str = Field(
        ...,
        min_length=1,
        description="Question en français sur les événements du catalogue.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"question": "Quels concerts de jazz à Paris cette semaine ?"},
                {"question": "Une exposition d'art contemporain à Lyon ?"},
            ]
        }
    )


class Source(BaseModel):
    """Référence à un événement utilisé pour rédiger la réponse.

    Reflète un sous-ensemble de la metadata d'un Document parent — les
    champs utiles à l'utilisateur (titre, description courte, lieu,
    dates, lien). Les champs nullables le sont vraiment dans le
    dataset : par exemple `location_city` est absent sur les events
    en ligne."""

    uid: str
    title: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    location_city: Optional[str] = None
    first_date: Optional[str] = None
    last_date: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "uid": "12345678",
                    "title": "Festival de Jazz de Paris",
                    "description": "Trois jours de concerts de jazz dans le parc de la Villette.",
                    "url": "https://openagenda.com/festival-jazz-paris",
                    "location_city": "Paris",
                    "first_date": "2026-06-01T18:00:00+00:00",
                    "last_date": "2026-06-03T23:00:00+00:00",
                }
            ]
        }
    )


class AskResponse(BaseModel):
    """Réponse de `/ask` : texte généré + sources structurées + traces."""

    answer: str
    sources: list[Source]
    filters_used: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Filtres extraits de la question par self-querying et "
            "effectivement appliqués. Vide si aucun filtre n'a été extrait."
        ),
    )
    filter_relaxed: bool = Field(
        default=False,
        description=(
            "True si les filtres ont été extraits mais relâchés faute de "
            "résultats — la réponse a été générée sur le retrieval non filtré."
        ),
    )


class RebuildResponse(BaseModel):
    """Accusé de réception d'une demande `/rebuild` (job lancé en arrière-plan)."""

    status: str = Field(..., description="Statut du déclenchement, ex: 'started'.")
    started_at: str = Field(
        ..., description="Horodatage ISO 8601 du démarrage du job."
    )


class RebuildStatusResponse(BaseModel):
    """État courant du dernier job `/rebuild` (succès, échec ou en cours)."""

    in_progress: bool
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    last_error: Optional[str] = None
