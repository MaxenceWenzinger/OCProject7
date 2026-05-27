"""Reconstruction de l'index FAISS en arrière-plan.

`POST /rebuild` (cf. `api/main.py`) déclenche `run_rebuild` via le
mécanisme `BackgroundTasks` de FastAPI : la requête HTTP renvoie `202`
immédiatement et le job tourne dans le même process. À la fin, on
recharge un nouveau `RAGService` dans `app.state` pour que les requêtes
`/ask` suivantes voient le nouvel index sans redémarrer.

Trois pièces ici :

- `RebuildState` : container d'état (in_progress, started_at, ...)
  partagé via `app.state.rebuild_state`. Mute en place.
- `verify_admin_token` : dépendance FastAPI qui valide le header
  Authorization Bearer en temps constant contre `ADMIN_TOKEN`.
- `run_rebuild` : la fonction job, synchrone, qui enchaîne
  fetch → clean → build → hot-swap du RAGService. C'est ce que
  `BackgroundTasks.add_task` exécute.

Le job est *synchrone* et long (~2 h en réel). FastAPI le déporte sur
un thread du pool via `BackgroundTasks` — pas besoin d'`async`.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
STATE_FILENAME = "rebuild_state.json"


@dataclass
class RebuildState:
    """État partagé du dernier (ou courant) job de rebuild.

    `in_progress` est levé avant le démarrage du job et baissé en
    `finally`, succès comme échec — c'est ce flag que `/ask` consulte
    pour répondre 503 pendant un rebuild.

    Les trois autres champs sont persistés sur disque (`save`/`load`)
    pour survivre aux redémarrages de l'API."""

    in_progress: bool = False
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    last_error: Optional[str] = None

    def save(self, index_dir: Path = INDEX_DIR) -> None:
        """Persiste l'état (sauf `in_progress`) dans `index_dir/rebuild_state.json`.

        On exclut `in_progress` car il n'a de sens qu'au runtime — un
        process tué pendant un rebuild ne doit pas redémarrer avec
        `in_progress=True` (sinon `/ask` répondrait 503 indéfiniment)."""
        index_dir.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in asdict(self).items() if k != "in_progress"}
        path = index_dir / STATE_FILENAME
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "RebuildState":
        """Recharge l'état depuis le disque, ou renvoie un état vide si absent.

        En cas de fichier corrompu, log et renvoie un état vide plutôt
        que de planter le démarrage de l'API."""
        path = index_dir / STATE_FILENAME
        if not path.exists():
            return cls()
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Lecture de %s échouée (%s), reset de l'état.", path, exc)
            return cls()
        return cls(
            in_progress=False,
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            last_error=payload.get("last_error"),
        )


# Authentification Bearer pour `/rebuild`.
# `auto_error=False` : on veut renvoyer 401 même quand le header est
# absent (sinon FastAPI renvoie 403). On gère la validation à la main
# dans `verify_admin_token`.
_bearer_scheme = HTTPBearer(auto_error=False, description="Admin token (Bearer)")


def verify_admin_token(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> None:
    """Dépendance FastAPI : valide le header Authorization Bearer.

    - Si `ADMIN_TOKEN` n'est pas configuré → 503 (fail-secure).
    - Si le header est absent ou mal formé → 401.
    - Si le token ne matche pas → 401.
    - Sinon, retourne `None` (la dépendance ne renvoie rien d'utile)."""
    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        log.warning(
            "ADMIN_TOKEN non configuré : /rebuild est désactivé (fail-secure)."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ADMIN_TOKEN non configuré côté serveur.",
        )

    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token admin requis (Authorization: Bearer <token>).",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(creds.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token admin invalide.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_rebuild(request: Request) -> None:
    """Job exécuté en arrière-plan : fetch → clean → build → hot-swap.

    On accepte `request` (et donc `request.app`) pour pouvoir muter
    `app.state.rebuild_state`, `app.state.rebuild_in_progress` et
    `app.state.rag_service` sans dépendance globale."""
    app = request.app
    state: RebuildState = app.state.rebuild_state

    # Imports paresseux : les scripts métier chargent eux-mêmes
    # sentence-transformers, faiss, etc. — pas la peine de payer ça
    # au démarrage de l'API si /rebuild n'est jamais appelé.
    from scripts.build_index import build as build_index
    from scripts.clean_events import clean_stream
    from scripts.fetch_openagenda import build_session, fetch

    state.in_progress = True
    state.started_at = utc_now_iso()
    state.finished_at = None
    state.last_error = None
    app.state.rebuild_in_progress = True

    today = date.today().isoformat()
    raw_path = RAW_DIR / f"events_{today}.jsonl"
    clean_path = PROCESSED_DIR / f"events_clean_{today}.jsonl"

    try:
        log.info("Rebuild | fetch → %s", raw_path)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        n_lines, n_bytes = fetch(raw_path, build_session())
        log.info("Rebuild | fetch terminé : %d lignes, %.1f MB",
                 n_lines, n_bytes / (1024 * 1024))

        log.info("Rebuild | clean → %s", clean_path)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        clean_stats = clean_stream(raw_path, clean_path)
        log.info("Rebuild | clean terminé : %d gardés / %d lus",
                 clean_stats["kept"], clean_stats["read"])

        log.info("Rebuild | build FAISS → %s", INDEX_DIR)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        build_stats = build_index(clean_path, INDEX_DIR, batch_size=5000, limit=None)
        log.info("Rebuild | build terminé : %d chunks / %d parents",
                 build_stats.get("n_chunks", 0), build_stats.get("n_parents", 0))

        log.info("Rebuild | hot-swap du RAGService")
        from src.rag.service import RAGService
        try:
            new_service = RAGService()
            app.state.rag_service = new_service
            log.info("Rebuild | RAGService rechargé.")
        except Exception as exc:
            log.exception("Rebuild | échec du hot-swap : on garde l'ancien service")
            state.last_error = f"hot-swap échoué : {exc}"

    except Exception as exc:
        log.exception("Rebuild | échec")
        state.last_error = str(exc)
    finally:
        state.in_progress = False
        state.finished_at = utc_now_iso()
        app.state.rebuild_in_progress = False
        try:
            state.save()
        except OSError as exc:
            log.warning("Persistance de rebuild_state.json échouée : %s", exc)
        log.info("Rebuild | terminé (last_error=%s)", state.last_error)
