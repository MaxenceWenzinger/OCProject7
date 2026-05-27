"""Service RAG : compose extraction self-query, retrieval pre-filtré et génération.

`RAGService` est le point d'entrée unique de la chaîne RAG côté API. Il
charge l'index FAISS, le parent_store, le LUT inverse et instancie les
deux instances LLM (extractor + générateur) une seule fois au démarrage,
puis sert N requêtes via `answer(question)`.

Architecture du flow `answer(question)` :

    question (str)
        │
        ▼
    extractor LLM ──────────────────────► QueryFilters
        │
        ▼
    retrieve_parents ───────────────────► list[Document parents]
    (pre-filter exact + post-filter
     date, fail-open si 0 résultat)
        │
        ▼
    chain LCEL (prompt | llm | parser) ─► réponse (str)
        │
        ▼
    {answer, sources, filters_used, filter_relaxed}

Le retour est délibérément riche : `filters_used` et `filter_relaxed`
permettent à l'API et aux tests/évaluations d'inspecter la trajectoire
d'une requête sans devoir relancer.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from langchain_core.documents import Document

from src.rag.chain import build_chain
from src.rag.llm import get_llm
from src.rag.query_parser import QueryFilters, build_extractor
from src.rag.retrieval import (
    DEFAULT_INDEX_DIR,
    load_embeddings,
    load_parent_store,
    load_uid_to_faiss_ids,
    load_vector_store,
    retrieve_parents,
)

log = logging.getLogger(__name__)


class RAGService:
    """Orchestrateur de la chaîne RAG complète, prêt à servir des questions.

    L'init est coûteux (~20-30s : modèle d'embedding + index FAISS +
    parent_store + LUT + connexion Ollama) et doit être fait une seule
    fois au démarrage de l'API. Toutes les ressources lourdes sont
    réutilisées d'une question à l'autre."""

    def __init__(
        self,
        index_dir: Path = DEFAULT_INDEX_DIR,
        k_parents: int = 10,
        k_chunks: int = 30,
        fetch_k: int = 200,
    ) -> None:
        t0 = time.perf_counter()
        log.info("Initialisation RAGService...")

        self._embeddings = load_embeddings()
        self._vector_store = load_vector_store(index_dir, embeddings=self._embeddings)
        self._parent_store = load_parent_store(index_dir)
        self._uid_to_faiss_ids = load_uid_to_faiss_ids(self._vector_store, index_dir)

        # Deux LLM clients : l'extractor en structured_output, le générateur
        # en texte libre. On peut partager la même instance ChatOllama vu
        # que `with_structured_output` retourne un wrapper indépendant.
        llm = get_llm()
        self._extractor = build_extractor(llm)
        self._chain = build_chain(llm)

        self.k_parents = k_parents
        self.k_chunks = k_chunks
        self.fetch_k = fetch_k

        log.info("RAGService prêt en %.1fs (parents=%d, chunks=%d)",
                 time.perf_counter() - t0,
                 len(self._parent_store),
                 self._vector_store.index.ntotal)

    def answer(self, question: str) -> dict:
        """Pipeline complet pour une question : extraction → retrieval → génération.

        Renvoie un dict :
            answer (str)            : la réponse générée par le LLM
            sources (list[Document]): les events parents passés au LLM
            filters_used (dict)     : filtres extraits effectivement non-nuls
            filter_relaxed (bool)   : True si on a fail-open (filtre relâché)
            timings (dict)          : temps en ms par étape (debug)
        """
        if not question or not question.strip():
            raise ValueError("question vide")

        timings: dict[str, float] = {}

        # 1. Extraction self-query
        t0 = time.perf_counter()
        try:
            filters = self._extractor.invoke({"question": question})
        except Exception as exc:
            # L'extraction peut planter (JSON invalide, timeout Ollama...).
            # On dégrade gracieusement en « pas de filtre » plutôt que de
            # remonter l'erreur — le RAG reste fonctionnel.
            log.warning("Extraction self-query échouée (%s), continue sans filtre", exc)
            filters = QueryFilters()
        timings["extract_ms"] = (time.perf_counter() - t0) * 1000

        # 2. Retrieval pre-filtré + dédup parent
        t0 = time.perf_counter()
        sources, relaxed = retrieve_parents(
            query=question,
            vector_store=self._vector_store,
            parent_store=self._parent_store,
            uid_to_faiss_ids=self._uid_to_faiss_ids,
            embeddings=self._embeddings,
            filters=filters,
            k_parents=self.k_parents,
            k_chunks=self.k_chunks,
            fetch_k=self.fetch_k,
        )
        timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000

        # 3. Génération
        t0 = time.perf_counter()
        answer = self._chain.invoke({"context": sources, "question": question})
        timings["generate_ms"] = (time.perf_counter() - t0) * 1000

        timings["total_ms"] = sum(timings.values())

        log.info(
            "RAG | filters=%s relaxed=%s n_sources=%d | "
            "extract=%.0fms retrieve=%.0fms generate=%.0fms total=%.0fms",
            filters.as_dict() or "{}", relaxed, len(sources),
            timings["extract_ms"], timings["retrieve_ms"],
            timings["generate_ms"], timings["total_ms"],
        )

        return {
            "answer": answer,
            "sources": sources,
            "filters_used": filters.as_dict(),
            "filter_relaxed": relaxed,
            "timings": timings,
        }
