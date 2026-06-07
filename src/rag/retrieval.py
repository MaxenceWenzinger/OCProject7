"""Retrieval parent-child sur l'index FAISS, avec pre-filtering metadata.

Trois responsabilités :

1. **Chargement** de l'index FAISS, du `parent_store` (`uid → Document parent`)
   et d'un **LUT inverse** `uid → list[faiss_id]` mis en cache disque.
2. **Pre-filtering vrai** sur city/region : au lieu de laisser FAISS
   ramener les k plus similaires puis post-filtrer (qui peut renvoyer 0
   résultat pour une ville rare comme Reims), on calcule en amont la
   liste des `faiss_id` autorisés via le LUT, on reconstruit leurs
   vecteurs, et on calcule la distance L2 directement en numpy. Pas de
   `fetch_k` à ajuster, on est sûr de couvrir tous les events qui
   passent le filtre.
3. **Dédup parent + fail-open** : on dédoublonne par `parent_uid` après
   le ranking. Si un filtre extrait ne ramène aucun parent (filtre trop
   strict ou typo dans la question), on relance sans filtre — un filtre
   est un signal, pas une contrainte stricte.

Les filtres temporels (`date_after`/`date_before`) sont eux aussi
appliqués en pre-filter, sur les champs `first_date`/`last_date` de
la metadata parent. Un event est gardé si sa fenêtre `[first_date,
last_date]` chevauche `[date_after, date_before]`.
"""

from __future__ import annotations

import logging
import pickle
import time
import unicodedata
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.query_parser import QueryFilters

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LUT_FILENAME = "uid_to_faiss_ids.pkl"

# Variantes anglaises observées dans le dataset → forme française canonique.
# Le filtre normalise les deux côtés, donc seule la forme canonique compte ici.
REGION_ALIASES: dict[str, str] = {
    "brittany": "bretagne",
    "normandy": "normandie",
    "occitania": "occitanie",
}


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------


def load_embeddings() -> HuggingFaceEmbeddings:
    """Charge le modèle d'embedding (~5s, à appeler une seule fois)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 32},
    )


def load_vector_store(
    index_dir: Path = DEFAULT_INDEX_DIR,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    """Charge l'index FAISS + sa mapping vector_id→chunk depuis `data/index/`.

    `allow_dangerous_deserialization=True` est requis : LangChain
    désérialise un pickle pour reconstruire le `docstore`. Acceptable
    car le fichier est produit par notre propre `build_index.py`."""
    embeddings = embeddings or load_embeddings()
    return FAISS.load_local(
        str(index_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def load_parent_store(index_dir: Path = DEFAULT_INDEX_DIR) -> dict[str, Document]:
    """Charge le mapping `uid → Document parent` (~289 MB, ~2-3s)."""
    with (index_dir / "parent_store.pkl").open("rb") as fh:
        return pickle.load(fh)


def load_uid_to_faiss_ids(
    vector_store: FAISS,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> dict[str, list[int]]:
    """LUT inverse `parent_uid → list[faiss_id]`, avec cache disque.

    Au premier appel ou si le cache est plus ancien que `index.faiss`,
    on (re)construit en parcourant `index_to_docstore_id`. Le cache est
    invalidé automatiquement par mtime : un rebuild via `build_index.py`
    réécrit `index.faiss` avec un mtime plus récent → on reconstruit.

    Coût : ~5-10s de build (one-shot), ~1-2s de chargement ensuite.
    Disque : ~30 MB (252 901 events × en moyenne 2-3 ids)."""
    cache_path = index_dir / LUT_FILENAME
    index_path = index_dir / "index.faiss"

    if (
        cache_path.exists()
        and index_path.exists()
        and cache_path.stat().st_mtime >= index_path.stat().st_mtime
    ):
        t0 = time.perf_counter()
        with cache_path.open("rb") as fh:
            mapping = pickle.load(fh)
        log.info("LUT uid→faiss_ids chargé depuis cache en %.1fs (%d uids)",
                 time.perf_counter() - t0, len(mapping))
        return mapping

    log.info("LUT uid→faiss_ids absent ou périmé, construction (~5-10s)...")
    t0 = time.perf_counter()
    mapping = _build_uid_to_faiss_ids(vector_store)
    with cache_path.open("wb") as fh:
        pickle.dump(mapping, fh)
    log.info("LUT construit en %.1fs (%d uids), cache écrit dans %s",
             time.perf_counter() - t0, len(mapping), cache_path.name)
    return mapping


def _build_uid_to_faiss_ids(vector_store: FAISS) -> dict[str, list[int]]:
    """Parcourt tous les chunks de l'index pour construire le LUT inverse."""
    mapping: dict[str, list[int]] = {}
    docstore = vector_store.docstore
    for faiss_id, doc_id in vector_store.index_to_docstore_id.items():
        doc = docstore.search(doc_id)
        uid = doc.metadata.get("parent_uid") or doc.metadata.get("uid")
        if uid is None:
            continue
        mapping.setdefault(uid, []).append(faiss_id)
    return mapping


# ---------------------------------------------------------------------------
# Normalisation et matching metadata
# ---------------------------------------------------------------------------


def _normalize(s: str | None) -> str | None:
    """Casse insensible + suppression des accents (NFKD).

    Permet de matcher « Île-de-France » (metadata) contre « Ile-de-France »,
    ou « Versailles » contre « versailles »."""
    if s is None:
        return None
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()


def _normalize_region(s: str | None) -> str | None:
    """Comme `_normalize`, plus aliases anglais → français."""
    norm = _normalize(s)
    if norm is None:
        return None
    return REGION_ALIASES.get(norm, norm)


def _select_allowed_uids(
    filters: QueryFilters,
    parent_store: dict[str, Document],
) -> set[str] | None:
    """Calcule l'ensemble des `uid` qui passent les filtres exacts.

    Champs filtrés : `city`, `region`, `date_after`, `date_before`.
    Renvoie `None` si aucun de ces filtres n'est défini (= « pas de
    pre-filter, on tombe sur la voie standard »). Sinon, itère sur les
    252 901 events du `parent_store` (~50ms) et garde ceux qui matchent.

    Convention dates : un event « chevauche » la fenêtre `[date_after,
    date_before]` si `last_date >= date_after` ET `first_date <=
    date_before`. Les dates ISO 8601 se comparent lexicographiquement
    sur les 10 premiers caractères (YYYY-MM-DD). Une année entière voulue
    par l'utilisateur arrive ici sous forme de bornes (`YYYY-01-01` /
    `YYYY-12-31`), résolues en amont par l'extracteur self-querying."""
    has_city = filters.city is not None
    has_region = filters.region is not None
    has_date_after = filters.date_after is not None
    has_date_before = filters.date_before is not None

    if not (has_city or has_region or has_date_after or has_date_before):
        return None

    city_norm = _normalize(filters.city) if has_city else None
    region_norm = _normalize_region(filters.region) if has_region else None
    date_after = filters.date_after
    date_before = filters.date_before

    allowed: set[str] = set()
    for uid, parent in parent_store.items():
        meta = parent.metadata
        if has_city and _normalize(meta.get("location_city")) != city_norm:
            continue
        if has_region and _normalize_region(meta.get("location_region")) != region_norm:
            continue
        if has_date_after:
            last = meta.get("last_date") or meta.get("first_date")
            if last is None or last[:10] < date_after:
                continue
        if has_date_before:
            first = meta.get("first_date")
            if first is None or first[:10] > date_before:
                continue
        allowed.add(uid)
    return allowed


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve_parents(
    query: str,
    vector_store: FAISS,
    parent_store: dict[str, Document],
    uid_to_faiss_ids: dict[str, list[int]],
    embeddings: HuggingFaceEmbeddings,
    filters: QueryFilters | None = None,
    k_parents: int = 5,
    k_chunks: int = 15,
    fetch_k: int = 200,
    fail_open: bool = True,
) -> tuple[list[Document], bool]:
    """Recherche les `k_parents` events parents les plus pertinents.

    Pipeline (deux voies selon les filtres) :

    **Voie pre-filter** (au moins un filtre extrait — city, region,
    date_after ou date_before) :
    1. Calcule `allowed_uids` depuis le parent_store (itération ~50ms).
    2. Récupère `allowed_faiss_ids` via le LUT inverse (O(n_uids)).
    3. Reconstruit les vecteurs correspondants depuis FAISS.
    4. Calcule la distance L2 query-vecteurs en numpy (vectorisé).
    5. Garde les `k_chunks` meilleurs.

    **Voie standard** (aucun filtre extrait) :
    1. `similarity_search(query, k=k_chunks, fetch_k=fetch_k)`.

    Puis dans les deux cas : dédup par `parent_uid`, lookup parent, retour
    top `k_parents`. Fail-open si 0 résultat et `fail_open=True`.

    Retour : `(parents, filter_was_relaxed)` — le bool dit si on a
    dégradé en no-filter (utile pour logger et debugger)."""
    use_prefilter = filters is not None and not filters.is_empty()

    if use_prefilter:
        chunks = _search_prefiltered(
            query, vector_store, parent_store, uid_to_faiss_ids,
            embeddings, filters, k_chunks,
        )
    else:
        chunks = vector_store.similarity_search(query, k=k_chunks, fetch_k=fetch_k)

    parents = _dedup_to_parents(chunks, parent_store, k_parents)

    if not parents and filters is not None and not filters.is_empty() and fail_open:
        log.info("retrieve_parents: 0 résultat avec filtre %s, fail-open vers no-filter",
                 filters.as_dict())
        chunks = vector_store.similarity_search(query, k=k_chunks, fetch_k=fetch_k)
        parents = _dedup_to_parents(chunks, parent_store, k_parents)
        return parents, True

    return parents, False


def _search_prefiltered(
    query: str,
    vector_store: FAISS,
    parent_store: dict[str, Document],
    uid_to_faiss_ids: dict[str, list[int]],
    embeddings: HuggingFaceEmbeddings,
    filters: QueryFilters,
    k_chunks: int,
) -> list[Document]:
    """Recherche vectorielle restreinte aux faiss_ids des uids autorisés.

    Renvoie une liste de chunks Documents triés par distance L2 croissante.
    Renvoie `[]` si aucun uid n'est autorisé."""
    allowed_uids = _select_allowed_uids(filters, parent_store)
    if not allowed_uids:
        return []

    allowed_ids: list[int] = []
    for uid in allowed_uids:
        ids = uid_to_faiss_ids.get(uid)
        if ids:
            allowed_ids.extend(ids)

    if not allowed_ids:
        return []

    # Embedding de la question + reconstruction des vecteurs candidats
    qv = np.asarray(embeddings.embed_query(query), dtype=np.float32)
    # reconstruct_batch attend un numpy array d'int64
    ids_arr = np.asarray(allowed_ids, dtype=np.int64)
    sub_vectors = vector_store.index.reconstruct_batch(ids_arr)

    # Distance L2 — IndexFlatL2 ranke par distance croissante. On évite
    # le sqrt (préserve l'ordre) : ||q-v||² = ||q||² + ||v||² - 2·q·v.
    # Implémentation directe via numpy plus lisible et toujours rapide
    # sur ~quelques milliers de vecteurs.
    diffs = sub_vectors - qv
    sq_dists = np.einsum("ij,ij->i", diffs, diffs)

    # Tri partiel : on ne ranke vraiment que les k_chunks meilleurs
    top_n = min(k_chunks, len(sq_dists))
    top_idx = np.argpartition(sq_dists, top_n - 1)[:top_n]
    top_idx = top_idx[np.argsort(sq_dists[top_idx])]

    docstore = vector_store.docstore
    results: list[Document] = []
    for local_idx in top_idx:
        faiss_id = int(allowed_ids[local_idx])
        doc_id = vector_store.index_to_docstore_id[faiss_id]
        results.append(docstore.search(doc_id))
    return results


def _dedup_to_parents(
    chunks: list[Document],
    parent_store: dict[str, Document],
    k_parents: int,
) -> list[Document]:
    """Dédoublonne les chunks par `parent_uid`, garde l'ordre, top `k_parents`."""
    seen: set[str] = set()
    parents: list[Document] = []
    for chunk in chunks:
        uid = chunk.metadata.get("parent_uid") or chunk.metadata.get("uid")
        if uid is None or uid in seen:
            continue
        seen.add(uid)
        parent = parent_store.get(uid)
        if parent is None:
            log.warning("Parent uid=%s absent du parent_store, chunk ignoré", uid)
            continue
        parents.append(parent)
        if len(parents) >= k_parents:
            break
    return parents
