"""Construction des `Document` LangChain à partir des événements clean.

Fonctions pures sans I/O — l'orchestration streaming sur le fichier JSONL
est dans `scripts/build_index.py`.

Stratégie d'indexation : **parent-child chunking**.

- Chaque événement (le « parent ») est représenté par un `page_content`
  composé de 5 champs textuels et par une `metadata` à 10 champs. Cette
  représentation est utilisée à la génération côté LLM (contexte riche).
- Pour le retrieval, le `page_content` parent est découpé en N « chunks »
  d'au plus 120 tokens MiniLM (sous la limite des 128 tokens du modèle),
  avec un recouvrement de 24 tokens. Chaque chunk est embeddé séparément
  et indexé dans FAISS. C'est à ces chunks que la similarity search répond.
- À l'inférence, on dédoublonne les chunks retrouvés par `parent_uid` puis
  on récupère les events parents complets pour les passer au LLM.

Bénéfices : retrieval précis sur les détails enfouis, MiniLM utilisable
malgré sa fenêtre courte de 128 tokens, build rapide (~80 min).
"""

from __future__ import annotations

from typing import Any, Protocol

from langchain_core.documents import Document


class _Tokenizer(Protocol):
    """Interface minimale du tokenizer HuggingFace utilisée par le chunker.

    Permet d'injecter un fake en test sans dépendre de sentence-transformers."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str: ...


# Paramètres de chunking, calibrés pour MiniLM
# (paraphrase-multilingual-MiniLM-L12-v2, max_seq_length=128).
# 120 tokens de contenu + 2 tokens spéciaux [CLS]/[SEP] = 122, marge confortable.
CHUNK_SIZE_TOKENS = 120
CHUNK_OVERLAP_TOKENS = 24  # 20 % de CHUNK_SIZE_TOKENS


# Choix de l'attribut `lastdate_end` avec fallback sur `firstdate_end` —
# même logique que `src.data.clean.last_relevant_date`. On la duplique
# plutôt que d'importer pour garder ce module indépendant du package data.
def _last_date(event: dict[str, Any]) -> str | None:
    return event.get("lastdate_end") or event.get("firstdate_end")


def build_page_content(event: dict[str, Any]) -> str:
    """Compose le texte indexable d'un événement.

    Structure : titre, description courte, longue description, puis
    `Mots-clés : ...` et `Conditions : ...` préfixés pour donner au modèle
    d'embedding un signal sur la nature de chaque bloc. Champs absents
    omis."""
    blocks: list[str] = []

    title = event.get("title_fr")
    if title:
        blocks.append(title)

    description = event.get("description_fr")
    if description:
        blocks.append(description)

    longdescription = event.get("longdescription_fr")
    if longdescription:
        blocks.append(longdescription)

    keywords = event.get("keywords_fr")
    if keywords:
        blocks.append(f"Mots-clés : {keywords}")

    conditions = event.get("conditions_fr")
    if conditions:
        blocks.append(f"Conditions : {conditions}")

    return "\n\n".join(blocks)


def build_metadata(event: dict[str, Any]) -> dict[str, Any]:
    """Extrait les 9 champs metadata du Document.

    Pas de duplication des champs textuels déjà dans `page_content`.
    Valeurs None autorisées : ce sont des info auxiliaires, et l'API
    affichera ce qui est présent.

    Note : `event_year` n'est volontairement PAS indexé. Il reste calculé
    au cleaning (critère de validation `is_valid`) mais le filtrage par
    année se fait désormais via `first_date`/`last_date` (chevauchement de
    fenêtre), pas par égalité sur une année dérivée de la seule date de
    début."""
    return {
        "uid": event.get("uid"),
        "title": event.get("title_fr"),
        "url": event.get("canonicalurl"),
        "first_date": event.get("firstdate_begin"),
        "last_date": _last_date(event),
        "location_name": event.get("location_name"),
        "location_city": event.get("location_city"),
        "location_region": event.get("location_region"),
        "attendance_mode": event.get("attendance_mode"),
    }


def event_to_document(event: dict[str, Any]) -> Document:
    """Convertit un événement clean en `Document` LangChain « parent ».

    C'est ce Document qu'on passera au LLM en génération, après avoir
    trouvé un de ses chunks au retrieval. Pas embeddé tel quel : c'est
    `event_to_chunks` qui produit les Documents indexés."""
    return Document(
        page_content=build_page_content(event),
        metadata=build_metadata(event),
    )


def event_to_chunks(
    event: dict[str, Any],
    tokenizer: _Tokenizer,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[Document]:
    """Découpe un event en N chunks Documents pour l'indexation FAISS.

    Le découpage est fait sur les **vrais tokens** du modèle (et non en
    caractères) pour garantir que chaque chunk tient dans la fenêtre du
    modèle. Le `page_content` parent est concaténé puis tokenisé une seule
    fois, puis découpé par fenêtre glissante.

    Chaque chunk porte la metadata complète du parent + deux clés ajoutées
    `parent_uid` (= `uid`, dupliqué pour clarté côté code de jointure) et
    `chunk_index` (position 0-indexée dans le parent). Un event court tient
    en un seul chunk (index 0) ; un event long produit plusieurs chunks
    qui se recouvrent.

    Les events au `page_content` vide produisent une liste vide."""
    parent_content = build_page_content(event)
    if not parent_content:
        return []

    parent_metadata = build_metadata(event)
    parent_uid = parent_metadata["uid"]

    # Tokenisation unique du contenu parent. add_special_tokens=False car
    # on ne veut pas que [CLS]/[SEP] s'invitent au milieu du contenu —
    # le modèle les ajoutera de toute façon à l'embedding de chaque chunk.
    token_ids = tokenizer.encode(parent_content, add_special_tokens=False)

    # Cas court : tout tient dans un chunk → on garde le page_content
    # original (pas de re-décodage qui pourrait altérer le texte).
    if len(token_ids) <= chunk_size:
        return [
            Document(
                page_content=parent_content,
                metadata={**parent_metadata, "parent_uid": parent_uid, "chunk_index": 0},
            )
        ]

    # Cas long : fenêtre glissante avec pas (chunk_size - overlap)
    step = chunk_size - overlap
    chunks: list[Document] = []
    for chunk_index, start in enumerate(range(0, len(token_ids), step)):
        end = start + chunk_size
        slice_ids = token_ids[start:end]
        if not slice_ids:
            break
        chunk_text = tokenizer.decode(slice_ids, skip_special_tokens=True)
        chunks.append(
            Document(
                page_content=chunk_text,
                metadata={
                    **parent_metadata,
                    "parent_uid": parent_uid,
                    "chunk_index": chunk_index,
                },
            )
        )
        # Si on a déjà couvert la fin du texte, pas la peine d'ajouter
        # un chunk supplémentaire qui répèterait l'overlap final.
        if end >= len(token_ids):
            break

    return chunks
