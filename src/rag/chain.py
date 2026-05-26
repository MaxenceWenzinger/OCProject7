"""Chaîne RAG LCEL : (contexte d'événements + question) → réponse texte.

Cette chaîne **ne fait pas** le retrieval. Elle reçoit en entrée des
Documents parents déjà sélectionnés en amont (par `retrieval.py`) et les
formate avant de les passer au prompt système. Le découpage est délibéré :

- Le retrieval (similarity search FAISS + dédup par `parent_uid` + lookup
  `parent_store`) reste du Python normal, facile à logger, tester et
  étendre (filtrage metadata futur sur ville/date).
- La chaîne LCEL ne s'occupe que de ce qu'elle fait bien : formater le
  contexte, appliquer le template, appeler le LLM, parser la sortie.

Usage typique (le caller orchestre retrieval + chaîne) :

    parents = retrieve_parents(question, k_chunks=15, k_parents=5)
    chain = build_chain()
    answer = chain.invoke({"context": parents, "question": question})
"""

from __future__ import annotations

from operator import itemgetter
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from src.rag.llm import get_llm

SYSTEM_PROMPT = """\
Tu es un assistant culturel pour la plateforme Puls-Events. Tu réponds en \
français à des questions sur des événements culturels (concerts, expositions, \
spectacles, festivals, visites guidées, etc.).

Règles strictes, à appliquer dans cet ordre :

1. Si la question n'a aucun rapport avec des événements culturels \
(météo, politique, calcul, code, vie privée, etc.), réponds exactement : \
« Je ne peux répondre qu'à des questions sur les événements culturels du \
catalogue. » et n'ajoute rien d'autre.

2. Sinon, si la question porte bien sur des événements culturels mais \
qu'aucun événement de la liste ne correspond, réponds exactement : \
« Je n'ai pas trouvé d'événement correspondant. » et n'ajoute rien \
d'autre. Une question sur un type d'événement (théâtre, concert, expo, \
festival, atelier...) est culturelle même si la liste ne contient rien \
de ce type.

3. Sinon, présente les événements pertinents : une phrase d'introduction \
puis une liste à puces. Pour chaque événement cite son titre, sa ville \
et sa date quand ces informations sont disponibles. N'invente jamais \
d'événement, de date, de lieu ou de prix qui ne figurent pas dans la \
liste. Pas de paragraphe d'analyse, pas de recommandation personnelle.

Événements disponibles :
{context}"""

USER_PROMPT = "{question}"


def _format_metadata_header(meta: dict) -> str:
    """Compose la ligne d'en-tête d'un event : titre, ville, plage de dates.

    Champs absents omis. Les dates sont coupées à la date seule (les
    ISO 8601 du dataset incluent l'heure UTC, peu utile au LLM)."""
    parts: list[str] = []
    title = meta.get("title")
    if title:
        parts.append(title)

    city = meta.get("location_city")
    if city:
        parts.append(city)

    first = meta.get("first_date")
    last = meta.get("last_date")
    first_day = first.split("T", 1)[0] if isinstance(first, str) else None
    last_day = last.split("T", 1)[0] if isinstance(last, str) else None
    if first_day and last_day and first_day != last_day:
        parts.append(f"{first_day} → {last_day}")
    elif first_day:
        parts.append(first_day)
    elif last_day:
        parts.append(last_day)

    return " — ".join(parts) if parts else "(événement)"


def format_docs(docs: Iterable[Document]) -> str:
    """Sérialise une liste de Documents parents en string pour le prompt.

    Format par event :

        [N] {title} — {city} — {first_date} → {last_date}
        {page_content}

    La numérotation `[N]` permet au LLM de citer une source de façon
    traçable. Les metadata sont en en-tête car elles ne figurent pas dans
    le `page_content` (cf. `build_page_content`)."""
    blocks: list[str] = []
    for i, doc in enumerate(docs, start=1):
        header = _format_metadata_header(doc.metadata)
        blocks.append(f"[{i}] {header}\n{doc.page_content}")
    return "\n\n".join(blocks) if blocks else "(aucun événement)"


def build_prompt() -> ChatPromptTemplate:
    """Template chat à 2 messages : système (consignes + contexte) + user (question)."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )


def build_chain(llm: BaseChatModel | None = None) -> Runnable:
    """Construit la chaîne LCEL `{context, question} → réponse texte`.

    L'entrée attendue est un dict `{"context": list[Document], "question": str}`.
    L'étape de tête transforme les Documents en string formatée via
    `format_docs` ; la `question` est passée telle quelle. Le LLM injecté
    par défaut est `get_llm()` (Ollama / mistral-small)."""
    llm = llm or get_llm()
    prompt = build_prompt()
    return (
        {
            "context": itemgetter("context") | RunnableLambda(format_docs),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
