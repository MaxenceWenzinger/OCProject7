"""Extraction self-query : transformer une question utilisateur en filtres metadata structurés.

Approche : on demande au LLM (Mistral via Ollama) d'extraire les
contraintes structurées exprimées en langage naturel — ville, région,
année, plage de dates — en s'appuyant sur `with_structured_output` de
LangChain, qui contraint la sortie à un schéma Pydantic.

Tout champ non extrait reste `None` ; le caller (`retrieval.py`)
décide quels filtres appliquer effectivement. La normalisation des
écarts d'écriture (« Ile-de-France » vs « Île-de-France », « Brittany »
vs « Bretagne ») se fait en aval, dans le callable de filtre.

Coût : un appel LLM par question, ~200 tokens en sortie max, sub-seconde
sur mistral-small. Acceptable au regard de la latence totale du RAG.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnablePassthrough
from pydantic import BaseModel, Field

from src.rag.llm import get_llm

# Jours de la semaine en français, indexés sur date.weekday() (lundi=0).
_WEEKDAY_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

# Instructions données au LLM. Pensé court car mistral-small respecte
# mieux des consignes denses qu'un mur de texte. Les exemples sont
# essentiels : ils ancrent le format attendu et les cas vides.
EXTRACTION_SYSTEM_PROMPT = """\
Tu extrais des filtres structurés depuis une question en français portant \
sur des événements culturels en France.

La date du jour est {today} ({weekday}). Utilise-la pour résoudre toutes \
les références temporelles relatives.

Champs à extraire (tous facultatifs — laisse `null` si la question ne le \
mentionne pas explicitement ou implicitement) :
- `city` : nom de ville française mentionné (ex : "Reims", "Lyon", "Paris"). \
Ne déduis pas la ville depuis une région ou un lieu (ex : "Bretagne" → city = null).
- `region` : nom de région française mentionné (ex : "Bretagne", \
"Île-de-France", "Auvergne-Rhône-Alpes"). Utilise l'orthographe officielle \
française avec accents.
- `year` : année entière mentionnée (ex : 2025, 2026). Uniquement si une \
année précise est dite ; sinon `null`.
- `date_after` : borne basse au format `YYYY-MM-DD`.
- `date_before` : borne haute au format `YYYY-MM-DD`.

Règles pour les dates :

1. Résous TOUTES les expressions temporelles relatives en dates absolues \
en t'appuyant sur la date du jour : "ce dimanche", "demain", "la semaine \
prochaine", "cet été", "le mois prochain", "dans 15 jours", etc. \
Choisis l'occurrence à venir la plus proche pour les expressions ambiguës \
("ce dimanche" = le prochain dimanche ≥ aujourd'hui).

2. Une période nommée sans année ("entre juin et octobre", "en novembre") \
réfère à la prochaine occurrence à venir de cette période. Si on est déjà \
dans la période ou après, prends l'année prochaine.

3. Analyse le TEMPS GRAMMATICAL de la requête principale (pas des \
subordonnées de contexte) :
   - Futur, présent, conditionnel, impératif → si aucune date n'est \
   exprimée, pose `date_after` à la date du jour (l'utilisateur veut des \
   événements à venir).
   - Passé clair sur la requête elle-même ("quelles expos ont eu lieu", \
   "qu'est-ce qu'il y a eu") → pose `date_before` à la date du jour.
   - Ne te laisse pas tromper par des subordonnées au passé qui ne portent \
   pas sur l'événement : "j'ai entendu qu'il y avait des concerts à \
   Toulouse, peux-tu m'en lister ?" — la requête principale ("peux-tu \
   m'en lister") est au présent/conditionnel → événements à venir.

4. Si la question demande explicitement "n'importe quand", "toutes \
périodes", "passé ou futur" → laisse `date_after` et `date_before` à `null`.

5. N'invente jamais une valeur qui ne découle ni d'une mention explicite \
ni des règles ci-dessus.

Exemples (en supposant aujourd'hui = 2026-05-27, mercredi) :
- "Concert de jazz à Reims" → city="Reims", date_after="2026-05-27"
- "Expos en Bretagne en 2026" → region="Bretagne", year=2026, \
date_after="2026-05-27" (présent, futur implicite)
- "Quels festivals en juillet 2025 ?" → date_after="2025-07-01", \
date_before="2025-07-31" (année explicite, on respecte)
- "Expositions à Paris entre juin et octobre" → city="Paris", \
date_after="2026-06-01", date_before="2026-10-31" (prochaine occurrence)
- "Que se passe-t-il ce week-end ?" → date_after="2026-05-30", \
date_before="2026-05-31"
- "Quelles expos ont eu lieu à Lyon l'an dernier ?" → city="Lyon", \
year=2025, date_before="2026-05-27"
- "Spectacle pour enfants à n'importe quelle période" → tout à null pour \
les dates (l'utilisateur a explicitement levé la contrainte temporelle)"""


class QueryFilters(BaseModel):
    """Filtres metadata extraits d'une question utilisateur.

    Tous les champs sont optionnels. Un filtre absent signifie « pas de
    contrainte sur ce champ »."""

    city: Optional[str] = Field(
        default=None,
        description="Ville française mentionnée explicitement, ex: 'Reims'",
    )
    region: Optional[str] = Field(
        default=None,
        description="Région française mentionnée, ex: 'Bretagne'",
    )
    year: Optional[int] = Field(
        default=None,
        description="Année entière (2024-2030)",
    )
    date_after: Optional[str] = Field(
        default=None,
        description="Borne basse de date au format YYYY-MM-DD",
    )
    date_before: Optional[str] = Field(
        default=None,
        description="Borne haute de date au format YYYY-MM-DD",
    )

    def is_empty(self) -> bool:
        """True si aucun filtre n'est défini."""
        return all(
            getattr(self, f) is None
            for f in ("city", "region", "year", "date_after", "date_before")
        )

    def as_dict(self) -> dict:
        """Représentation compacte des filtres effectivement définis (pour les logs)."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


def _today_context() -> dict[str, str]:
    """Renvoie `{today, weekday}` à utiliser comme variables de prompt.

    Recalculé à chaque appel : `RAGService` instancie l'extracteur une
    seule fois mais peut tourner plusieurs jours, on ne veut pas figer
    la date au démarrage.

    Si l'env var `EVAL_FROZEN_DATE` est définie (format `YYYY-MM-DD`),
    on l'utilise à la place de `date.today()`. Cette porte est uniquement
    là pour rendre les runs d'évaluation reproductibles dans le temps
    (les ground truths du `qa_dataset.jsonl` ont été annotés avec une
    date système précise). En prod, la var est absente et le comportement
    est identique à `date.today()`."""
    frozen = os.environ.get("EVAL_FROZEN_DATE")
    today = date.fromisoformat(frozen) if frozen else date.today()
    return {
        "today": today.isoformat(),
        "weekday": _WEEKDAY_FR[today.weekday()],
    }


def build_extractor(llm: BaseChatModel | None = None) -> Runnable:
    """Construit un Runnable LangChain `{question} → QueryFilters`.

    Le prompt système référence `{today}` et `{weekday}` ; un
    `RunnablePassthrough.assign` les calcule à chaque invocation pour
    que la date soit toujours fraîche, même si l'API tourne plusieurs
    jours d'affilée.

    Utilise `with_structured_output` qui contraint la sortie du LLM au
    schéma Pydantic via le mode JSON natif d'Ollama. Si le LLM échoue à
    produire un JSON valide, LangChain remonte une exception ; c'est au
    caller de fail-open (cf. `RAGService`)."""
    from langchain_core.prompts import ChatPromptTemplate

    llm = llm or get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EXTRACTION_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    structured_llm = llm.with_structured_output(QueryFilters)
    return (
        RunnablePassthrough.assign(
            today=lambda _: _today_context()["today"],
            weekday=lambda _: _today_context()["weekday"],
        )
        | prompt
        | structured_llm
    )


def extract_filters(question: str, extractor: Runnable | None = None) -> QueryFilters:
    """Helper de convenance : crée un extractor à la volée si non fourni.

    En production (RAGService), on construit l'extractor une seule fois
    et on le réutilise — cette helper sert surtout aux tests et au REPL."""
    extractor = extractor or build_extractor()
    return extractor.invoke({"question": question})
