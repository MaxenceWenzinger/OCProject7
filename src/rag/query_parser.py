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

from typing import Optional

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from src.rag.llm import get_llm

# Instructions données au LLM. Pensé court car mistral-small respecte
# mieux des consignes denses qu'un mur de texte. Les exemples sont
# essentiels : ils ancrent le format attendu et les cas vides.
EXTRACTION_SYSTEM_PROMPT = """\
Tu extrais des filtres structurés depuis une question en français portant \
sur des événements culturels en France.

Champs à extraire (tous facultatifs — laisse `null` si la question ne le \
mentionne pas explicitement) :
- `city` : nom de ville française mentionné (ex : "Reims", "Lyon", "Paris"). \
Ne déduis pas la ville depuis une région ou un lieu (ex : "Bretagne" → city = null).
- `region` : nom de région française mentionné (ex : "Bretagne", \
"Île-de-France", "Auvergne-Rhône-Alpes"). Utilise l'orthographe officielle \
française avec accents.
- `year` : année entière mentionnée (ex : 2025, 2026). Uniquement si une \
année précise est dite ; sinon `null`.
- `date_after` : borne basse au format `YYYY-MM-DD` si la question évoque \
un mois, une saison ou une date précise. Exemples : "en novembre 2025" → \
"2025-11-01" ; "cet été" si l'année est claire → "2025-06-01".
- `date_before` : borne haute au format `YYYY-MM-DD` symétrique. \
"en novembre 2025" → "2025-11-30".

Règles :
- Si la question est générique ("quoi de neuf", "des concerts"), tous les \
champs valent `null`.
- N'invente jamais une valeur non mentionnée explicitement ou \
implicitement dans la question.
- Pour les dates relatives ("ce week-end", "demain", "la semaine \
prochaine"), laisse `date_after` et `date_before` à `null` — le système \
n'a pas accès à la date du jour.

Exemples :
- "Concert de jazz à Reims" → city="Reims"
- "Expos en Bretagne en 2026" → region="Bretagne", year=2026
- "Quels festivals en juillet 2025 ?" → date_after="2025-07-01", \
date_before="2025-07-31"
- "Spectacle pour enfants" → tout à null
- "Visite du château de Versailles" → city="Versailles" (Versailles est \
une ville)"""


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


def build_extractor(llm: BaseChatModel | None = None):
    """Construit un Runnable LangChain `question (str) → QueryFilters`.

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
    return prompt | structured_llm


def extract_filters(question: str, extractor=None) -> QueryFilters:
    """Helper de convenance : crée un extractor à la volée si non fourni.

    En production (RAGService), on construit l'extractor une seule fois
    et on le réutilise — cette helper sert surtout aux tests et au REPL."""
    extractor = extractor or build_extractor()
    return extractor.invoke({"question": question})
