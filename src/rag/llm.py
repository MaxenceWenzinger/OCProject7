"""Wrapper LangChain autour d'un LLM Mistral servi localement par Ollama.

Le LLM est exposé via une fonction `get_llm()` qui renvoie un
`ChatOllama` configuré. Les paramètres sont lus depuis l'environnement
avec des valeurs par défaut, et peuvent être surchargés via arguments
(utile pour les tests et la conteneurisation Docker, où l'hôte Ollama
devient `host.docker.internal:11434`).
"""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama

DEFAULT_MODEL = "mistral-small:latest"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.0


def get_llm(
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    **kwargs,
) -> ChatOllama:
    """Construit un `ChatOllama` prêt à être inséré dans une chaîne LCEL.

    L'ordre de résolution pour chaque paramètre est : argument explicite
    > variable d'environnement > défaut. Les `kwargs` supplémentaires
    sont transmis tels quels à `ChatOllama` (ex. `num_ctx`, `top_p`).

    Variables d'environnement reconnues :
        OLLAMA_MODEL       : tag du modèle Ollama (défaut `mistral-small:latest`)
        OLLAMA_HOST        : URL du serveur Ollama (défaut `http://localhost:11434`)
        LLM_TEMPERATURE    : température d'échantillonnage (défaut 0.0)
    """
    resolved_model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    resolved_base_url = base_url or os.getenv("OLLAMA_HOST", DEFAULT_BASE_URL)

    if temperature is None:
        env_temp = os.getenv("LLM_TEMPERATURE")
        resolved_temperature = float(env_temp) if env_temp is not None else DEFAULT_TEMPERATURE
    else:
        resolved_temperature = temperature

    return ChatOllama(
        model=resolved_model,
        base_url=resolved_base_url,
        temperature=resolved_temperature,
        **kwargs,
    )
