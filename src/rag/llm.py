"""Wrapper LangChain autour du LLM utilisé par le RAG.

Deux providers supportés via l'env var `LLM_PROVIDER` :

- `mistral` (défaut) : Mistral via API cloud (`langchain-mistralai`).
  Exige `MISTRAL_API_KEY`. Plus stable que mistral-small local
  (notamment sur le JSON-strict utilisé par le self-querying et par
  Ragas), tier gratuit Mistral suffisant pour le volume du POC.
- `ollama` : Mistral-small servi localement par Ollama (mode offline,
  pas d'API key). Tradeoff documenté dans le rapport — qualité
  d'extraction et de génération inférieure, mais autonomie complète.

L'abstraction reste minimale : on renvoie un `BaseChatModel` LangChain,
les chaînes LCEL n'ont pas à connaître le provider sous-jacent.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel

DEFAULT_PROVIDER = "mistral"

# Mistral API.
# `mistral-medium-3.5` retenu après benchmarks : 50 req/min (vs 4 pour
# large, 23 pour medium-2508) et latence /ask plus faible. Plafond tokens
# plus bas (25k/min vs 356k pour medium-2508) mais workable pour le scope
# POC avec workers sérialisés côté Ragas. Surchargeable via env var
# `MISTRAL_MODEL` (ex. `mistral-large-latest` si tier payant).
DEFAULT_MISTRAL_MODEL = "mistral-medium-3.5"

# Ollama
DEFAULT_OLLAMA_MODEL = "mistral-small:latest"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

DEFAULT_TEMPERATURE = 0.0


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Construit le LLM configuré pour le provider choisi.

    Ordre de résolution : argument explicite > variable d'environnement
    > défaut. Les `kwargs` supplémentaires sont transmis tels quels au
    client sous-jacent.

    Variables d'environnement :
        LLM_PROVIDER       : `mistral` (défaut) ou `ollama`
        LLM_TEMPERATURE    : température d'échantillonnage (défaut 0.0)

    Variables spécifiques Mistral :
        MISTRAL_API_KEY    : clé API (obligatoire si provider=mistral)
        MISTRAL_MODEL      : modèle (défaut `mistral-large-latest`)

    Variables spécifiques Ollama :
        OLLAMA_MODEL       : tag du modèle (défaut `mistral-small:latest`)
        OLLAMA_HOST        : URL du serveur (défaut `http://localhost:11434`)
    """
    resolved_provider = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    resolved_temperature = _resolve_temperature(temperature)

    if resolved_provider == "mistral":
        return _build_mistral(model, resolved_temperature, **kwargs)
    if resolved_provider == "ollama":
        return _build_ollama(model, resolved_temperature, **kwargs)
    raise ValueError(
        f"LLM_PROVIDER inconnu : {resolved_provider!r}. Valeurs acceptées : "
        "'mistral', 'ollama'."
    )


def _resolve_temperature(temperature: float | None) -> float:
    if temperature is not None:
        return temperature
    env_temp = os.getenv("LLM_TEMPERATURE")
    return float(env_temp) if env_temp is not None else DEFAULT_TEMPERATURE


def _build_mistral(model: str | None, temperature: float, **kwargs: Any) -> BaseChatModel:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY n'est pas défini. Soit basculer en Ollama via "
            "LLM_PROVIDER=ollama, soit renseigner MISTRAL_API_KEY dans .env."
        )
    _install_mistral_retry_on_429()
    _install_combine_llm_outputs_patch()

    from langchain_mistralai import ChatMistralAI

    resolved_model = model or os.getenv("MISTRAL_MODEL", DEFAULT_MISTRAL_MODEL)
    # 8 tentatives avec backoff exponentiel jusqu'à 16s donne ~1-2 min de
    # patience cumulée sur les 429 du tier gratuit. `max_retries` par défaut
    # de ChatMistralAI = 5, insuffisant en concurrence Ragas.
    kwargs.setdefault("max_retries", 8)
    llm = ChatMistralAI(
        model=resolved_model,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )
    _patch_mistral_strip_fences(llm)
    return llm


# ---------------------------------------------------------------------------
# Post-traitement : retrait des code fences markdown autour des JSON
# ---------------------------------------------------------------------------

# mistral-medium-* (et beaucoup d'autres LLMs) enveloppent leurs JSON dans
# des fences ```json ... ``` malgré les instructions du prompt. Le parser
# Pydantic de Ragas reçoit le texte brut et essaie de le parser comme JSON
# → échec, métrique = NaN. On patche les méthodes generate du LLM pour
# stripper ces fences en sortie quand le content ressemble à du JSON.
import re as _re  # noqa: E402
_FENCE_RE = _re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$",
    flags=_re.DOTALL,
)


def _strip_fences(text: Any) -> Any:
    """Retire les fences markdown ```...``` si elles enveloppent tout le texte."""
    if not isinstance(text, str):
        return text
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


def _patch_mistral_strip_fences(llm: BaseChatModel) -> None:
    """Patche les méthodes `_generate`/`_agenerate` de l'instance LLM pour
    stripper les fences markdown du `content` des AIMessage retournés.

    On patche au niveau instance plutôt que classe pour ne pas affecter
    d'autres instances ChatMistralAI éventuellement créées ailleurs (tests,
    notebooks). LangChain expose `_generate`/`_agenerate` comme méthodes
    standardisées appelées par `invoke`, `ainvoke`, `batch`, etc."""
    import types

    orig_generate = llm._generate
    orig_agenerate = llm._agenerate

    def _strip_in_result(result: Any) -> Any:
        for gen in getattr(result, "generations", []):
            if hasattr(gen, "message") and hasattr(gen.message, "content"):
                gen.message.content = _strip_fences(gen.message.content)
            if hasattr(gen, "text"):
                gen.text = _strip_fences(gen.text)
        return result

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        return _strip_in_result(orig_generate(*args, **kwargs))

    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[no-untyped-def]
        return _strip_in_result(await orig_agenerate(*args, **kwargs))

    llm._generate = types.MethodType(_generate, llm)
    llm._agenerate = types.MethodType(_agenerate, llm)


_mistral_retry_installed = False


def _install_mistral_retry_on_429() -> None:
    """Monkey-patch `_create_retry_decorator` du module `langchain_mistralai`.

    `ChatMistralAI` ne retry par défaut que les erreurs réseau pures
    (`httpx.RequestError`, `httpx.StreamError`). Sur le tier gratuit
    Mistral (0.07 req/s = 4.2 req/min), tout dépassement de débit lève
    un `httpx.HTTPStatusError` 429 qui n'est pas retenté — Ragas, qui
    lance plusieurs métriques en parallèle, se prend immédiatement 100 %
    d'échecs.

    On remplace donc le decorator par une variante tenacity qui retry
    aussi sur les `HTTPStatusError` de status 429 (rate limit) ou 5xx
    (overloaded/transient server errors). Backoff exponentiel avec
    jitter, 5 tentatives. Idempotent — appelable plusieurs fois sans
    effet additionnel."""
    global _mistral_retry_installed
    if _mistral_retry_installed:
        return

    import logging
    import httpx
    from tenacity import (
        before_sleep_log,
        retry,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential_jitter,
    )
    import langchain_mistralai.chat_models as mod

    _RETRY_STATUSES = {429, 500, 502, 503, 504}
    _retry_log = logging.getLogger("mistral_retry")

    def _is_retryable(exc: BaseException) -> bool:
        if isinstance(exc, (httpx.RequestError, httpx.StreamError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in _RETRY_STATUSES
        return False

    def _create_retry_decorator(llm, run_manager=None):  # type: ignore[no-untyped-def]
        # Backoff exponentiel avec jitter sur 429/5xx. Pas de rate-limiter
        # proactif : Ragas utilise asyncio en interne (un seul thread, plusieurs
        # coroutines), donc un threading.Lock ne sérialise rien — une tentative
        # antérieure (commit dans l'historique) augmentait la latence sans
        # éliminer les 429. Le modèle Mistral est choisi de façon à ce que le
        # rate limit gratuit soit confortable : `mistral-medium-latest` à 23
        # req/min couvre largement nos besoins Ragas.
        return retry(
            reraise=True,
            stop=stop_after_attempt(llm.max_retries),
            wait=wait_exponential_jitter(initial=1, max=16),
            retry=retry_if_exception(_is_retryable),
            before_sleep=before_sleep_log(_retry_log, logging.WARNING),
        )

    mod._create_retry_decorator = _create_retry_decorator
    _mistral_retry_installed = True


_combine_outputs_patch_installed = False


def _install_combine_llm_outputs_patch() -> None:
    """Patche `ChatMistralAI._combine_llm_outputs` pour gérer les dicts nested.

    Quand Ragas appelle `agenerate_prompt([prompt × n])` pour produire
    plusieurs générations (cas `answer_relevancy` avec `strictness=3`),
    langchain-mistralai tente d'agréger les `token_usage` retournés par
    l'API. Le code amont fait `overall_token_usage[k] += v` sans gérer
    le cas où `v` est lui-même un dict (Mistral renvoie maintenant des
    sous-objets `prompt_tokens_details`, `completion_tokens_details`).
    Résultat : `TypeError: unsupported operand type(s) for +=: 'dict'
    and 'dict'` et toute la métrique passe à NaN.

    On remplace la méthode par une variante qui descend récursivement
    dans les dicts. Idempotent."""
    global _combine_outputs_patch_installed
    if _combine_outputs_patch_installed:
        return

    from langchain_mistralai import ChatMistralAI

    def _add_nested(a: Any, b: Any) -> Any:
        if isinstance(a, dict) and isinstance(b, dict):
            merged: dict = dict(a)
            for k, v in b.items():
                merged[k] = _add_nested(merged[k], v) if k in merged else v
            return merged
        if a is None:
            return b
        if b is None:
            return a
        return a + b

    def _combine_llm_outputs(self, llm_outputs: list[dict | None]) -> dict:  # type: ignore[no-untyped-def]
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                continue
            token_usage = output.get("token_usage")
            if token_usage is None:
                continue
            for k, v in token_usage.items():
                overall_token_usage[k] = _add_nested(overall_token_usage.get(k), v)
        return {"token_usage": overall_token_usage, "model_name": self.model}

    ChatMistralAI._combine_llm_outputs = _combine_llm_outputs
    _combine_outputs_patch_installed = True


def _build_ollama(model: str | None, temperature: float, **kwargs: Any) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    resolved_model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    resolved_base_url = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    return ChatOllama(
        model=resolved_model,
        base_url=resolved_base_url,
        temperature=temperature,
        **kwargs,
    )
