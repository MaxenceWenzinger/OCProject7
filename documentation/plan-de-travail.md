# Plan de travail — POC RAG Puls-Events

## Résumé du projet

Livrer un **POC de chatbot RAG** pour Puls-Events, capable de répondre à des questions sur les événements culturels en **France entière** récupérés depuis l'API **Open Agenda**. On garde uniquement les événements dont la dernière occurrence se termine en **2025 ou après** (les événements purement passés sont écartés) — soit ~253 k events sur les ~1 M du dataset brut. Scope élargi par rapport à l'énoncé (France entière vs IDF), recentré ensuite sur les événements actuels et futurs, validé avec le professeur. Le système combine :

- **FAISS** (CPU) comme base vectorielle
- **Mistral-small local via Ollama** pour la génération (pas d'API cloud → démo offline)
- **LangChain** pour orchestrer la chaîne RAG
- **FastAPI** pour exposer `/ask` et `/rebuild`, avec Swagger auto sur `/docs`
- **Docker** pour packager l'API pour la démo locale
- **Ragas** + jeu de test annoté pour l'évaluation automatisée
- **GitHub Actions** pour CI (tests unitaires + évaluation Ragas)

### Livrables finaux (rappel)

| # | Livrable | Format |
|---|----------|--------|
| 1 | Système RAG fonctionnel | Code Python dans repo GitHub |
| 2 | API REST exposant le système | FastAPI + Docker |
| 3 | Rapport technique | PDF ou README, basé sur le template `.docx` fourni |
| 4 | Tests unitaires + jeu de test annoté | `tests/` + fichier Q/A de référence |
| 5 | Présentation soutenance | PowerPoint 10–15 slides |
| 6 | Démo live | Conteneur Docker exécuté en local |

### Définition de Done (globale)

- `docker build` + `docker run` suffisent pour lancer l'API sur une machine vierge
- `pytest` passe sans échec en CI
- L'évaluation Ragas tourne sur le jeu de test annoté et produit un score reproductible
- Le rapport technique couvre les 5 sections imposées : architecture, choix techno, modèles, résultats, pistes d'amélioration

---

## Epics et tâches

Découpage en 8 epics. Chaque tâche est dimensionnée pour ≤ une demi-journée. Les dépendances strictes entre epics sont indiquées.

---

### EPIC 1 — Initialisation du projet et environnement reproductible ✅

*Couvre l'étape 1 de l'énoncé. Aucune dépendance. **Livré.***

- [x] **P7-1.1** Initialiser le repo Git et créer un `.gitignore` couvrant Python (caches, venv), uv (cache mais pas le lockfile), données régénérables (`data/raw/`, `data/index/`, etc.), secrets (`.env`) et OS
- [x] **P7-1.2** Initialiser le projet uv (`uv init --bare --python 3.12`), pinner Python via `.python-version`, resserrer `requires-python = ">=3.12,<3.13"` dans le `pyproject.toml`, puis créer l'arborescence cible : `src/` (logique métier, sous-packages `data/`, `indexing/`, `rag/`), `api/` (FastAPI), `scripts/`, `tests/`, `data/`, `documentation/`, `evaluation/`
- [x] **P7-1.3** Ajouter les dépendances runtime via `uv add` : `faiss-cpu`, `langchain`, `langchain-community`, `langchain-ollama`, `sentence-transformers` (embeddings HF), `requests`, `pandas`, `python-dotenv`, `fastapi`, `uvicorn[standard]`, `pydantic`. Le `pyproject.toml` et le `uv.lock` sont les sources de vérité — l'export `requirements.txt` est reporté en fin de projet (livrable).
- [x] **P7-1.4** Ajouter les dépendances dev via `uv add --dev` (groupe `[dependency-groups] dev` du PEP 735) : `pytest`, `pytest-cov`, `httpx` (test API), `ragas`, `datasets`, `ruff`
- [x] **P7-1.5** Rédiger un `README.md` minimal de bootstrap : objectifs du projet, prérequis (Python 3.12, uv, Ollama), procédure d'install (`uv sync` suffit), commandes principales (`uv run pytest`, `uv run uvicorn …`)
- [x] **P7-1.6** Documenter dans le README l'installation de **Ollama** + commande `ollama pull mistral-small` (prérequis local hors `uv sync`)

---

### EPIC 2 — Ingestion et préparation des données Open Agenda ✅

*Couvre l'étape 2. Dépend de l'Epic 1. **Livré.***

- [x] **P7-2.1** `scripts/fetch_openagenda.py` : streaming depuis l'endpoint `/exports/jsonl` (et non `/exports/json` initialement prévu — `jsonl` permet un vrai streaming ligne par ligne sans charger 1 M d'objets en RAM). Scope élargi à **France entière** (décision projet validée par le professeur ; le filtre temporel « 2025+ » est appliqué plus tard, au cleaning, pour rester libre d'itérer dessus sans re-télécharger). Filtres `where=` côté API : `country_fr="France (Métropole)"` + `description_fr IS NOT NULL`. **Résultat : 1 051 298 events, 2,16 GB, 12 min.**
- [x] **P7-2.2** Retries + backoff exponentiel via `urllib3.Retry` (5 tentatives, backoff 0/2/4/8/16 s, sur 429/5xx + erreurs réseau). Écriture atomique `.tmp` → rename. Inclus dans `fetch_openagenda.py`.
- [x] **P7-2.3** `src/data/clean.py` (fonctions pures) + `scripts/clean_events.py` (wrapper streaming). Pipeline : strip HTML (BeautifulSoup), normalisation espaces, gestion `list` (`keywords_fr`/`accessibility_label_fr`), suppression surrogates Unicode isolés, parsing JSON imbriqué `attendancemode` → enum `attendance_mode`, dérivation `event_year`, validation (titre/description/année/filtre temporel `lastdate_end ≥ 2025-01-01` avec fallback `firstdate_end`), dédup sur `(title, date, location_name)`. Drop des champs `country_fr` (constant) et `category` (null à 100 %). **Résultat : 252 901 events conservés (24,1 %), 401 MB, 4 min 6 s.**
- [x] **P7-2.4** `tests/test_clean.py` — **48 tests** couvrant strip HTML, normalisation, gestion listes, surrogates, parsing attendancemode, validation, filtre temporel (lastdate_end + fallback firstdate_end), dédup, et un scénario bout-en-bout sur mini-fixture inline.
- [x] **P7-2.5** `documentation/data.md` finalisé : source, filtres, schéma des 23 champs retenus + 1 dérivé, qualité du dataset, distribution temporelle, pipeline de cleaning, chiffres finaux.

---

### EPIC 3 — Indexation vectorielle FAISS ✅

*Couvre l'étape 3. Dépend de l'Epic 2. **Livré.***

- [x] **P7-3.1** Profilage des longueurs sur le dataset clean (`scripts/profile_lengths.py`) : médiane 169 tokens, P95 411, max 2 541 — 2,5 % d'events dépassent 512 tokens. Sur recommandation du mentor, on bascule sur **parent-child chunking** : chaque event est découpé en N chunks ≤ 120 tokens (MiniLM) avec recouvrement 24 tokens, indexés séparément, dédupliqués par `parent_uid` au retrieval.
- [x] **P7-3.2** `src/indexing/build_documents.py` : `build_page_content` (concat titre/description/longdescription/keywords/conditions avec préfixes), `build_metadata` (10 champs : uid, title, url, dates, lieu, attendance_mode, event_year), `event_to_document` (parent Document complet), `event_to_chunks` (découpage tokenisé MiniLM avec recouvrement, metadata enrichie `parent_uid`+`chunk_index`). 24 tests dans `tests/test_build_documents.py` couvrant chaque fonction, **fake tokenizer** `_CharTokenizer` pour ne pas charger MiniLM en test unitaire.
- [x] **P7-3.3** Modèle d'embedding retenu : `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dim, fenêtre 128 tokens). Choix arbitré par benchmark (`scripts/benchmark_embeddings.py`) : MiniLM ~9× plus rapide qu'e5-base sur CPU et viable malgré sa fenêtre courte grâce au parent-child. Dépendance ajoutée : `langchain-huggingface`.
- [x] **P7-3.4** `scripts/build_index.py` : streaming jsonl → chunking → embedding par batchs de 5 000 → `FAISS.from_documents` puis `add_documents` → sauvegarde atomique `.tmp/` → swap. **Résultat : 252 901 events → 579 652 chunks indexés en 1h 42m, ~1,5 GB sur disque** (`index.faiss` 890 MB + `index.pkl` 395 MB + `parent_store.pkl` 289 MB).
- [x] **P7-3.5** Test d'intégration `tests/test_indexing.py` : sur un mini-fixture de 5 events thématiquement distincts (jazz, peinture, théâtre, randonnée, cuisine), exécute la chaîne complète (modèle MiniLM réel + FAISS réel dans un `tmp_path`) et vérifie (1) les 3 fichiers produits, (2) `db.index.ntotal == n_chunks`, (3) parent_store complet, (4) chaque requête mot-clé remonte le bon event en top-1 après dédup parent. **9 tests, ~41 s** (chargement modèle inclus). Marqué `@pytest.mark.slow` pour permettre `pytest -m "not slow"` (boucle de dev rapide en 0,14 s).
- [x] **P7-3.6** Sanity manuel après le build du 26 mai : 5 requêtes en français lancées sur l'index final (« concert de jazz à Paris », « exposition de peinture contemporaine », « spectacle pour enfants pendant les vacances », « visite guidée du château de Versailles », « festival de musique en plein air été 2026 »). Résultats pertinents sur 4/5 requêtes, signal que le filtrage par metadata (ville/date) sera à câbler dans la chaîne RAG en Epic 4. Détails dans `documentation/data.md`.

**Tâche initialement prévue, sciemment écartée** :
- ~~`scripts/rebuild_index.py` (pipeline complet fetch → clean → index)~~ — décision projet : on garde les trois scripts séparés (`fetch_openagenda.py`, `clean_events.py`, `build_index.py`) et on documente leur enchaînement dans le README. Pas de wrapper, parce que le contexte « projet étudiant » ne justifie pas la complexité supplémentaire (idempotence, flags `--skip-*`, gestion d'état). Endpoint `POST /rebuild` de l'API (Epic 5) appellera directement `build_index.py` sur le clean le plus récent.

---

### EPIC 4 — Chaîne RAG (LangChain + Mistral-small local) ✅

*Couvre l'étape 4. Dépend de l'Epic 3. **Livré.***

- [x] **P7-4.1** `src/rag/llm.py` : wrapper `get_llm()` autour de `langchain_ollama.ChatOllama` (modèle `mistral-small:latest`, `temperature=0`). Paramètres surchargables via env vars (`OLLAMA_MODEL`, `OLLAMA_HOST`, `LLM_TEMPERATURE`) ou arguments — la surcharge par env est utile pour Docker (Epic 7) où Ollama tourne sur l'hôte via `host.docker.internal:11434`.
- [x] **P7-4.2** `src/rag/chain.py` : chaîne LCEL `{context: list[Document], question: str} → str` au format `{...} | prompt | llm | StrOutputParser()`. Option A retenue (chaîne fine, génération seule) : le retrieval est fait en Python hors-chaîne par `retrieval.py`, le pipe LCEL ne s'occupe que de la mise en forme du contexte (`format_docs` numéroté avec en-tête titre/ville/date) et de la génération. Plus simple à tester et à débugger qu'une chaîne « full RAG » avec retriever inclus, et le besoin de renvoyer des sources structurées (`{answer, sources}`) côté API est plus naturel sans `RunnableParallel`.
- [x] **P7-4.3** Prompt système final écrit directement dans `chain.py` (P7-4.2). Trois règles hiérarchisées : (1) hors-domaine → « Je ne peux répondre qu'à des questions sur les événements culturels du catalogue. », (2) catalogue vide pour la question → « Je n'ai pas trouvé d'événement correspondant. », (3) sinon liste à puces avec titre/ville/date. L'ordre explicite a corrigé une confusion observée au premier test entre les deux fallbacks. Tâche fusionnée avec P7-4.2.
- [x] **P7-4.4** `src/rag/service.py` : classe `RAGService` qui orchestre **extraction self-query → retrieval pre-filtré → génération**, et renvoie un dict riche `{answer, sources, filters_used, filter_relaxed, timings}`. Init coûteux (~15s : embeddings + index FAISS + parent_store + LUT + Ollama) une seule fois ; chaque `answer(q)` enchaîne deux appels LLM (extraction + génération). Décomposition de la latence sur la machine de dev (mistral-small via Ollama, hors cold start) : extract ~7s + retrieve ~0.4s + generate ~11s = **~18s par requête**. Deux modules dédiés écrits dans la foulée :
   - `src/rag/query_parser.py` — **self-querying** : extracteur LLM en `with_structured_output(QueryFilters)` (Pydantic). Extrait `city`, `region`, `year`, `date_after`, `date_before`. Fail-safe : si l'extraction plante (JSON invalide, Ollama down), `RAGService.answer` dégrade en filtres vides plutôt que de remonter l'erreur. Mistral-small en JSON-strict est plus lent que prévu (~6-8s pour 50 tokens en sortie) — acceptable mais à garder à l'esprit.
   - `src/rag/retrieval.py` — **pre-filtering vrai** sur city/region/year via un LUT inverse `uid → list[faiss_id]` construit au premier démarrage et mis en cache disque (`data/index/uid_to_faiss_ids.pkl`, ~30 MB, ~1.6s à reconstruire, invalidation auto par mtime check sur `index.faiss`). Quand un filtre exact est extrait, on calcule l'ensemble d'`uids` autorisés, on reconstruit leurs vecteurs FAISS via `reconstruct_batch`, et on calcule la distance L2 en numpy direct. Plus de problème de `fetch_k` trop petit pour les filtres rares (avant : 0 résultat pour `city="Reims"` car les 200 chunks les plus similaires à « jazz » étaient à Paris/Lyon). Les filtres temporels (`date_after`/`date_before`) restent en post-filter — un LUT par date serait disproportionné pour un POC. Fail-open : si filtre extrait → 0 parents → on relance sans filtre, et on remonte le drapeau `filter_relaxed` au caller. Normalisation casse-insensible + suppression d'accents + aliases anglais→français (« Brittany » → « Bretagne ») pour absorber les variantes du dataset.
- [x] **P7-4.5** Tests d'intégration `tests/test_rag.py` (5 tests, marqués `@pytest.mark.slow`) sur le mini-index partagé via `tests/conftest.py` (refactor : fixture `built_index` extraite de `test_indexing.py`, scope session, partagée entre les deux fichiers, le build n'est payé qu'une fois). Fixture `ollama_available` qui ping `localhost:11434/api/tags` au démarrage de session et `pytest.skip(...)` si pas de réponse — la CI sans Ollama affichera SKIPPED, pas FAILED. Couvre : contrat ValueError sur question vide, shape du dict retourné, question thématique (« jazz » → ev-jazz en source + mot dans la réponse), self-query avec ville (« Lyon » → `filters_used.city="lyon"` + ev-cuisine en source), fallback hors-domaine. **5 tests, ~80s** (~15s init + 4×~16s par appel `answer()`).

**Tâche initialement prévue, sciemment écartée** :
- ~~**P7-4.6** Logging structuré (question / IDs sources / temps de réponse)~~ — décision projet : l'énoncé n'en parle pas, et `service.py` log déjà une ligne récap par requête (`filters`, `n_sources`, timings extract/retrieve/generate/total) via le `logging` standard. Suffisant pour la démo et pour le rapport technique ; formaliser plus (JSON, agrégation) supposerait des consommateurs aval qui n'existent pas dans le scope du POC. Un `logging.basicConfig(level=INFO, ...)` centralisé sera ajouté dans `api/main.py` en Epic 5 pour que ces logs apparaissent au démarrage de l'API.

**Décisions techniques notables prises pendant l'Epic** :
- **LCEL plutôt que `RetrievalQA`** : la classe historique est dépréciée depuis LangChain 0.2, son prompt par défaut est en anglais et caché, et surtout elle ne sait pas faire le parent-child custom (pas d'endroit où insérer notre dédup par `parent_uid`).
- **Self-querying via LLM plutôt que NER ou pas de filtre** : extraire les filtres avec regex ou un modèle léger est fragile sur les villes françaises ; pas de filtre du tout perdait les questions du type « concert à Reims » dans le bruit (Reims a 980 events, mais aucun ne remonte par similarité pure sur « concert de jazz »).
- **Pre-filter vrai plutôt que `fetch_k` escaladant** : on a regardé les deux. Le retry escaladant (200 → 2000 → 10000) était plus simple (~10 lignes) mais ne garantit pas de couvrir les filtres ultra-rares. Le pre-filter vrai avec LUT inverse cache (~80 lignes) ajoute ~5s au premier démarrage, ~1.6s aux suivants, et coûte ~370ms/requête (itération sur 252k events du parent_store), mais garantit que Reims-jazz remonte les 5 vrais events à Reims. Choisi pour la qualité de retrieval ; on aurait pu argumenter à l'inverse pour un POC, en justifiant « FAISS n'est pas le bon outil au-delà, il faudrait Qdrant/pgvector ». Trace conservée dans le commit history.

---

### EPIC 5 — API REST FastAPI ✅

*Couvre l'étape 5 (partie API). Dépend de l'Epic 4. **Livré.***

Endpoints imposés par l'énoncé (cités tels quels) : « Un endpoint `/ask` (POST) qui prend une question et renvoie une réponse générée. Un endpoint `/rebuild` (GET ou POST) pour reconstruire la base vectorielle à la demande. ». Plus la recommandation : « Protégez les endpoints sensibles (comme `/rebuild`) s'ils étaient un jour exposés publiquement. ».

- [x] **P7-5.1** Squelette FastAPI dans `api/main.py` : app + `HealthResponse` typé sur `GET /` + `logging.basicConfig(level=INFO, ...)` + `lifespan` qui instancie le `RAGService` singleton, charge `RebuildState.load()` depuis `data/index/rebuild_state.json`, et log un warning bruyant si `ADMIN_TOKEN` n'est pas défini. Pas de CORS — l'API n'est pas appelée depuis un navigateur dans le scope du POC.
- [x] **P7-5.2** Schémas Pydantic dans `api/schemas.py` : `AskRequest { question: str (min_length=1) }`, `AskResponse { answer, sources, filters_used: dict, filter_relaxed: bool }`, `Source { uid, title, description, url, location_city, first_date, last_date }` (champ `description` ajouté en cours d'epic pour donner plus de contexte à l'utilisateur, extrait à la volée depuis le `page_content` du parent — voir P7-5.6 ci-dessous), `RebuildResponse { status, started_at }`, `RebuildStatusResponse { in_progress, started_at?, finished_at?, last_error? }`, plus `HealthResponse { status, rag_ready }`. Exemples Swagger sur `AskRequest` et `Source`.
- [x] **P7-5.3** Endpoint `POST /ask` : injecte le `RAGService` depuis l'état du `lifespan`. `rebuild_in_progress=True` → **`503`** avec message explicite. Service non initialisé → `503`. Question vide → `422` (validation Pydantic). Question valide → `service.answer(question)` puis mapping vers `AskResponse`. Helper `_extract_short_description` qui découpe `page_content` sur `\n\n` et prend le 2e bloc (description courte) en sautant le titre et les blocs préfixés `Mots-clés :`/`Conditions :` — plafonné à 300 chars + ellipsis Unicode.
- [x] **P7-5.4** Endpoint `POST /rebuild` **protégé par Bearer token statique** + endpoint `GET /rebuild/status` non protégé. Code dans `api/rebuild.py`.
   - **Auth** : header `Authorization: Bearer <token>` comparé à `os.getenv("ADMIN_TOKEN")` via `secrets.compare_digest` (constant-time). Token généré une fois (`secrets.token_urlsafe(32)`), stocké dans `.env` (gitignored), documenté dans le README + `.env.example`. Refus → `401 Unauthorized` avec header `WWW-Authenticate: Bearer`. `ADMIN_TOKEN` absent → `503` (fail-secure). Dépendance FastAPI `verify_admin_token` réutilisable.
   - **Portée du rebuild** : **fetch + clean + build** dans cet ordre. Appelle directement les fonctions métier des scripts existants — `scripts.fetch_openagenda.fetch(...)`, `scripts.clean_events.clean_stream(...)`, `scripts.build_index.build(...)` — sans créer de `scripts/rebuild_index.py` centralisé (cohérent avec la décision Epic 3). Imports paresseux dans `run_rebuild` pour ne pas payer le coût au démarrage de l'API.
   - **Concurrence** : `app.state.rebuild_in_progress: bool` levé avant le job, baissé en `finally`. `/ask` check ce flag → `503` pendant le rebuild. Deuxième `POST /rebuild` simultané → `409 Conflict`.
   - **Asynchronisme** : la requête HTTP renvoie `202 Accepted` immédiatement via `BackgroundTasks` FastAPI ; le job tourne en arrière-plan dans le même process. ~12 min (fetch) + 4 min (clean) + 1h42 (build) = ~2h sur la machine de dev.
   - **Hot-swap de l'index** : à la fin du build, on instancie un nouveau `RAGService()` et on remplace `app.state.rag_service`. Si l'init plante (manque RAM, fichier corrompu), on garde l'ancien service et on log l'erreur dans `state.last_error` — l'API reste fonctionnelle.
   - **`GET /rebuild/status`** non protégé : renvoie l'état + horodatages début/fin + dernière erreur éventuelle.
   - **Persistance de l'état** : `RebuildState` exposé via dataclass + `save()` (écriture atomique `.tmp` → rename dans `data/index/rebuild_state.json`) + `load()` (état vide si fichier absent). `in_progress` n'est pas persisté (fail-safe : un crash pendant rebuild ne ressort pas avec un flag bloqué). Le fichier est gitignored. Décidé en cours d'epic pour qu'un redémarrage d'API ne perde pas `started_at`/`finished_at`/`last_error`.
- [x] **P7-5.5** Test fonctionnel `tests/test_api.py` avec `fastapi.testclient.TestClient` synchrone + `lifespan` :
   - `/ask` répond 200 sur question valide avec sources non-vides + payload aligné sur le dict de `RAGService.answer()`
   - `/ask` répond 422 sur question vide / champ manquant
   - `/ask` répond 503 pendant rebuild
   - `/rebuild` sans token → 401, mauvais token → 401, `ADMIN_TOKEN` absent → 503
   - `/rebuild` avec bon token → 202, BackgroundTask exécutée
   - `/rebuild` pendant rebuild → 409
   - `/rebuild/status` (idle, in_progress, sans auth) → 200
   - `/` health-check
   - **Choix** : `TestClient` (synchrone) plutôt que `httpx.AsyncClient` mentionné dans le plan initial — évite d'ajouter `pytest-asyncio` aux deps. Mock `MagicMock()` du `RAGService` et `monkeypatch.setattr` de `api.main.run_rebuild` — aucune charge réelle d'index ni de fetch Open Agenda en CI. **14 tests, ~3,5 s, marqués `not slow`.**
- [x] **P7-5.6** Swagger `/docs` polissé : tous les schémas Pydantic ont un `response_model=` typé (y compris `HealthResponse`, pour éviter le placeholder `additionalProp1`), `summary` + docstring sur chaque route, `responses` documentés (401/409/422/503), exemples Pydantic via `json_schema_extra` sur les schémas d'entrée. Bearer Auth câblé via `fastapi.security.HTTPBearer(auto_error=False)` → cadenas Authorize fonctionnel pour la démo (coller le token sans le préfixe `Bearer`).

**Améliorations qualité menées en plus du périmètre P7-5.x** (suite aux tests live via Swagger) :

- **Pre-filter étendu aux dates** (`src/rag/retrieval.py`) : `date_after` et `date_before` étaient en post-filter, ce qui causait des résultats vides pour les questions « Paris entre juin et octobre » (sur 15 chunks top-similarité Paris, presque aucun n'était dans la fenêtre temporelle, donc 0-1 source au final). Désormais les dates sont intégrées à `_select_allowed_uids` au même titre que city/region/year — pre-filter complet, plus de post-filter, `_build_date_filter` supprimé.
- **Prompt extracteur self-querying enrichi avec la date système** (`src/rag/query_parser.py`) : Mistral-small produisait `date_after=2023-06-01` quand on demandait « entre juin et octobre » (cutoff de son entraînement). Solution : injection de `{today}` et `{weekday}` via `RunnablePassthrough.assign` à chaque invocation (frais à chaque appel, pas figé au démarrage), prompt système réécrit pour exiger la résolution des expressions relatives (« ce dimanche », « cet été »...), l'analyse du temps grammatical de la **requête principale** (présent/futur/conditionnel → `date_after = aujourd'hui` ; passé sur la requête elle-même → `date_before = aujourd'hui`), et la levée explicite via « n'importe quand » → tout à null. Validé sur 6 cas dont le piège « j'ai entendu qu'il y avait... pourrais-tu m'en lister » (subordonnée au passé, requête au conditionnel).
- **`k_parents`/`k_chunks` augmentés** (`src/rag/service.py`) : passés de 5/15 à **10/30**. Mistral-small encaisse facilement 10 parents (~3-6k tokens, fenêtre 32k). Ratio chunks/parents conservé à 3. Améliore les réponses sur les questions ouvertes (« liste-moi des expos ») au prix d'un léger surcoût de génération.

**Note rapport (section perspectives)** : ce qui manque pour une vraie prod sur `/rebuild` — rotation du token, audit log des accès admin, HTTPS obligatoire, IP allowlist, rate-limiting global. Conforme à la formulation de l'énoncé (« s'ils étaient un jour exposés publiquement »).

**Tâche initialement prévue, sciemment écartée** :
- ~~Tests via `httpx.AsyncClient`~~ — `TestClient` synchrone (qui enveloppe httpx) couvre 100 % du besoin, gère le lifespan via context manager, et évite l'ajout de `pytest-asyncio` aux dev deps.

---

### EPIC 6 — Évaluation : jeu de test annoté + Ragas

*Couvre la partie évaluation des étapes 4 et 5. 6.1 à 6.4 livrées, 6.5 en cours.*

- [x] **P7-6.1** Construire `evaluation/qa_dataset.jsonl` : 30 questions/réponses annotées manuellement à partir d'événements réels de l'index, échantillonnage stratifié sur villes (Paris/grandes/rares), régions, années 2025/2026 et thèmes. Réparties en 5 catégories : 6 `factual`, 7 `filter_geo`, 7 `filter_temporal`, 7 `exploratory`, 3 `out_of_domain`.
- [x] **P7-6.2** Pour chaque entrée, schéma enrichi : `id`, `category`, `question`, `ground_truth`, `expected_contexts` (UIDs), `expected_filter` (filtre attendu du self-querying pour validation), `notes`. Tâche fusionnée avec 6.1.
- [x] **P7-6.3** `evaluation/evaluate_rag.py` : charge le dataset, fait tourner `RAGService`, calcule les 4 métriques Ragas sur les in-domain, check booléen séparé sur les out_of_domain (regex sur le pattern de refus). Export CSV+JSON dans `evaluation/results/run_<ts>/`. Flags `--sample N`, `--skip-ragas`. Date système figée via `EVAL_FROZEN_DATE` (défaut `2026-06-02`) pour reproductibilité.
- [x] **P7-6.4** `documentation/evaluation.md` rédigé en référence complète : méthodologie, choix d'implémentation (3 patches Mistral + frozen date), commandes, baseline complète du 2026-06-02 commentée par catégorie, findings priorisés. Pas de seuils chiffrés (purement descriptif).
- [ ] **P7-6.5** Itération qualité : si scores faibles, ajuster (top-k retriever, prompt, modèle d'embedding) et re-mesurer — boucler 2-3 fois max, garder une trace des runs

**Décisions techniques notables prises pendant l'Epic** :
- **Bascule Ollama → Mistral API** (commit pendant l'Epic). Initialement mistral-small via Ollama (POC offline) ; révélation en 6.3 que (a) extracteur hallucinait dates fabriquées, (b) ~40 % des prompts Ragas internes échouaient en `RagasOutputParserException`. Bascule sur `mistral-medium-3.5` via cloud (`langchain-mistralai`), conserve Ollama via `LLM_PROVIDER=ollama`. Démo Docker n'est plus offline, tradeoff documenté.
- **Trois patches d'infrastructure dans `src/rag/llm.py`** (Mistral API) : retry custom sur 429/5xx (Ragas en concurrence saturait le quota gratuit), strip des fences markdown autour des JSON outputs (mistral-medium les ajoute systématiquement, parser Pydantic Ragas s'en étouffait), aggrégation récursive de `token_usage` (vrai bug `langchain-mistralai._combine_llm_outputs` sur les dicts imbriqués `prompt_tokens_details`, expose `TypeError: += dict and dict` sur `answer_relevancy` qui appelle `agenerate_prompt(prompts × strictness=3)`). Détails dans `documentation/evaluation.md`.
- **`max_workers=1` côté Ragas** : avec 4 workers, 429 et « capacity exceeded » (3505) s'enchaînaient et provoquaient des TimeoutError malgré le retry. En série on absorbe les pauses sans empiler les retries.
- **Check OOD maison séparé** : Ragas ne sait pas évaluer un refus, on contrôle la formulation attendue à coup de regex sur la règle 1 du prompt système.

**Baseline du 2026-06-02** (`run_20260602_230903/`, mistral-medium-3.5, commit 4de4d46) :

| Métrique | Score | Lecture |
|---|---:|---|
| `faithfulness` | 0.39 | Hallucinations fréquentes quand le retriever ne ramène pas la bonne info |
| `answer_relevancy` | 0.73 | Les réponses traitent bien le sujet posé |
| `context_precision` | 0.22 | Beaucoup de bruit |
| `context_recall` | 0.28 | Events attendus souvent manqués |
| OOD pass rate | 2/3 | q30 a refusé mais avec le fallback « pas d'event » au lieu de la garde hors-domaine |

5 findings priorisés pour 6.5 (cf. `evaluation.md`) :
1. Retrieval rate les noms propres rares (q03, q04) — limite MiniLM
2. Pre-filter date trop laxiste sur fenêtres larges — limite du dataset (1 event annoté pour des dizaines de matches valides)
3. Pas de mapping toponymique implicite (q12 Vercors → Lans-en-Vercors)
4. Check OOD trop strict (q30, faux négatif)
5. `faithfulness` faible (0.39) → renforcer prompt système contre l'inférence depuis les connaissances générales du LLM

---

### EPIC 7 — Conteneurisation Docker et CI

*Couvre l'étape 6 + l'aspect CI. Dépend des Epics 5 et 6.*

- **P7-7.1** `Dockerfile` API : base `python:3.12-slim`, install des deps, copie du code et de l'index pré-construit (l'index est buildé hors Docker pour ne pas dépendre d'Ollama au build), `CMD uvicorn api.main:app --host 0.0.0.0`
- **P7-7.2** Gérer la connexion à Ollama depuis le conteneur : Ollama tourne sur l'hôte → `host.docker.internal:11434` (variable d'env `OLLAMA_HOST`)
- **P7-7.3** (Optionnel) `docker-compose.yml` orchestrant API + Ollama si on veut tout containeriser
- **P7-7.4** Test E2E manuel : `docker build` → `docker run` → `curl POST /ask` répond correctement
- **P7-7.5** Workflow GitHub Actions `.github/workflows/ci.yml` : matrix Python 3.11, lint (ruff), `pytest` avec couverture, upload du rapport coverage en artifact
- **P7-7.6** Workflow `.github/workflows/eval.yml` : déclenchable manuellement (`workflow_dispatch`), exécute `evaluate_rag.py` sur un mini-jeu de test (les 20-30 Q/R), commit le rapport ou l'expose en artifact. Note : Ollama non disponible en CI → soit mocker le LLM, soit utiliser un petit modèle HF en fallback dédié à la CI
- **P7-7.7** Documenter dans le README la commande complète pour reproduire l'environnement et lancer la démo

---

### EPIC 8 — Rapport technique et soutenance

*Dépend des Epics 6 et 7 (besoin des résultats d'évaluation).*

- **P7-8.1** Squelette du rapport technique (Markdown ou Word à partir du template `.docx`) avec les 5 sections imposées : architecture, choix techno, modèles, résultats, perspectives
- **P7-8.2** Rédiger section **architecture** : schéma de la chaîne RAG (ingestion → index → retrieval → génération → API), diagramme Mermaid ou image
- **P7-8.3** Rédiger section **choix technologiques** : justifier FAISS vs alternatives, Ollama-Mistral vs API cloud, FastAPI vs Flask, embeddings multilingues
- **P7-8.4** Rédiger section **modèles** : modèle d'embedding, modèle de génération, paramètres (top-k, température, taille de chunk)
- **P7-8.5** Rédiger section **résultats** : tableau des scores Ragas, exemples de réponses (bonnes et mauvaises), analyse qualitative
- **P7-8.6** Rédiger section **perspectives d'amélioration** : reranking, hybrid search BM25+vector, historique de conversation, fine-tuning, monitoring en prod, sécurisation `/rebuild`
- **P7-8.7** Finaliser le README projet (objectifs, archi, install, run, tests, évaluation) — c'est aussi un livrable
- **P7-8.8** Créer le support PowerPoint (10–15 slides) : problème → solution → archi → démo → résultats → perspectives
- **P7-8.9** Préparer 2-3 scénarios de démo live testés à l'avance (ex : « Quels concerts de jazz à Paris cette semaine ? », « Une expo d'art contemporain ce week-end ? », « Du théâtre pour enfants ? »)
- **P7-8.10** Répétition complète de la soutenance + checklist anti-bug (Ollama lancé, conteneur up, exemples copiables)

---

## Suggestions d'ordonnancement

Chemin critique : **Epic 1 → 2 → 3 → 4 → 5 → 7 → 8**.

Parallélisable :
- **Epic 6** peut démarrer dès que l'Epic 4 est fonctionnelle (avant l'Epic 5)
- **Epic 8.1 à 8.3** (sections architecture / techno / modèles du rapport) peuvent être amorcées dès la fin de l'Epic 4 — pas besoin d'attendre les résultats finaux pour décrire les choix faits
- Le PowerPoint (**Epic 8.8**) peut être préparé en parallèle du rapport

## Risques identifiés à surveiller

- **~~Open Agenda~~** ✅ matérialisé et géré en Epic 2 : ~3 % de doublons sémantiques, ~7 % hors-France à filtrer, HTML omniprésent dans `longdescription_fr`, champs `list` au lieu de `str` sur certains, surrogates Unicode corrompus, `category` null à 100 %. Tout est traité dans `clean.py` (cf. `documentation/data.md`).
- **~~Volume de l'index~~** ✅ matérialisé et géré en Epic 3 : ~253 k events → 579 652 chunks (parent-child) embeddés en 1h 42m sur CPU avec MiniLM. Index final ~1,5 GB sur disque, chargement ~15 s, latence retrieval ~50 ms par requête. Conforme au budget mémoire d'une API FastAPI.
- **Ollama en Docker** : la connexion `host.docker.internal` ne fonctionne pas pareil sur Linux vs Windows/Mac → tester tôt sur la machine cible de la démo
- **CI sans Ollama** : ne pas découvrir le dernier jour que Ragas tente d'appeler le LLM en CI sans avoir prévu de fallback
- **Démo live** : toujours avoir un plan B (capture vidéo de la démo qui marche) si un imprévu réseau survient
