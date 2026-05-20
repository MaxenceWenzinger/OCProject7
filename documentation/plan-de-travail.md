# Plan de travail — POC RAG Puls-Events

## Résumé du projet

Livrer un **POC de chatbot RAG** pour Puls-Events, capable de répondre à des questions sur les événements culturels d'**Île-de-France** récupérés depuis l'API **Open Agenda** (≤ 1 an d'historique + événements à venir). Le système combine :

- **FAISS** (CPU) comme base vectorielle
- **Mistral local via Ollama** pour la génération (pas d'API cloud → démo offline)
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

### EPIC 1 — Initialisation du projet et environnement reproductible

*Couvre l'étape 1 de l'énoncé. Aucune dépendance.*

- **P7-1.1** Initialiser le repo Git et créer un `.gitignore` couvrant Python (caches, venv), uv (cache mais pas le lockfile), données régénérables (`data/raw/`, `data/index/`, etc.), secrets (`.env`) et OS
- **P7-1.2** Initialiser le projet uv (`uv init --bare --python 3.12`), pinner Python via `.python-version`, resserrer `requires-python = ">=3.12,<3.13"` dans le `pyproject.toml`, puis créer l'arborescence cible : `src/` (logique métier, sous-packages `data/`, `indexing/`, `rag/`), `api/` (FastAPI), `scripts/`, `tests/`, `data/`, `documentation/`, `evaluation/`
- **P7-1.3** Ajouter les dépendances runtime via `uv add` : `faiss-cpu`, `langchain`, `langchain-community`, `langchain-ollama`, `sentence-transformers` (embeddings HF), `requests`, `pandas`, `python-dotenv`, `fastapi`, `uvicorn[standard]`, `pydantic`. Le `pyproject.toml` et le `uv.lock` sont les sources de vérité — l'export `requirements.txt` est reporté en fin de projet (livrable).
- **P7-1.4** Ajouter les dépendances dev via `uv add --dev` (groupe `[dependency-groups] dev` du PEP 735) : `pytest`, `pytest-cov`, `httpx` (test API), `ragas`, `datasets`, `ruff`
- **P7-1.5** Rédiger un `README.md` minimal de bootstrap : objectifs du projet, prérequis (Python 3.12, uv, Ollama), procédure d'install (`uv sync` suffit), commandes principales (`uv run pytest`, `uv run uvicorn …`)
- **P7-1.6** Smoke-test : script `scripts/check_env.py` qui importe `faiss`, `langchain`, `sentence_transformers`, vérifie que Ollama répond sur `localhost:11434`. À lancer via `uv run python scripts/check_env.py`.
- **P7-1.7** Documenter dans le README l'installation de **Ollama** + commande `ollama pull mistral` (prérequis local hors `uv sync`)

---

### EPIC 2 — Ingestion et préparation des données Open Agenda

*Couvre l'étape 2. Dépend de l'Epic 1.*

- **P7-2.1** Explorer l'API Open Agenda (endpoint `public.opendatasoft.com`, dataset `evenements-publics-openagenda`) — un notebook jetable `notebooks/01_explore_openagenda.ipynb` ou script — pour identifier les champs disponibles, la structure des réponses, la pagination, les filtres géographiques
- **P7-2.2** Écrire `scripts/fetch_openagenda.py` : récupère les événements filtrés par région Île-de-France et fenêtre temporelle [aujourd'hui − 1 an ; aujourd'hui + 1 an], gestion de la pagination, sauvegarde brute en `data/raw/events_<date>.jsonl`
- **P7-2.3** Gérer les limites de l'API (rate-limit, taille max par requête, retries avec backoff exponentiel)
- **P7-2.4** Écrire `src/data/clean.py` : nettoyage (HTML strip sur les descriptions, dédup, suppression événements sans description ou sans date, normalisation des champs `title`/`description`/`location`/`startDate`/`endDate`/`keywords`)
- **P7-2.5** Test unitaire `tests/test_clean.py` : vérifie le strip HTML, la dédup, le rejet des entrées invalides sur un mini-fixture
- **P7-2.6** Documenter dans `documentation/data.md` la source, les filtres appliqués, les champs retenus, la taille du dataset final

---

### EPIC 3 — Indexation vectorielle FAISS

*Couvre l'étape 3. Dépend de l'Epic 2.*

- **P7-3.1** Choisir et justifier la stratégie de chunking : événement entier vs. chunks de description. Pour des événements courts, 1 événement = 1 document est probablement suffisant — à valider sur la distribution des longueurs
- **P7-3.2** Écrire `src/indexing/build_index.py` : transforme les événements nettoyés en `langchain.schema.Document` avec `page_content` (titre + description + mots-clés) et `metadata` (date, lieu, URL source, id événement)
- **P7-3.3** Intégrer le modèle d'embedding **HuggingFace** local (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` ou `intfloat/multilingual-e5-base` — multilingue car données en français) via `HuggingFaceEmbeddings`
- **P7-3.4** Construire l'index FAISS (`FAISS.from_documents`) et le sérialiser dans `data/index/` (`index.faiss` + `index.pkl`)
- **P7-3.5** Écrire `scripts/rebuild_index.py` : pipeline complet fetch → clean → index, idempotent, ré-exécutable
- **P7-3.6** Test unitaire `tests/test_indexing.py` : sur un fixture de 5 événements, vérifie que l'index se construit, contient le bon nombre de docs, et qu'une recherche par mot-clé connu remonte le bon événement en top-1
- **P7-3.7** Test de sanity manuel : 3 requêtes en français (« concert jazz », « exposition peinture », « théâtre Molière ») → vérifier la pertinence des top-5

---

### EPIC 4 — Chaîne RAG (LangChain + Mistral local)

*Couvre l'étape 4. Dépend de l'Epic 3.*

- **P7-4.1** Wrapper Ollama dans LangChain via `langchain_ollama.ChatOllama` (modèle `mistral`) — fonction `get_llm()` dans `src/rag/llm.py`
- **P7-4.2** Écrire `src/rag/chain.py` : chaîne RAG (retriever FAISS top-k + prompt template + LLM). Utiliser LCEL (`prompt | llm | parser`) plutôt que les anciennes `RetrievalQA`
- **P7-4.3** Rédiger le prompt système en français : rôle assistant culturel, consigne d'utiliser uniquement le contexte fourni, format de réponse attendu, comportement si aucun événement pertinent (« je n'ai pas trouvé d'événement correspondant »)
- **P7-4.4** Encapsuler dans une classe `RAGService` (init coûteux = chargement index + LLM une seule fois, méthode `answer(question: str) -> dict` qui renvoie réponse + sources)
- **P7-4.5** Test d'intégration `tests/test_rag.py` : 3 questions sur un mini-index fixture, assertions sur la présence d'un mot-clé attendu dans la réponse (test léger, pas une éval sémantique)
- **P7-4.6** Logging structuré : pour chaque appel, logger question / IDs des sources retournées / temps de réponse

---

### EPIC 5 — API REST FastAPI

*Couvre l'étape 5 (partie API). Dépend de l'Epic 4.*

- **P7-5.1** Squelette FastAPI dans `api/main.py` : app, route racine `/` health-check, configuration CORS minimale
- **P7-5.2** Schémas Pydantic dans `api/schemas.py` : `AskRequest { question: str }`, `AskResponse { answer: str, sources: list[Source] }`, `Source { title, date, url }`
- **P7-5.3** Endpoint `POST /ask` : injecte le `RAGService` (singleton via `lifespan` FastAPI pour éviter de recharger l'index à chaque requête), gestion d'erreur question vide → 422
- **P7-5.4** Endpoint `POST /rebuild` : déclenche `scripts/rebuild_index.py` (en arrière-plan via `BackgroundTasks` pour ne pas bloquer la réponse) — noter dans la doc qu'en prod il faudrait protéger cet endpoint
- **P7-5.5** Test fonctionnel `tests/test_api.py` avec `httpx.AsyncClient` : `/ask` répond 200 sur question valide, 422 sur question vide, présence du champ `sources`
- **P7-5.6** Vérifier la doc Swagger générée sur `/docs` et compléter les descriptions de routes / exemples Pydantic pour qu'elle soit présentable en démo

---

### EPIC 6 — Évaluation : jeu de test annoté + Ragas

*Couvre la partie évaluation des étapes 4 et 5. Peut commencer en parallèle de l'Epic 5 dès que l'Epic 4 est livrée.*

- **P7-6.1** Construire `evaluation/qa_dataset.jsonl` : 20 à 30 questions/réponses annotées manuellement à partir d'événements réels de l'index. Diversité : factuelles (« où a lieu X »), de filtre (« concerts gratuits ce mois-ci »), exploratoires (« quelque chose de fun en famille »), hors-domaine (« quel temps fera-t-il »)
- **P7-6.2** Pour chaque entrée du dataset, capturer : `question`, `ground_truth`, `expected_contexts` (IDs ou titres d'événements attendus dans le retrieval)
- **P7-6.3** Écrire `evaluation/evaluate_rag.py` : charge le dataset, fait tourner le `RAGService`, calcule les métriques Ragas (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`), exporte un rapport CSV/JSON dans `evaluation/results/`
- **P7-6.4** Documenter les seuils acceptables dans `documentation/evaluation.md` et lancer une première baseline complète
- **P7-6.5** Itération qualité : si scores faibles, ajuster (top-k retriever, prompt, modèle d'embedding) et re-mesurer — boucler 2-3 fois max, garder une trace des runs

---

### EPIC 7 — Conteneurisation Docker et CI

*Couvre l'étape 6 + l'aspect CI. Dépend des Epics 5 et 6.*

- **P7-7.1** `Dockerfile` API : base `python:3.11-slim`, install des deps, copie du code et de l'index pré-construit (l'index est buildé hors Docker pour ne pas dépendre d'Ollama au build), `CMD uvicorn api.main:app --host 0.0.0.0`
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

- **Open Agenda** : qualité variable des descriptions, événements en double, champs manquants → prévoir du temps sur le cleaning
- **Ollama en Docker** : la connexion `host.docker.internal` ne fonctionne pas pareil sur Linux vs Windows/Mac → tester tôt sur la machine cible de la démo
- **CI sans Ollama** : ne pas découvrir le dernier jour que Ragas tente d'appeler le LLM en CI sans avoir prévu de fallback
- **Démo live** : toujours avoir un plan B (capture vidéo de la démo qui marche) si un imprévu réseau survient
