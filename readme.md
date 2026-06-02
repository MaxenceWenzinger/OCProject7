# OCProject7 — POC chatbot RAG Puls-Events

Proof of Concept d'un chatbot **RAG** (Retrieval-Augmented Generation) répondant à des questions sur les événements culturels en **France**, à partir des données publiques de l'API **Open Agenda**.

Projet réalisé dans le cadre du parcours **OpenClassrooms — Ingénieur Machine Learning**, Projet 7 « Développez un assistant pour la recommandation d'événements culturels ».

> Ce README est une version de bootstrap : il documente l'installation et la structure du projet. Le rapport technique complet (architecture, choix, résultats, perspectives) viendra en fin de projet.

## Stack technique

| Composant | Choix |
|---|---|
| Orchestration RAG | LangChain |
| Base vectorielle | FAISS (CPU) |
| Embeddings | HuggingFace `sentence-transformers` (modèle multilingue, données en français) |
| LLM de génération | **Mistral** via API cloud (défaut `mistral-medium-3.5`), avec fallback **Ollama** local pour usage hors-ligne |
| API REST | FastAPI (+ Swagger auto sur `/docs`) |
| Conteneurisation | Docker |
| Évaluation | Ragas + jeu de 30 Q/R annoté |
| Gestion deps | uv |

## Prérequis

- **Python 3.12** (le projet est strictement pinné sur 3.12 via `.python-version` et `requires-python = ">=3.12,<3.13"` dans `pyproject.toml`)
- **uv** — gestionnaire de paquets et d'environnement Python. Installation : voir [docs.astral.sh/uv/getting-started/installation](https://docs.astral.sh/uv/getting-started/installation/)
- **Une clé API Mistral** — création gratuite sur [console.mistral.ai](https://console.mistral.ai/). Le tier gratuit suffit pour le POC (50 req/min sur `mistral-medium-3.5`).
- **Ollama** *(optionnel)* — runtime LLM local, uniquement si tu veux passer le RAG en mode hors-ligne via `LLM_PROVIDER=ollama`. Installation : voir [ollama.com/download](https://ollama.com/download).
- **Docker** (pour la démo / livrable final, pas indispensable en développement)

## Installation

### 1. Cloner et installer les dépendances Python

```powershell
git clone <url-du-repo>
cd OCProject7
uv sync
```

`uv sync` lit `pyproject.toml` + `uv.lock`, télécharge un Python 3.12 si besoin, crée `.venv/`, et installe **toutes** les dépendances (runtime + dev). Pour n'installer que les deps runtime :

```powershell
uv sync --no-dev
```

### 2. Configurer le fichier `.env`

Copie `.env.example` en `.env` (gitignored) et renseigne au minimum :

- **`MISTRAL_API_KEY`** — clé créée sur [console.mistral.ai](https://console.mistral.ai/). Obligatoire en `LLM_PROVIDER=mistral` (le défaut). Sans clé, `get_llm()` lève une erreur explicite au premier appel.
- **`ADMIN_TOKEN`** — Bearer token statique requis pour `POST /rebuild`. Génère-le avec :

  ```powershell
  uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

  Si absent au démarrage, l'API log un warning et `POST /rebuild` répond systématiquement `503` (fail-secure) — les autres endpoints restent fonctionnels.

Voir la section [Variables d'environnement](#variables-denvironnement) pour la liste complète.

### 3. *(Optionnel)* Installer Ollama pour le mode hors-ligne

Seulement si tu veux faire tourner le RAG sans la cloud API Mistral. Une fois Ollama installé et lancé :

```powershell
ollama pull mistral-small:latest
```

Puis dans `.env` : `LLM_PROVIDER=ollama`. La qualité d'extraction self-querying et la stabilité des prompts Ragas internes sont nettement moins bonnes qu'avec l'API cloud — décision documentée dans [`documentation/evaluation.md`](documentation/evaluation.md).

## Lancer l'API

```powershell
uv run uvicorn api.main:app --reload
```

L'API démarre sur `http://localhost:8000`. Le premier démarrage prend ~15 s (chargement du modèle d'embedding + index FAISS + parent_store + connexion Ollama). Endpoints disponibles :

| Méthode | Route | Description |
|---|---|---|
| GET | `/` | Health-check (renvoie `rag_ready: true` quand le service est chargé). |
| POST | `/ask` | Question/réponse RAG. Body : `{ "question": "..." }`. |
| POST | `/rebuild` | Reconstruction complète de l'index (fetch + clean + build, ~2 h). **Protégé par Bearer token** (`Authorization: Bearer <ADMIN_TOKEN>`). |
| GET | `/rebuild/status` | État du dernier (ou courant) job de rebuild. Non protégé. |

Documentation interactive Swagger : [http://localhost:8000/docs](http://localhost:8000/docs). Le cadenas « Authorize » accepte le token brut (sans le préfixe `Bearer`).

Exemple `/ask` au curl :

```powershell
curl -X POST http://localhost:8000/ask `
  -H "Content-Type: application/json" `
  -d '{\"question\": \"Quels concerts de jazz à Paris cet été ?\"}'
```

## Structure du projet

```
OCProject7/
├── api/                # Application FastAPI
│   ├── main.py             lifespan + routes /, /ask, /rebuild, /rebuild/status
│   ├── schemas.py          schémas Pydantic d'entrée/sortie
│   └── rebuild.py          auth Bearer + job de rebuild en BackgroundTasks
├── data/               # Données — sous-dossiers ignorés par git
│   ├── raw/                events bruts Open Agenda (~2 GB, ignorés)
│   ├── interim/            transformations intermédiaires (ignorées)
│   ├── processed/          dataset nettoyé (~400 MB, ignoré)
│   └── index/              index FAISS + parent_store (~1,5 GB, ignoré)
├── documentation/
│   ├── enonce.txt              énoncé officiel du projet
│   ├── plan-de-travail.md      suivi détaillé des epics et tâches
│   ├── data.md                 référence dataset : source, schéma, cleaning, index
│   └── evaluation.md           méthodologie Ragas, choix techniques, baseline, findings
├── evaluation/         # Évaluation Ragas
│   ├── qa_dataset.jsonl    30 Q/R annotées (5 catégories)
│   ├── evaluate_rag.py     pipeline RAG + Ragas, exports CSV + JSON
│   └── results/            sorties d'évaluation (ignorées)
├── scripts/            # Scripts I/O (wrappers autour de src/)
│   ├── fetch_openagenda.py     téléchargement des données brutes
│   ├── clean_events.py         pipeline de nettoyage
│   ├── build_index.py          construction de l'index FAISS
│   └── *_benchmark/profile_*.py   scripts d'exploration ponctuels
├── src/                # Logique métier (fonctions pures, testables sans I/O)
│   ├── data/clean.py           nettoyage des events
│   ├── indexing/build_documents.py   conversion event → Document(s) LangChain
│   └── rag/                    chaîne LangChain + dispatch LLM (Mistral / Ollama)
├── tests/              # Tests pytest
├── pyproject.toml      # Dépendances + config pytest (source de vérité)
├── uv.lock             # Lockfile des versions exactes (commité)
└── .python-version     # Python 3.12 pinné
```

**Convention** : la logique métier vit dans `src/` (testable en isolation, sans I/O ni dépendance disque/réseau), les `scripts/` ne font qu'orchestrer le I/O (argparse, logs, lecture/écriture de fichiers) autour de cette logique.

## Pipeline de données

Pour reconstruire l'index FAISS à partir de zéro, lancer les trois scripts dans l'ordre. Chacun écrit son résultat dans `data/` ; le suivant le récupère automatiquement.

```powershell
# 1. Télécharge les events bruts depuis Open Agenda → data/raw/events_<date>.jsonl
uv run python scripts/fetch_openagenda.py
# ~12 min, ~2 GB téléchargés (France métro, ~1 M events)

# 2. Nettoie et filtre (HTML, dédup, filtre temporel 2025+) → data/processed/events_clean_<date>.jsonl
uv run python scripts/clean_events.py
# ~5 min, sortie ~400 MB, ~253 k events conservés

# 3. Indexe avec parent-child chunking + embeddings MiniLM → data/index/
uv run python scripts/build_index.py
# ~1h 40min sur CPU, ~1,5 GB d'index produit (FAISS + parent_store)
```

Chaque script est idempotent et peut être lancé seul. Les fichiers de sortie sont datés (`<date>.jsonl`), donc relancer ne réécrit pas les anciens. Tous les détails (volumes, choix de filtres, paramètres de chunking, distribution des résultats) sont dans [`documentation/data.md`](documentation/data.md).

## Évaluation Ragas

Le pipeline d'évaluation charge le jeu de 30 Q/R annoté ([`evaluation/qa_dataset.jsonl`](evaluation/qa_dataset.jsonl)), fait tourner le `RAGService` sur chacune, calcule les 4 métriques Ragas (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) sur les 27 questions in-domain, et applique un check booléen sur les 3 questions hors-domaine. Sortie dans `evaluation/results/run_<timestamp>/` : un `per_question.csv` (détail par question + scores) et un `summary.json` (agrégats globaux et par catégorie).

```powershell
# Run complet (~30-35 min, dominé par le rate-limit Mistral gratuit)
uv run python evaluation/evaluate_rag.py

# Boucle de dev rapide (tirage seed=42)
uv run python evaluation/evaluate_rag.py --sample 5

# RAG seul, sans le judge Ragas (utile pour debug)
uv run python evaluation/evaluate_rag.py --skip-ragas
```

La date système utilisée par l'extracteur self-querying est figée à `2026-06-02` (variable `EVAL_FROZEN_DATE`) pour que les questions à expressions temporelles relatives (« ce week-end », « cet été ») donnent les mêmes filtres extraits d'un run à l'autre. En production l'API ne fixe pas cette variable.

Méthodologie complète, choix d'implémentation (3 patches `langchain-mistralai`, sérialisation côté Ragas, check OOD…), lecture détaillée de la baseline et findings priorisés : [`documentation/evaluation.md`](documentation/evaluation.md).

## Commandes courantes

| Action | Commande |
|---|---|
| Installer / synchroniser les deps | `uv sync` |
| Ajouter une dépendance runtime | `uv add <package>` |
| Ajouter une dépendance dev | `uv add --dev <package>` |
| Lancer un script | `uv run python scripts/<script>.py` |
| Lancer tous les tests | `uv run pytest` |
| Lancer uniquement les tests rapides (skip les tests d'intégration ML) | `uv run pytest -m "not slow"` |
| Couverture | `uv run pytest --cov=src --cov=api` |
| Linter / formater | `uv run ruff check .` / `uv run ruff format .` |
| Lancer l'API | `uv run uvicorn api.main:app --reload` |
| Évaluation Ragas (run complet) | `uv run python evaluation/evaluate_rag.py` |
| Évaluation Ragas (rapide) | `uv run python evaluation/evaluate_rag.py --sample 5` |

## État d'avancement

Suivi détaillé dans [`documentation/plan-de-travail.md`](documentation/plan-de-travail.md). Le projet est découpé en 8 epics couvrant les 6 étapes de l'énoncé.

**Livré** : Epic 1 (env), Epic 2 (ingestion + cleaning), Epic 3 (indexation FAISS parent-child), Epic 4 (chaîne RAG LangChain + Mistral), Epic 5 (API FastAPI : `/ask` + `/rebuild` + auth Bearer + Swagger), Epic 6.1-6.4 (jeu de Q/R annoté, script Ragas, méthodologie documentée, baseline du 2026-06-02).

**En cours / à venir** : Epic 6.5 (itération qualité), Epic 7 (Docker + CI), Epic 8 (rapport + soutenance).

## Variables d'environnement

Copie [`.env.example`](.env.example) en `.env` (gitignored) et renseigne les valeurs nécessaires.

| Variable | Requis pour | Notes |
|---|---|---|
| `ADMIN_TOKEN` | `POST /rebuild` | Bearer token statique requis pour déclencher la reconstruction de l'index. Génère-le avec `uv run python -c "import secrets; print(secrets.token_urlsafe(32))"`. Si absent, `/rebuild` répond `503` (fail-secure). |
| `MISTRAL_API_KEY` | RAG en mode `mistral` (défaut) | Clé créée sur [console.mistral.ai](https://console.mistral.ai/). Le tier gratuit suffit pour le POC. Si absente quand `LLM_PROVIDER=mistral`, `get_llm()` lève une erreur explicite. |
| `LLM_PROVIDER` | (optionnel) | `mistral` (défaut) ou `ollama`. Bascule le RAG + le judge Ragas d'un seul coup. |
| `MISTRAL_MODEL` | (optionnel) | Override du modèle Mistral. Défaut : `mistral-medium-3.5` (50 req/min sur le tier gratuit). |
| `OLLAMA_HOST` | (optionnel) | URL d'Ollama. Défaut : `http://localhost:11434`. En Docker : `http://host.docker.internal:11434`. |
| `OLLAMA_MODEL` | (optionnel) | Modèle Ollama servi. Défaut : `mistral-small:latest`. |
| `LLM_TEMPERATURE` | (optionnel) | Température du LLM (Mistral ou Ollama). Défaut : `0`. |
| `EVAL_FROZEN_DATE` | Évaluation Ragas | Date système figée (format `YYYY-MM-DD`) pour reproductibilité des questions à expressions temporelles. Posée automatiquement à `2026-06-02` par `evaluate_rag.py`. Ignorée en production. |

## Licence

Projet pédagogique — usage non commercial.
