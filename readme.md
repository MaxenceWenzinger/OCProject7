# OCProject7 — POC chatbot RAG Puls-Events

Proof of Concept d'un chatbot **RAG** (Retrieval-Augmented Generation) répondant à des questions sur les événements culturels en **Île-de-France**, à partir des données publiques de l'API **Open Agenda**.

Projet réalisé dans le cadre du parcours **OpenClassrooms — Ingénieur Machine Learning**, Projet 7 « Développez un assistant pour la recommandation d'événements culturels ».

> Ce README est une version de bootstrap : il documente l'installation et la structure du projet. Le rapport technique complet (architecture, choix, résultats, perspectives) viendra en fin de projet.

## Stack technique

| Composant | Choix |
|---|---|
| Orchestration RAG | LangChain |
| Base vectorielle | FAISS (CPU) |
| Embeddings | HuggingFace `sentence-transformers` (modèle multilingue, données en français) |
| LLM de génération | **Mistral** servi localement via **Ollama** |
| API REST | FastAPI (+ Swagger auto sur `/docs`) |
| Conteneurisation | Docker |
| Évaluation | Ragas + jeu de Q/R annoté |
| Gestion deps | uv |

## Prérequis

- **Python 3.12** (le projet est strictement pinné sur 3.12 via `.python-version` et `requires-python = ">=3.12,<3.13"` dans `pyproject.toml`)
- **uv** — gestionnaire de paquets et d'environnement Python. Installation : voir [docs.astral.sh/uv/getting-started/installation](https://docs.astral.sh/uv/getting-started/installation/)
- **Ollama** — runtime LLM local. Installation : voir [ollama.com/download](https://ollama.com/download)
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

### 2. Installer le modèle Mistral via Ollama

Une fois Ollama installé et lancé :

```powershell
ollama pull mistral-small:latest
```

Vérifier qu'Ollama répond bien sur le port par défaut :

```powershell
curl http://localhost:11434/api/tags
```

### 3. Smoke-test (à venir — tâche P7-1.6)

```powershell
uv run python scripts/check_env.py
```

## Structure du projet

```
OCProject7/
├── api/                # Application FastAPI (endpoints /ask, /rebuild)
├── data/               # Données — sous-dossiers ignorés par git
│   ├── raw/                données brutes Open Agenda (ignorées)
│   ├── interim/            transformations intermédiaires (ignorées)
│   ├── processed/          dataset nettoyé (ignoré)
│   └── index/              index FAISS sérialisé (ignoré)
├── documentation/      # Énoncé, livrables, plan de travail, rapport
├── evaluation/         # Jeu de Q/R annoté + script Ragas
│   └── results/            sorties d'évaluation (ignorées)
├── scripts/            # Scripts standalone (fetch, rebuild_index, check_env)
├── src/                # Logique métier
│   ├── data/               ingestion + cleaning Open Agenda
│   ├── indexing/           construction de l'index FAISS
│   └── rag/                chaîne LangChain + wrapper Ollama
├── tests/              # Tests pytest
├── pyproject.toml      # Dépendances + métadonnées projet (source de vérité)
├── uv.lock             # Lockfile des versions exactes (commité)
└── .python-version     # Python 3.12 pinné
```

## Commandes courantes

| Action | Commande |
|---|---|
| Installer / synchroniser les deps | `uv sync` |
| Ajouter une dépendance runtime | `uv add <package>` |
| Ajouter une dépendance dev | `uv add --dev <package>` |
| Lancer un script | `uv run python scripts/<script>.py` |
| Lancer les tests | `uv run pytest` |
| Couverture | `uv run pytest --cov=src --cov=api` |
| Linter / formater | `uv run ruff check .` / `uv run ruff format .` |
| Lancer l'API (à venir) | `uv run uvicorn api.main:app --reload` |

## État d'avancement

Suivi détaillé dans [`documentation/plan-de-travail.md`](documentation/plan-de-travail.md). Le projet est découpé en 8 epics couvrant les 6 étapes de l'énoncé.

## Variables d'environnement

Pour l'instant aucune n'est requise (Ollama tourne en local sans clé). Si un `.env` devient nécessaire plus tard, un `.env.example` documentera les variables attendues.

## Licence

Projet pédagogique — usage non commercial.
