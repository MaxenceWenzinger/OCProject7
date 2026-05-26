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

## Structure du projet

```
OCProject7/
├── api/                # Application FastAPI (endpoints /ask, /rebuild)
├── data/               # Données — sous-dossiers ignorés par git
│   ├── raw/                events bruts Open Agenda (~2 GB, ignorés)
│   ├── interim/            transformations intermédiaires (ignorées)
│   ├── processed/          dataset nettoyé (~400 MB, ignoré)
│   └── index/              index FAISS + parent_store (~1,5 GB, ignoré)
├── documentation/
│   ├── enonce.txt              énoncé officiel du projet
│   ├── plan-de-travail.md      suivi détaillé des epics et tâches
│   └── data.md                 référence dataset : source, schéma, cleaning, index
├── evaluation/         # Jeu de Q/R annoté + script Ragas
│   └── results/            sorties d'évaluation (ignorées)
├── scripts/            # Scripts I/O (wrappers autour de src/)
│   ├── fetch_openagenda.py     téléchargement des données brutes
│   ├── clean_events.py         pipeline de nettoyage
│   ├── build_index.py          construction de l'index FAISS
│   └── *_benchmark/profile_*.py   scripts d'exploration ponctuels
├── src/                # Logique métier (fonctions pures, testables sans I/O)
│   ├── data/clean.py           nettoyage des events
│   ├── indexing/build_documents.py   conversion event → Document(s) LangChain
│   └── rag/                    chaîne LangChain + wrapper Ollama
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
| Lancer l'API (à venir) | `uv run uvicorn api.main:app --reload` |

## État d'avancement

Suivi détaillé dans [`documentation/plan-de-travail.md`](documentation/plan-de-travail.md). Le projet est découpé en 8 epics couvrant les 6 étapes de l'énoncé.

**Livré** : Epic 1 (env), Epic 2 (ingestion + cleaning), Epic 3 (indexation FAISS parent-child).

**En cours / à venir** : Epic 4 (chaîne RAG LangChain + Ollama), Epic 5 (API FastAPI), Epic 6 (évaluation Ragas), Epic 7 (Docker + CI), Epic 8 (rapport + soutenance).

## Variables d'environnement

Pour l'instant aucune n'est requise (Ollama tourne en local sans clé). Si un `.env` devient nécessaire plus tard, un `.env.example` documentera les variables attendues.

## Licence

Projet pédagogique — usage non commercial.
