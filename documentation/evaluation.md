# Évaluation du RAG — méthodologie et résultats

Documentation de référence sur la stratégie d'évaluation du POC : jeu de
test annoté, métriques Ragas, choix d'implémentation, lecture de la
baseline produite et findings qualité priorisés.

## Vue d'ensemble

L'évaluation repose sur trois briques :

1. **Un jeu de test annoté à la main** : `evaluation/qa_dataset.jsonl`,
   30 questions/réponses structurées en 5 catégories pour stresser
   chaque facette du système (factuel, filtre géo, filtre temporel,
   exploratoire, hors-domaine).
2. **Ragas 0.4.3** : framework standard pour évaluer un RAG, calcule
   4 métriques LLM-as-judge sur les questions in-domain.
3. **Un check booléen maison** pour les questions hors-domaine : Ragas
   ne sait pas vraiment évaluer un refus, on contrôle la formulation
   attendue à coup de regex.

Le pipeline complet est encapsulé dans `evaluation/evaluate_rag.py`.
Une invocation produit un dossier horodaté `evaluation/results/run_<ts>/`
avec deux fichiers : `per_question.csv` (détail par question + scores) et
`summary.json` (agrégats globaux et par catégorie + métadonnées du run).

## Comprendre Ragas en deux minutes

Ragas évalue un RAG sur quatre axes **indépendants**, chacun via un
LLM-as-judge qui décompose ce qu'il observe et tranche. Les entrées
sont systématiquement : la `question`, l'`answer` réellement générée
par le système, les `retrieved_contexts` réellement remontés, et le
`ground_truth` annoté à la main.

| Métrique | Ce qu'elle mesure | Question implicite |
|---|---|---|
| **faithfulness** | La réponse est-elle ancrée dans les contextes ? | « Le LLM hallucine ou cite vraiment ses sources ? » |
| **answer_relevancy** | La réponse répond-elle à la question ? | « Le LLM répond à côté ? » |
| **context_precision** | Les contextes remontés sont-ils tous pertinents ? | « Le retriever ramène-t-il du bruit ? » |
| **context_recall** | Les contextes remontés couvrent-ils le ground truth ? | « Le retriever a-t-il loupé l'info clé ? » |

Important pour ne pas se tromper de lecture : Ragas **ne compare pas la
réponse au ground_truth mot à mot**. Le `ground_truth` est utilisé comme
*cible* pour évaluer le retriever (CP/CR), pas pour pointer une réponse
attendue littérale. `faithfulness` et `answer_relevancy` ne le regardent
même pas. C'est pour ça qu'un dataset annoté Ragas peut couvrir des
questions exploratoires où il n'y a pas une seule bonne réponse.

## Jeu de test annoté (`qa_dataset.jsonl`)

30 questions construites à partir d'événements réels de l'index — chacun
sélectionné via un échantillonnage stratifié manuel sur villes (Paris,
grandes villes, petites villes rares pour stresser le pre-filter),
régions, années (2025/2026) et thèmes (musique, expo, théâtre, sport,
patrimoine, etc.).

### Schéma enrichi (1 ligne JSON par question)

```json
{
  "id": "q01",
  "category": "factual",
  "question": "Quand a lieu le Marathon de Bordeaux ?",
  "ground_truth": "Le Marathon de Bordeaux a lieu le dimanche 8 novembre 2026 à Bordeaux.",
  "expected_contexts": ["80404875"],
  "expected_filter": {},
  "notes": "Event nommé, date précise, ville explicite."
}
```

Le schéma standard Ragas ne demande que `question`, `ground_truth` et
`expected_contexts`. Les trois champs supplémentaires (`category`,
`expected_filter`, `notes`) servent à analyser les échecs **par type**
dans le rapport et à valider que le pré-filtre self-querying extrait
bien ce qu'on attendait.

### Répartition par catégorie

| Catégorie | n | Ce qu'elle stresse |
|---|---:|---|
| `factual` | 6 | Réponse précise sur un event nommé. Référence pour la précision retrieval. |
| `filter_geo` | 7 | Pre-filter ville/région (dont villes rares : Reims, Charleville-Mézières). |
| `filter_temporal` | 7 | Pre-filter date (fenêtres explicites, expressions relatives, années). |
| `exploratory` | 7 | Sémantique sans filtre dur (« quelque chose d'insolite »). |
| `out_of_domain` | 3 | Garde du prompt système : météo, connaissance générale, assistance générique. |

## Choix d'implémentation

### Date système figée pour reproductibilité

Plusieurs questions du dataset contiennent des expressions temporelles
relatives résolues par l'extracteur self-querying (« ce week-end »,
« cet été », « à venir »…). Si on laisse la date système courir, le
même run lancé deux jours différents extrait des `date_after` / `date_before`
différents, modifie le pre-filter et casse la comparabilité.

`evaluate_rag.py` pose donc `EVAL_FROZEN_DATE=2026-06-02` (la date à
laquelle le dataset a été annoté) avant tout import LLM. La porte est
lue par `src/rag/query_parser._today_context()`, qui l'utilise à la
place de `date.today()` quand elle est définie. En production l'API ne
définit pas cette variable et le comportement reste celui du temps
réel.

Limite connue : cette stratégie marche tant que les *events* de l'index
ne deviennent pas tous obsolètes par rapport à `2026-06-02`. Un rebuild
de l'index avec un dataset Open Agenda futur invaliderait le jeu de
test (les events annotés disparaitraient ou changeraient). Pour un
projet pérenne, il faudrait soit reformuler les questions en dates
absolues, soit conserver une copie figée de l'index. Hors scope POC.

### LLM utilisé : Mistral via API

Initialement, le POC partait sur Mistral-small en local via Ollama
(décision « offline, pas d'API key »). L'Epic 6.3 a révélé deux blocages
sur cette voie pour l'évaluation :

1. **Hallucinations à l'extraction self-querying** sur des questions
   factuelles (ex. q01 « Quand a lieu X ? » → fenêtre temporelle
   fabriquée).
2. **JSON malformé** sur les prompts internes de Ragas
   (`faithfulness`, `answer_relevancy`, etc.) — ~40 % des jobs Ragas
   échouaient en `RagasOutputParserException`.

Bascule sur Mistral via API cloud (`langchain-mistralai`), modèle
`mistral-medium-3.5` par défaut. Choix retenu après benchmark des trois
modèles disponibles sur le tier gratuit :

| Modèle | req/min | tokens/min | Latence /ask | Notes |
|---|---:|---:|---:|---|
| `mistral-large-latest` | 4 | 250k | élevée | Bloquant pour Ragas (~5 appels × 30 Q = 150 req → 38 min minimum) |
| `mistral-medium-latest` (= 2508) | 23 | 356k | ~4 s | Confortable mais latence plus haute |
| **`mistral-medium-3.5`** | **50** | 25k | ~2 s | Retenu : plus de débit, latence /ask plus faible |

Le plafond tokens-minute de 3.5 (25k) est plus serré, mais sérialiser
Ragas (`max_workers=1`) suffit à rester dans la fenêtre. La démo Docker
finale n'est donc plus offline — limitation assumée et documentée dans
le rapport technique (section perspectives).

Fallback `LLM_PROVIDER=ollama` toujours câblé pour des runs hors-ligne,
au prix des NaN évoqués plus haut.

### Trois patches d'infrastructure dans `src/rag/llm.py`

L'intégration `langchain-mistralai` + `Ragas` + tier gratuit nécessite
trois monkey-patches que `_build_mistral()` installe à l'instanciation
du LLM. Idempotents, sans effet en `LLM_PROVIDER=ollama`.

1. **Retry custom sur HTTP 429/5xx**
   (`_install_mistral_retry_on_429`). Par défaut `ChatMistralAI` ne
   retry que les erreurs réseau pures (`httpx.RequestError`,
   `httpx.StreamError`) ; un dépassement de quota lève un
   `HTTPStatusError` que rien n'attrape, et Ragas tombe sur 100 %
   d'échecs au moindre burst. On enrichit le décorateur tenacity pour
   inclure les statuts transients {429, 500, 502, 503, 504}, avec
   backoff exponentiel à jitter (1 s → 16 s plafonné) et 8 tentatives.

2. **Strip des fences markdown autour du JSON**
   (`_patch_mistral_strip_fences`). `mistral-medium-3.5` enveloppe
   systématiquement ses sorties JSON dans ` ```json … ``` ` malgré les
   instructions explicites du prompt Ragas. Le parser Pydantic reçoit
   le markdown brut, échoue, et la métrique passe à NaN. On patche
   `_generate` / `_agenerate` au niveau instance pour stripper les
   fences quand elles enveloppent tout le `content`. Effet observé :
   `faithfulness` débloqué.

3. **Aggrégation récursive de `token_usage`**
   (`_install_combine_llm_outputs_patch`). Bug réel de `langchain-
   mistralai` : `_combine_llm_outputs` fait `overall[k] += v` sur les
   compteurs de tokens. Mistral renvoie maintenant des sous-objets
   imbriqués (`prompt_tokens_details`, `completion_tokens_details`),
   qui plantent l'addition (`TypeError: unsupported operand type(s)
   for +=: 'dict' and 'dict'`). Cette voie est seulement empruntée
   quand Ragas appelle `agenerate_prompt([prompt × n])` pour produire
   plusieurs générations en batch — c'est le cas spécifique
   d'`answer_relevancy` (`strictness=3`). On remplace la méthode par
   une variante qui descend récursivement dans les dicts. Effet
   observé : `answer_relevancy` débloqué.

Bug `langchain-mistralai` officiellement déposable upstream, mais le
patch local règle 100 % du cas sans dépendre du calendrier des fixes
amont.

### Concurrence et timeouts Ragas

`RunConfig(timeout=300, max_workers=1, max_retries=3)` pour les deux
providers (Mistral et Ollama). La sérialisation n'est pas un caprice :
en mode 4 workers, on a observé que les rate-limit 429 et les
« Service tier capacity exceeded » (code 3505, transient côté infra
Mistral) se cumulaient en burst et provoquaient des TimeoutError
malgré le retry. En série on absorbe naturellement les pauses sans
faire grimper l'horloge Ragas.

Timeout 300 s : large marge pour qu'un job lent (jusqu'à 8 retries
httpx avec backoff exponentiel) termine avant que Ragas n'expire.

### Check OOD séparé

Ragas ne sait pas évaluer un refus. Les 3 questions `out_of_domain`
sont donc retirées du calcul des 4 métriques principales et scorées
par une regex maison sur la réponse :

```python
re.compile(
    r"je ne peux r[ée]pondre qu['’]?\s*[àa]\s*des questions sur les "
    r"[ée]v[ée]nements culturels",
    re.IGNORECASE,
)
```

Cohérent avec la règle 1 du prompt système (`src/rag/chain.py`).
Score reporté à part dans `summary.json` (`n_passed / n`).

## Exécution

### Run complet (baseline du rapport)

```bash
uv run python evaluation/evaluate_rag.py
```

Sortie : `evaluation/results/run_<YYYYMMDD_HHMMSS>/per_question.csv` +
`summary.json`. Durée typique : ~30-35 min sur la machine de dev,
dominée par les attentes Ragas (les rate-limit Mistral imposent ~3-5
appels parallèles maximum sur le tier gratuit, et chaque question
émet ~5 appels judge).

### Boucle de dev rapide

```bash
uv run python evaluation/evaluate_rag.py --sample 2     # tirage aléatoire seed=42
uv run python evaluation/evaluate_rag.py --skip-ragas   # RAG seul, pas de judge
```

### Variables d'environnement reconnues

| Variable | Effet | Défaut |
|---|---|---|
| `EVAL_FROZEN_DATE` | Date système figée pour l'extracteur self-querying | `2026-06-02` |
| `LLM_PROVIDER` | `mistral` (défaut) ou `ollama` | `mistral` |
| `MISTRAL_API_KEY` | Clé API Mistral. Obligatoire en provider mistral. | — |
| `MISTRAL_MODEL` | Override du modèle | `mistral-medium-3.5` |

## Baseline du 2026-06-02

Run de référence : `evaluation/results/run_20260602_230903/`
(commit `4de4d46`, `mistral-medium-3.5`, frozen `2026-06-02`).

### Scores globaux in-domain (n = 27)

| Métrique | Score | Lecture |
|---|---:|---|
| `faithfulness` | **0.39** | Les réponses sont partiellement ancrées dans les contextes — hallucinations fréquentes quand le retriever ne ramène pas la bonne info |
| `answer_relevancy` | **0.73** | Les réponses traitent bien le sujet posé, même quand elles ne contiennent pas la bonne info |
| `context_precision` | **0.22** | Le retriever ramène beaucoup de bruit |
| `context_recall` | **0.28** | Et manque souvent les events attendus |

### Scores par catégorie

| Catégorie | n | F | AR | CP | CR |
|---|---:|---:|---:|---:|---:|
| `factual` | 6 | **0.77** | 0.55 | **0.67** | **0.61** |
| `filter_geo` | 7 | 0.27 | 0.79 | 0.29 | 0.24 |
| `filter_temporal` | 7 | 0.29 | 0.76 | 0.00 | 0.18 |
| `exploratory` | 7 | 0.30 | **0.80** | 0.00 | 0.14 |
| `out_of_domain` | 3 | — | — | — | — |

Out-of-domain : **2 / 3 passés**. q28 et q29 répondent correctement avec
la formulation attendue ; q30 (« rédige-moi un CV ») a bien refusé,
mais avec la formulation du fallback « pas d'événement correspondant »
(règle 2 du prompt) au lieu de la garde hors-domaine (règle 1). C'est
un faux négatif du check booléen plus qu'un échec du système.

### Lecture par catégorie

**`factual` — meilleurs scores partout.** Sur 6 questions, le retriever
trouve 4 fois l'event annoté en top-10 et le LLM le cite fidèlement.
Les deux échecs (q03 « Exposition Pauline Deltour », q04 « DRUGSTORE
MALONE — Jocelyn Mienniel Solo ») partagent un pattern : event avec un
**nom propre rare** noyé dans une description longue et littéraire.
MiniLM ranke ces chunks loin (distance L² 13.5 pour la cible contre
9.8 pour le top-1) — finding confirmé par une investigation manuelle
en Epic 6.3. Le LLM répond alors depuis d'autres events thématiquement
proches (autre concert DRUGSTORE MALONE, autres expos), d'où
`faithfulness` correct sur certains et `answer_relevancy` correct
partout, mais `context_precision` à 0 quand il fallait l'event précis.

**`filter_temporal` — talon d'Achille.** `context_precision = 0` sur 5
des 7 questions. Le pre-filter date fonctionne mécaniquement (les
filtres extraits sont corrects), mais sur une fenêtre comme « juin
2026 » ou « cet été 2026 », beaucoup d'events passent le filtre, et la
similarité sémantique ramène alors d'autres events que ceux annotés.
Ce n'est pas un bug — c'est une limite du dataset annoté : on a
annoté **un** event par question, mais la question matche en réalité
des dizaines voire des centaines d'events tous légitimes. Le score
faible reflète plus la difficulté du benchmark que la qualité du
retrieval.

**`filter_geo` — moyen, pollué par le même phénomène.** Sur les villes
rares (Reims, Charleville-Mézières), le pre-filter LUT fait son travail
et la précision retrieval grimpe (q08 = CP 1.0). Sur les grandes villes
(Lyon, Bordeaux), la sémantique est moins discriminante et les events
annotés se font noyer. Cas particulier q12 (« Vercors → Lans-en-Vercors ») :
le retriever ne fait pas le mapping toponymique implicite, tous les
scores à 0.

**`exploratory` — scores attendus.** `answer_relevancy = 0.80` le plus
haut : Mistral réussit à donner une réponse cohérente même quand le
contexte est large. Mais `context_precision = 0` et `context_recall =
0.14` parce qu'il y a *plein* de bonnes réponses à « quelque chose
d'insolite à Lyon » — l'event annoté est un parmi beaucoup.

### Analyse de la latence

| Mesure | Valeur |
|---|---:|
| Temps total run | ~33 min |
| Temps RAG (30 questions) | ~6:30 (13.1 s par question en moyenne) |
| Temps Ragas (27 questions, 4 métriques, ~108 jobs LLM) | ~26:30 |

La latence RAG moyenne (13.1 s) est plus élevée que les ~2 s observés
en smoke test isolé : le RAG concourt avec Ragas pour le quota Mistral
et subit aussi les 429. En production sans concurrence, on retombe à
~2-3 s par question.

## Findings priorisés pour l'itération qualité (Epic 6.5)

1. **Retrieval rate les noms propres rares** (q03, q04). Symptôme : la
   distance L² au chunk cible est plus élevée que celle au top-1
   non-pertinent. Hypothèses : remplacer MiniLM par un modèle
   d'embeddings plus précis sur les noms propres (au prix de la
   vitesse), ou ajouter un rerank sur les top-30 chunks.
2. **Pre-filter date trop laxiste sur les fenêtres larges** (q16-q20).
   Pas un bug du pre-filter mais une limite du dataset (un seul event
   annoté par question pour des fenêtres temporelles qui en
   contiennent des dizaines). Améliorer en annotant plusieurs events
   par question temporelle, ou en formulant des questions temporelles
   plus précises.
3. **Pas de mapping toponymique implicite** (q12 Vercors →
   Lans-en-Vercors). Le LLM extracteur ne propose pas la ville quand
   la question parle d'un massif/parc. Ajout possible : table de
   correspondance région naturelle → villes principales dans le
   prompt extracteur.
4. **Check OOD trop strict** (q30). Relâcher la regex pour accepter
   aussi le fallback « pas d'événement correspondant » comme refus
   valide, ou créer une seconde regex pour ce cas et compter les deux
   comme `passed`.
5. **Faithfulness faible (0.39)**. Renforcer le prompt système pour
   pousser Mistral à dire « pas d'info » quand le contexte ne contient
   pas la réponse au lieu d'inférer depuis ses connaissances.

Ces 5 findings constituent le backlog d'Epic 6.5 (itération qualité,
boucler 2-3 fois et garder une trace des runs).

## Tests

Pas de tests unitaires sur `evaluate_rag.py` : c'est un script
d'orchestration (chargement dataset → boucle RAG → wrappers Ragas →
écritures fichiers) dont les composantes sont déjà couvertes — RAG par
`tests/test_rag.py`, écritures par les utilitaires standards. Une
exécution réussie en `--sample 2` (~3 min sur Mistral, automatiquement
visible quand le pipeline rend les 4 métriques sans NaN systématique)
suffit comme smoke test avant d'engager un run complet.
