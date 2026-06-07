# Évaluation du RAG — méthodologie et résultats

Documentation de référence sur la stratégie d'évaluation du POC : jeu de
test annoté, métriques Ragas, choix d'implémentation, lecture de la
baseline produite et findings qualité priorisés.

## Vue d'ensemble

L'évaluation repose sur trois briques :

1. **Un jeu de test annoté à la main** : `evaluation/qa_dataset.jsonl`,
   7 questions/réponses (6 in-domain + 1 hors-domaine) choisies pour
   stresser chaque facette du système (factuel, filtre géo, filtre
   temporel, exploratoire, hors-domaine).
2. **Ragas 0.4.3** : framework standard pour évaluer un RAG, calcule
   4 métriques LLM-as-judge sur les questions in-domain.
3. **Un check booléen maison** pour la question hors-domaine : Ragas
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

7 questions construites à partir d'événements réels de l'index, choisies
manuellement pour couvrir les facettes du système : ville rare vs ville
majeure (stress du pre-filter), filtre temporel strict vs relatif,
sémantique floue, et un refus hors-domaine. Le volume est volontairement
réduit : pour ce POC, un petit jeu diversifié suffit à diagnostiquer le
comportement du RAG — 30 questions auraient été du sur-dimensionnement
(décision validée avec l'encadrant).

### Schéma (1 ligne JSON par question)

```json
{
  "id": "q01",
  "question": "Quand a lieu le Marathon de Bordeaux ?",
  "ground_truth": "Le Marathon de Bordeaux a lieu le dimanche 8 novembre 2026 à Bordeaux.",
  "contexts_uids": ["80404875"],
  "expected_filter": {"city": "Bordeaux", "date_after": "2026-06-02"}
}
```

Le schéma standard Ragas ne consomme que `question` et `ground_truth`
(plus les contextes réellement récupérés au runtime, cf. plus bas). Les
deux champs restants sont des **métadonnées d'inspection humaine**, jamais
envoyées à Ragas :

- `contexts_uids` : les UIDs des events qu'on **espère** voir remontés.
  Sert à comparer à l'œil (dans le CSV) la cible vs les `retrieved_uids`.
  À ne pas confondre avec le champ `contexts` de Ragas, qui contient le
  **texte** des documents réellement récupérés — d'où le nommage explicite.
- `expected_filter` : le filtre qu'on attend du pré-filtre self-querying.
  Permet de vérifier a posteriori que l'extraction `{city, region,
  date_after, date_before}` correspond bien à l'intention de la question.
  Une année mentionnée n'a pas de champ dédié : elle est attendue sous
  forme de bornes (`YYYY-01-01` / `YYYY-12-31`).

### Les 7 questions

| id | Facette stressée |
|---|---|
| q01 | Factuel : event nommé, date précise, ville explicite (Marathon de Bordeaux). |
| q02 | Filtre géo, **ville rare** (Reims, ~980 events) — stress du pré-filtre LUT. |
| q03 | Filtre géo : ville **+ année** combinés (Marseille 2026). |
| q04 | Filtre temporel : fenêtre stricte sans géo (marchés de Noël déc. 2025). |
| q05 | Filtre temporel : expression relative résolue (« ce week-end 6-7 juin 2026 »), today-aware. |
| q06 | Exploratoire : sémantique floue + ville majeure (jazz à Paris). |
| q07 | Hors-domaine : garde du prompt système (météo → refus attendu). |

### Détection du hors-domaine

Plus de champ `category` : la **seule** question hors-domaine se reconnaît
à son `contexts_uids` vide (`[]`). Sémantiquement, « aucun event cible »
signifie « la bonne réponse est un refus ». Le helper `is_out_of_domain`
(`evaluate_rag.py`) centralise ce critère ; il aiguille q28 vers le check
regex et l'exclut du calcul Ragas. Invariant : toutes les questions
in-domain ont au moins un UID dans `contexts_uids`.

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

Ragas ne sait pas évaluer un refus. C'est un choix de conception, pas un
détail : sur une question hors-domaine, le RAG ne récupère aucun event
(`contexts` vides), ce qui produit des **NaN** sur `faithfulness`,
`context_precision` et `context_recall` ; et `answer_relevancy`
**pénalise** un refus, alors que le refus est précisément le comportement
souhaité (son score serait donc inversé par rapport à l'intention).

La question hors-domaine — identifiée par son `contexts_uids` vide — est
donc retirée du calcul des 4 métriques principales et scorée par une regex
maison sur la réponse :

```python
re.compile(
    r"je ne peux r[ée]pondre qu['’]?\s*[àa]\s*des questions sur les "
    r"[ée]v[ée]nements culturels",
    re.IGNORECASE,
)
```

Cohérent avec la règle 1 du prompt système (`src/rag/chain.py`).
Score reporté à part dans `summary.json`, sous la clé `out_of_domain`
(`{n, n_passed, pass_rate}`).

## Exécution

### Run complet

```bash
uv run python evaluation/evaluate_rag.py
```

Sortie : `evaluation/results/run_<YYYYMMDD_HHMMSS>/per_question.csv` +
`summary.json`. La durée est dominée par les attentes Ragas (les
rate-limit Mistral imposent une exécution sérialisée sur le tier
gratuit, et chaque question in-domain émet ~5 appels judge).

### Boucle de dev rapide

```bash
uv run python evaluation/evaluate_rag.py --sample 3     # tirage aléatoire seed=42
uv run python evaluation/evaluate_rag.py --skip-ragas   # RAG seul, pas de judge
```

### Variables d'environnement reconnues

| Variable | Effet | Défaut |
|---|---|---|
| `EVAL_FROZEN_DATE` | Date système figée pour l'extracteur self-querying | `2026-06-02` |
| `LLM_PROVIDER` | `mistral` (défaut) ou `ollama` | `mistral` |
| `MISTRAL_API_KEY` | Clé API Mistral. Obligatoire en provider mistral. | — |
| `MISTRAL_MODEL` | Override du modèle | `mistral-medium-3.5` |

## Baseline et findings

> **À produire.** Le jeu de test a été réduit de 30 à 7 questions et son
> schéma simplifié (suppression de `category`/`notes`, renommage
> `expected_contexts` → `contexts_uids`). L'ancienne baseline du 2026-06-02
> (30 questions, 5 catégories) ne s'applique plus et a été retirée.
>
> La nouvelle baseline et les findings priorisés pour l'itération qualité
> (Epic 6.5) seront produits après le **premier run Ragas sur le dataset 7 Q**.
> Cette section sera alors renseignée avec : les scores globaux in-domain
> (n = 6), le résultat du check OOD (n = 1), une lecture question par
> question, et les pistes d'amélioration identifiées.

## Tests

Pas de tests unitaires sur `evaluate_rag.py` : c'est un script
d'orchestration (chargement dataset → boucle RAG → wrappers Ragas →
écritures fichiers) dont les composantes sont déjà couvertes — RAG par
`tests/test_rag.py`, écritures par les utilitaires standards. Une
exécution réussie en `--sample 3` (visible quand le pipeline rend les
4 métriques sans NaN systématique) suffit comme smoke test avant
d'engager un run complet.
