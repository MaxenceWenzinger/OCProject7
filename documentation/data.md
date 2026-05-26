# Source de données — Open Agenda

Documentation de référence sur les données utilisées par le POC : source,
filtres, schéma des champs retenus, pipeline de nettoyage, stratégie de
chunking et chiffres de l'index FAISS final. Produit au fil des Epics 2
(ingestion et cleaning) et 3 (indexation vectorielle).

## Dataset retenu

- **Nom** : `evenements-publics-openagenda`
- **Plateforme** : `public.opendatasoft.com` (instance publique Opendatasoft, pas d'authentification)
- **Éditeur** : OpenAgenda
- **Métadonnée `modified` du catalog Opendatasoft** : **2024-04-08**.
  *⚠ Cette valeur est trompeuse* : la distribution des `updatedat` côté API
  montre que **161 093 events ont été mis à jour en 2025** et **148 003 en
  2026** — le dataset est donc activement entretenu, contrairement à ce que
  laisse penser le champ `modified` du catalog. On ne sait pas pourquoi
  Opendatasoft ne met pas ce champ à jour (bug ou interprétation
  différente), mais on travaille bien sur un dataset vivant.
- **Volume total** : **1 126 911** événements
- **Couverture géographique** : majoritairement France métropolitaine
  (1 055 315 events, soit **93,6 %**). Le reste est principalement DOM-TOM et
  quelques événements européens.

## Scope du POC (décision projet)

Volontaire écart à l'énoncé, validé par le professeur (cf. mémoire projet
`project-stack`) :

- **Géographie** : France entière, pas de filtre régional.
- **Temps** : on ne garde que les événements dont la dernière occurrence se
  termine en **2025 ou après**. Autrement dit : les événements purement passés
  (terminés en 2024 ou avant) sont écartés. Ce critère est appliqué dans le
  cleaning, sur `lastdate_end` avec fallback sur `firstdate_end`. Pas de
  filtre au retrieval — les événements peuvent avoir commencé en 2024 ou avant
  tant qu'ils sont encore en cours ou à venir.

Conséquence : ~25 % du dataset brut est retenu (les ¾ sont des événements
historiques). Le POC se concentre sur les événements actuels et futurs, ce
qui est la valeur métier prioritaire pour un assistant culturel.

## Champs disponibles (56 au total)

Sélection des champs retenus pour le RAG, regroupés par usage prévisionnel.

### Identité et URL

| Champ | Type | Note |
|---|---|---|
| `uid` | text | identifiant unique → clé pour dédup |
| `slug` | text | slug humain |
| `canonicalurl` | text | URL canonique de l'événement → à exposer dans `sources` de la réponse API |

### Contenu textuel (entrera dans le `page_content` des Documents LangChain)

| Champ | Type | Note |
|---|---|---|
| `title_fr` | text | titre en français, **systématiquement présent** (médiane 41 chars, max 150) |
| `description_fr` | text | résumé court, **sans HTML** — null sur ~1,0 % des records, plafonné à 200 chars par Open Agenda |
| `longdescription_fr` | text | description longue, **contient du HTML** (`<p>`, `<br/>`, `<em>`...) — null sur ~10,7 % des records, médiane 498 chars / max ~10 000 |
| `keywords_fr` | list[text] | mots-clés sous forme de liste ; présent sur ~61 % des events (dataset filtré) |
| `conditions_fr` | text | conditions de participation, accès, etc. ; présent sur ~38 % des events (dataset filtré) |

### Dates (utiles pour metadata + filtrage qualité)

| Champ | Type | Note |
|---|---|---|
| `firstdate_begin` | datetime | début de la première occurrence (ISO 8601 UTC) |
| `firstdate_end` | datetime | fin de la première occurrence |
| `lastdate_begin` | datetime | début de la dernière occurrence |
| `lastdate_end` | datetime | fin de la dernière occurrence |
| `daterange_fr` | text | plage de dates humanisée en français (ex : « du 10 au 15 mai 2025 ») |

### Localisation (metadata pour retrieval géographique futur)

`location_name`, `location_address`, `location_city`, `location_postalcode`,
`location_department`, `location_region`, `location_coordinates`
(geo_point_2d).

`country_fr` est utilisé pour le filtre `where=` côté API mais n'est **pas**
dans le `select=` — il vaut alors par construction `"France (Métropole)"` sur
100 % des lignes (vérifié), inutile de le dupliquer dans chaque enregistrement.

### Public (metadata facultative)

`age_min`, `age_max`, `accessibility_label_fr`, `attendancemode`.

Le champ `category` du schéma Opendatasoft est **systématiquement null** sur
tout le dataset (1 051 298 lignes vérifiées, 0 valeur non-null). Il a été
retiré du `select=` du fetch et du dataset clean.

### Champs ignorés au POC

Tout ce qui concerne l'image (`image`, `thumbnail`, `originalimage`,
`imagecredits`), les contributeurs (`contributor_*`), l'agenda d'origine
(`originagenda_*`), les détails de lieu non textuels (`location_image`,
`location_phone`, etc.) et les liens additionnels (`links`,
`onlineaccesslink`). Ces champs peuvent rester dans le `raw` mais ne seront
pas embarqués dans l'index.

## Distribution temporelle du raw (firstdate_begin)

Cette distribution porte sur le dataset **brut** (1 051 298 events) avant
l'application du filtre temporel du cleaning. Elle reste utile pour montrer
la couverture historique disponible côté Open Agenda.

| Année | Nombre d'événements |
|---|---:|
| `null` ou anomalies (1900, 23, 2503...) | < 500 (à filtrer) |
| 2016 | 39 463 |
| 2017 | 31 110 |
| 2018 | 56 527 |
| 2019 | 77 432 |
| 2020 | 58 181 |
| 2021 | 65 910 |
| 2022 | 137 939 |
| 2023 | 180 254 |
| 2024 | 191 788 |
| 2025 | 166 661 |
| 2026 | 108 873 |
| 2027 | 158 |
| 2028+ (anomalies) | ~2 300 |

**Lecture** : le dataset est riche sur 2018–2026 (pic 2023–2024). Il contient bien
des événements futurs (2025–2026, voire 2027), ce qui rend le chatbot
pertinent sans tricher avec la date système. Les années aberrantes (1900, 23,
2503, 2032) totalisent quelques milliers d'entrées à nettoyer.

## Qualité des données

| Indicateur | Valeur | Conséquence |
|---|---:|---|
| `description_fr` NULL | 11 515 (~1,0 %) | Marginal — events à supprimer |
| `longdescription_fr` NULL | 121 037 (~10,7 %) | On utilise `description_fr` en fallback |
| Hors France métropolitaine | 71 596 (~6,4 %) | Filtré côté API via `where=` |
| HTML dans `longdescription_fr` | Confirmé sur tout le dataset | Strip HTML appliqué au cleaning |
| Dates aberrantes (1900, 2503, ...) | ~2 300 (< 1 %) | Filtré au cleaning (fenêtre [2010, 2030]) |
| Doublons stricts par `uid` | 0 | OpenAgenda garantit l'unicité |
| Doublons sémantiques par `(title, firstdate_begin, location_name)` | 33 643 (~3,2 % du raw) | Filtré au cleaning, réduit à ~5 400 après filtre temporel |
| `uid` unique sur 1 051 298 records | confirmé | RAS |

## Téléchargement effectué (P7-2.2)

**Endpoint retenu** : `/exports/jsonl` d'Opendatasoft (et non `/exports/json`
comme initialement envisagé). `jsonl` retourne un objet JSON par ligne, ce
qui permet un vrai streaming (pas besoin de charger tout le tableau en RAM)
et est exactement le format demandé par le plan de travail
(`events_<date>.jsonl`).

```
GET https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/
    evenements-publics-openagenda/exports/jsonl
    ?select=uid,slug,canonicalurl,title_fr,description_fr,longdescription_fr,
            keywords_fr,conditions_fr,firstdate_begin,firstdate_end,
            lastdate_begin,lastdate_end,location_name,location_address,
            location_city,location_postalcode,location_department,
            location_region,location_coordinates,age_min,age_max,
            accessibility_label_fr,attendancemode
    &where=country_fr="France (Métropole)" AND description_fr IS NOT NULL
```

`country_fr` et `category` sont volontairement absents du `select=` :
le premier est constant par construction (filtre `where=`), le second est
null à 100 % dans le dataset (vérifié sur les 1 051 298 lignes téléchargées).

Caractéristiques retenues :

- Pas de plafond à 10 000 résultats (contrairement à `/records?offset=`).
- Un seul appel HTTP, streamé chunk par chunk côté client.
- Rate-limit **10 000 000 appels/jour** observé (`X-RateLimit-Limit`) →
  aucun risque pour notre usage (1 appel/jour max).
- `select=` réduit à 23 champs sur 56 (cf. partition supra).

### Filtres `where=` appliqués

- `country_fr = "France (Métropole)"` (élimine ~6,4 % hors-périmètre).
- `description_fr IS NOT NULL` (élimine ~1 % de bruit).

Pas de filtre temporel à l'ingestion : le filtrage temporel (événements
purement passés) et celui des dates aberrantes sont reportés au cleaning.

### Résultats du run du 2026-05-21

| Mesure | Valeur |
|---|---:|
| Fichier produit | `data/raw/events_2026-05-21.jsonl` |
| Lignes JSONL | **1 051 298** |
| Taille fichier | **2,16 GB** |
| Durée totale | **12 min 1 s** |
| Champs par enregistrement | 23 (cohérent avec le `select=`) |

Le volume final correspond à 1 055 315 (events France métropole) − ~4 000
(events sans description) = 1 051 298. Cohérent avec les sondages
d'exploration.

La taille (2,16 GB) est plus élevée que l'estimation initiale (~30 MB) :
les `longdescription_fr` HTML pèsent en moyenne **~2 KB par événement**.
Note pratique : `data/raw/` est gitignored, donc pas de pollution du repo.
Le strip HTML appliqué au cleaning réduit la taille de ~24 % avant même
le filtre temporel (cf. section suivante).

### Robustesse

Le script utilise `urllib3.Retry` avec backoff exponentiel
(0s → 2s → 4s → 8s → 16s) sur les erreurs réseau et HTTP 429/5xx, et
écrit dans un fichier `.tmp` qu'il renomme atomiquement en fin de
stream — pas de risque de laisser un fichier partiel sous un nom
« propre » en cas de coupure.

### Reproduire le téléchargement

```bash
uv run python scripts/fetch_openagenda.py
```

Le fichier de sortie est nommé `events_<YYYY-MM-DD>.jsonl` (date du run).
Options : `--out-dir <chemin>` et `--name <nom>`.

## Nettoyage (P7-2.4) — pipeline et résultats

Le pipeline de nettoyage (`src/data/clean.py`, wrapper
`scripts/clean_events.py`) applique 8 étapes pures à chaque événement :

1. **Strip HTML** sur `longdescription_fr` et `conditions_fr` via
   `BeautifulSoup` (`html.parser` natif, pas de dépendance lxml ajoutée).
2. **Décodage des entités HTML** (`&amp;`, `&nbsp;`, `&eacute;`, ...) via
   `html.unescape`.
3. **Normalisation des espaces** sur tous les champs texte :
   `\s+` → un espace, trim. Une chaîne vide après nettoyage devient `None`
   (la distinction « pas de donnée » vs « donnée vide » est ainsi préservée).
4. **Champs de type `list` → string** : `keywords_fr` et
   `accessibility_label_fr` arrivent sous forme de listes dans l'export
   Opendatasoft, on les joint avec `", "`.
5. **Suppression des surrogates Unicode isolés** (`U+D800–U+DFFF`) :
   caractères mal encodés observés dans le dataset réel qui plantent
   l'écriture UTF-8 si laissés.
6. **Parsing de `attendancemode`** (JSON imbriqué) → champ dérivé
   `attendance_mode: "sur_place" | "en_ligne" | "mixte" | None`. Champ
   exposé en *metadata only* à l'indexation, pas dans le `page_content`
   embeddé : 98,8 % des events sont « sur_place », inclure cette valeur
   quasi-constante dans le texte indexé diluerait les embeddings sans
   apporter de signal discriminant.
7. **Champ dérivé `event_year`** extrait de `firstdate_begin`.
8. **Validation** : rejet des événements sans titre, sans description,
   d'année hors `[2010, 2030]` (filtre les aberrations type 1900, 23,
   2503, 2032), ou dont la dernière occurrence se termine avant 2025
   (événements purement passés). La règle temporelle utilise `lastdate_end`
   avec fallback sur `firstdate_end` ; les events sans aucune des deux
   sont également rejetés.

La **déduplication** utilise la clé
`(title_fr_lower, firstdate_begin, location_name_lower)`. On garde la
première occurrence ; les ~47 events sans l'un des 3 champs sont conservés
sans comparaison. `scripts/measure_duplicates.py` documente la mesure de
référence (4 stratégies de clé comparées) qui a guidé ce choix.

### Résultats du run du 2026-05-21

| Étape | Volume | Δ |
|---|---:|---:|
| Raw d'entrée | 1 051 298 | — |
| Invalides | − 792 972 | −75,4 % |
| ↳ dont `event_too_old` (terminés avant 2025) | − 790 629 | |
| ↳ dont `year_out_of_range` | − 2 290 | |
| ↳ dont `no_year` | − 47 | |
| ↳ dont `no_end_date` (les 2 dates de fin null) | − 6 | |
| Doublons (clé D2) | − 5 425 | −0,5 % |
| **Conservés** | **252 901** | **24,1 %** |

| Mesure | Valeur |
|---|---:|
| Fichier produit | `data/processed/events_clean_2026-05-21.jsonl` |
| Taille | **401 MB** (vs 2,16 GB raw → −81 % grâce au strip HTML, au drop de `country_fr` et au filtre temporel) |
| Durée | **4 min 6 s** |
| Champs par event | 24 (23 du raw + `event_year`, `attendancemode` remplacé par `attendance_mode`) |

### Distribution finale

- **`event_year`** (date de début, top 6) : 2025 (151k), 2026 (100k),
  2024 (666), 2023 (144), 2027 (131), 2022 (86). Les events <2025 conservés
  sont ceux dont une occurrence se prolonge en 2025+ (récurrents, expos
  longues, festivals pluri-annuels).
- **`last_relevant_date`** (date de fin, top 5) : 2025 (151k), 2026 (102k),
  2027 (195), 2028 (20), >2030 (~20). Quelques events s'étalent jusqu'en
  2049-2052 (expositions de longue durée).

### Reproduire le cleaning

```bash
uv run python scripts/clean_events.py
```

Par défaut prend le `events_*.jsonl` le plus récent dans `data/raw/` et
écrit dans `data/processed/events_clean_<YYYY-MM-DD>.jsonl`. Options
`--input` / `--output` disponibles.

## Tests (P7-2.5)

`tests/test_clean.py` couvre les comportements de `src.data.clean` :
strip HTML + entités, normalisation des espaces, gestion des listes
(`keywords_fr`), surrogates Unicode, parsing `attendancemode`,
extraction d'année, validation, **filtre temporel** (`lastdate_end`
avec fallback sur `firstdate_end`), clé de dédup, et un scénario
bout-en-bout sur mini-fixture qui simule le pipeline complet
(rejet + dédup).

```bash
uv run pytest tests/test_clean.py -v
```

48 tests, ~0,1 s.

## Indexation FAISS (Epic 3) — stratégie et chiffres

### Choix du modèle d'embedding

- **Modèle retenu** : `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - Multilingue (fr/en/...), 384 dimensions, fenêtre max **128 tokens**
  - Léger (~118 MB), rapide à l'inférence (~10 ms par requête sur CPU)
- **Modèle écarté** : `intfloat/multilingual-e5-base`
  - Plus gros (~280 MB), 768 dim, fenêtre 512 tokens
  - Benchmark CPU : 10,6 events/s vs 96,6 pour MiniLM (~9× plus lent)
  - Aurait été nécessaire **sans** parent-child chunking pour éviter la
    troncature des 2,5 % d'events qui dépassent 512 tokens. Le parent-child
    rend MiniLM viable car chaque chunk fait <128 tokens par construction.

Les deux modèles ont été benchmarkés via `scripts/benchmark_embeddings.py`
sur 100 events. Trace conservée pour reproduction.

### Stratégie de chunking : parent-child

Au lieu d'indexer un Document par event, on découpe chaque event en
**N chunks de ≤ 120 tokens MiniLM** (marge de 8 tokens sous la limite 128
pour les tokens spéciaux `[CLS]`/`[SEP]`) avec un **recouvrement de
24 tokens** (20 %).

À l'inférence :

- la similarity search retourne des chunks
- on dédoublonne par `parent_uid` (un même event peut produire plusieurs
  chunks proches d'une question)
- on récupère le Document **parent** complet (page_content + metadata
  intacts) via un `parent_store: dict[uid → Document]` chargé en RAM
- on passe les parents au LLM en génération

Bénéfices :

- retrieval précis sur les détails enfouis dans la longdescription
  (qui seraient invisibles avec un seul embedding par event tronqué à 128
  tokens)
- MiniLM utilisable malgré sa fenêtre courte (cf. ci-dessus)
- build relativement rapide (1h42 vs 6h30 pour Option A + e5-base)

Coût : ~30 lignes de code de jointure parent supplémentaires dans la chaîne
RAG (Epic 4), index plus gros (~1,5 GB total sur disque).

### Format des Documents

**`page_content`** d'un event (`build_page_content` dans
`src/indexing/build_documents.py`) :

```
{title_fr}

{description_fr}

{longdescription_fr}

Mots-clés : {keywords_fr}

Conditions : {conditions_fr}
```

Champs absents omis. Pas de préfixe répété dans les chunks : titre et
description courte se retrouvent automatiquement dans le 1er chunk par
construction.

**`metadata`** d'un Document parent (10 champs) :

| Clé | Source | Usage prévu |
|---|---|---|
| `uid` | `uid` du clean | identification / dédup |
| `title` | `title_fr` | affichage dans les sources de la réponse API |
| `url` | `canonicalurl` | lien vers la page Open Agenda d'origine |
| `first_date` | `firstdate_begin` | affichage + filtrage retrieval futur |
| `last_date` | `lastdate_end` (fallback `firstdate_end`) | affichage + filtrage |
| `location_name` | `location_name` | affichage |
| `location_city` | `location_city` | affichage + filtrage géo |
| `location_region` | `location_region` | filtrage géo régional |
| `attendance_mode` | `attendance_mode` | filtrage « en ligne » éventuel |
| `event_year` | dérivé | filtrage par année |

Les champs textuels du `page_content` (description, longdescription,
keywords, conditions) ne sont pas dupliqués en metadata pour limiter la
taille de l'index.

**`metadata`** d'un Document chunk : les 10 champs du parent + deux ajouts :

- `parent_uid` : duplique `uid`, rendu explicite pour le code de jointure
- `chunk_index` : position 0-indexée dans le parent (debug)

### Résultats du build du 2026-05-26

Pipeline complet `scripts/build_index.py` : streaming
`events_clean_*.jsonl` → chunking via tokenizer MiniLM → embedding par
batchs de 5 000 → sauvegarde FAISS + pickle parent_store. Écriture
atomique via `.tmp/` puis swap.

| Mesure | Valeur |
|---|---:|
| Events parents indexés | **252 901** |
| Chunks vectorisés | **579 652** |
| Ratio chunks / event | **2,29** (médiane 2, max 11) |
| Débit moyen | 95 chunks/s sur CPU |
| Durée totale du build | **1h 42min** |
| `data/index/index.faiss` | 890 MB |
| `data/index/index.pkl` (chunks LangChain) | 395 MB |
| `data/index/parent_store.pkl` (mapping uid → parent) | 289 MB |
| **Total disque** | **~1,5 GB** |

### Performances à l'inférence (mesurées)

- **Chargement** au démarrage de l'API : ~15 s (modèle MiniLM + index FAISS
  + parent_store)
- **Latence retrieval** : ~50 ms par requête (similarity search sur 580k
  vecteurs)

### Sanity check qualitatif (5 requêtes type)

Les 5 requêtes suivantes ont été lancées sur l'index produit, top-K=15
chunks dédupliqués par `parent_uid` → 5 parents :

- « concert de jazz à Paris » → 5 concerts de jazz retournés (mais pas
  filtré par ville — le tri géo est laissé pour Epic 4 via metadata)
- « exposition de peinture contemporaine » → 5 expositions de peinture
- « spectacle pour enfants pendant les vacances » → résultats mixtes,
  signal qu'un filtrage `age_min/age_max` en metadata pourrait aider
- « visite guidée du château de Versailles » → 3 events Versailles + 2
  events thématiques liés (« Si Versailles m'était conté », « Versailles
  de Charles V »)
- « festival de musique en plein air été 2026 » → 5 festivals de musique
  estivaux (mélange 2025/2026, filtrage temporel à faire côté metadata)

Conclusion : le retrieval sémantique fait son travail. Le filtrage par
metadata (ville, date) reste à câbler dans la chaîne RAG (Epic 4) pour
serrer la précision sur les questions géographiquement ou temporellement
explicites.

### Reproduire le build

```bash
uv run python scripts/build_index.py
uv run python scripts/build_index.py --limit 1000   # test rapide ~30 s
```

### Tests

`tests/test_build_documents.py` couvre :

- `build_page_content` : concaténation, omission des champs absents,
  ordre des blocs
- `build_metadata` : 10 clés exactes, mapping des champs, fallback
  `last_date`, pas de fuite de champs textuels ou exclus
- `event_to_chunks` : 1 chunk pour les courts events, N chunks
  glissants pour les longs, recouvrement effectif entre chunks
  consécutifs, propagation de la metadata parent + `parent_uid` +
  `chunk_index`, aucun chunk au-delà de `chunk_size`

Utilise un faux tokenizer `_CharTokenizer` (1 char = 1 token) pour
éviter de charger MiniLM pendant les tests unitaires.

```bash
uv run pytest tests/test_build_documents.py -v
```

24 tests, ~0,2 s.

