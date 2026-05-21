# Source de données — Open Agenda

Documentation de référence sur les données utilisées par le POC : source,
filtres, schéma des champs retenus, pipeline de nettoyage et chiffres du
dataset final. Produit au fil de l'Epic 2 (tâches P7-2.1 à P7-2.6).

## Dataset retenu

- **Nom** : `evenements-publics-openagenda`
- **Plateforme** : `public.opendatasoft.com` (instance publique Opendatasoft, pas d'authentification)
- **Éditeur** : OpenAgenda
- **Dernière mise à jour du dataset** : **2024-04-08** — *attention : le dataset
  agrège les déclarations des organisateurs, et le snapshot Opendatasoft n'a pas
  été rafraîchi depuis avril 2024. Conséquence : pas d'événements ajoutés depuis,
  mais le dataset contient déjà des événements déclarés dont la date est en 2025
  ou 2026.*
- **Volume total** : **1 126 911** événements
- **Couverture géographique** : majoritairement France métropolitaine
  (1 055 315 events, soit **93,6 %**). Le reste est principalement DOM-TOM et
  quelques événements européens.

## Scope du POC (décision projet)

Volontaire écart à l'énoncé, validé par le professeur (cf. mémoire projet
`project-stack`) :

- **Géographie** : France entière, pas de filtre régional.
- **Temps** : pas de filtre temporel, ni à l'ingestion ni au retrieval. Le
  chatbot peut donc être interrogé sur événements passés comme à venir.

Conséquence : on indexe potentiellement l'intégralité du dataset (après filtrage
qualité), pas un sous-ensemble géographique ou temporel.

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
| `title_fr` | text | titre en français, **systématiquement présent** |
| `description_fr` | text | résumé court, **sans HTML** d'après l'échantillon — null sur ~1,0 % des records |
| `longdescription_fr` | text | description longue, **contient du HTML** (`<p>`, `<br/>`, `<em>`...) — null sur ~10,7 % des records |
| `keywords_fr` | text | mots-clés ; **souvent null** d'après l'échantillon (à confirmer en 2.2) |
| `conditions_fr` | text | conditions de participation, accès, etc. |

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

## Distribution temporelle (firstdate_begin)

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
| Hors France métropolitaine | 71 596 (~6,4 %) | À filtrer si on veut un assistant strictement France métropole |
| HTML dans `longdescription_fr` | Confirmé sur l'échantillon | Strip HTML obligatoire en 2.4 |
| Dates aberrantes | < 1 % | Filtrer events dont l'année tombe hors [2010, 2030] par exemple |
| Doublons | À mesurer en 2.4 sur `uid` et `(title_fr, firstdate_begin, location_name)` | — |

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
null à 100 % dans le snapshot Opendatasoft d'avril 2024.

Caractéristiques retenues :

- Pas de plafond à 10 000 résultats (contrairement à `/records?offset=`).
- Un seul appel HTTP, streamé chunk par chunk côté client.
- Rate-limit **10 000 000 appels/jour** observé (`X-RateLimit-Limit`) →
  aucun risque pour notre usage (1 appel/jour max).
- `select=` réduit à 23 champs sur 56 (cf. partition supra).

### Filtres `where=` appliqués

- `country_fr = "France (Métropole)"` (élimine ~6,4 % hors-périmètre).
- `description_fr IS NOT NULL` (élimine ~1 % de bruit).

Pas de filtre temporel à l'ingestion : le filtrage des dates aberrantes est
reporté en 2.4 pour ne pas se priver d'événements 2025–2027 ni d'historique
récent.

### Résultats du run du 2026-05-21

| Mesure | Valeur |
|---|---:|
| Fichier produit | `data/raw/events_2026-05-21.jsonl` |
| Lignes JSONL | **1 051 298** |
| Taille fichier | **2,16 GB** |
| Durée totale | **12 min 1 s** |
| Champs par enregistrement | 23 (cohérent avec le `select=`) |

Le volume final correspond à 1 055 315 (events France métropole) − ~4 000
(events sans description) = 1 051 298. Cohérent avec les sondages 2.1.

La taille (2,16 GB) est bien plus élevée que mon estimation initiale
(~30 MB) — les `longdescription_fr` HTML pèsent en moyenne **~2 KB par
événement**. Conséquences :

- `data/raw/` est de toute façon gitignored, donc pas de pollution du repo.
- Le cleaning (2.4) devra strip le HTML, ce qui réduira significativement
  la taille (~30–50 % en moins probablement).
- Pour l'indexation FAISS (Epic 3), c'est le **nombre de documents** (1 M)
  qui dimensionne le temps d'embedding, pas la taille brute. À ré-évaluer
  une fois le cleaning fait.

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
   également exposé en *metadata only* dans Epic 3 (pas dans le
   `page_content` embeddé, vu sa quasi-monomodalité — cf. décision P7-2.2).
7. **Champ dérivé `event_year`** extrait de `firstdate_begin`.
8. **Validation** : rejet des événements sans titre, sans description, ou
   d'année hors `[2010, 2030]` (filtre les aberrations type 1900, 23,
   2503, 2032 vues en P7-2.1).

La **déduplication** suit la clé décidée en P7-2.4 :
`(title_fr_lower, firstdate_begin, location_name_lower)`. On garde la
première occurrence ; les ~47 events sans l'un des 3 champs sont conservés
sans comparaison. Cf. `scripts/measure_duplicates.py` pour la mesure de
référence ayant guidé ce choix.

### Résultats du run du 2026-05-21

| Étape | Volume | Δ |
|---|---:|---:|
| Raw d'entrée | 1 051 298 | — |
| Invalides (titre/description/année) | − 2 337 | −0,2 % |
| ↳ dont `year_out_of_range` | − 2 290 | |
| ↳ dont `no_year` | − 47 | |
| Doublons (clé D2) | − 33 266 | −3,2 % |
| **Conservés** | **1 015 695** | **96,6 %** |

| Mesure | Valeur |
|---|---:|
| Fichier produit | `data/processed/events_clean_2026-05-21.jsonl` |
| Taille | **1,65 GB** (vs 2,16 GB raw → −24 % grâce au strip HTML et au drop de `country_fr`) |
| Durée | **4 min 27 s** |
| Champs par event | 24 (23 du raw + `event_year`, `attendancemode` remplacé par `attendance_mode`) |

### Distribution finale

- **`attendance_mode`** : 1 003 796 sur place (98,8 %), 5 929 en ligne (0,6 %),
  5 751 mixte (0,6 %), 219 inconnu.
- **`event_year`** (top 5) : 2024 (162k), 2023 (160k), 2025 (151k),
  2022 (127k), 2026 (100k). Couverture concentrée sur 2022–2026, avec un
  long historique 2010–2021 qui apporte des cas type « Journées du
  Patrimoine récurrentes ».

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
extraction d'année, validation, clé de dédup, et un scénario
bout-en-bout sur mini-fixture qui simule le pipeline complet
(rejet + dédup).

```bash
uv run pytest tests/test_clean.py -v
```

37 tests, ~0,1 s.

## Reproduire l'exploration

```bash
uv run python scripts/explore_openagenda.py
```

Le script affiche schéma + volumétrie + distribution + smoke test
`/exports/json`, et écrit un échantillon de 20 événements bruts dans
`data/raw/sample_events.json` (non versionné — `data/raw/` est gitignored).
Cet échantillon sert de fixture pour développer le nettoyage en 2.4 et son
test en 2.5 sans avoir à retélécharger le dataset complet.
