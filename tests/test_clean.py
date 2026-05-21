"""Tests unitaires du pipeline de cleaning.

Couvre les comportements clés de `src.data.clean` :
  - strip HTML + entités sur longdescription/conditions
  - normalisation des espaces et préservation des None
  - parsing du champ JSON imbriqué `attendancemode`
  - extraction de l'année et validation (rejet d'entrées invalides)
  - clé de déduplication

Les fixtures sont des dicts minimaux écrits inline plutôt que d'inclure le
fichier `data/raw/sample_events.json` (gros, non versionné). Ça garde les
tests rapides, lisibles, et indépendants de l'état du disque.
"""

from __future__ import annotations

import pytest

from src.data.clean import (
    clean_event,
    dedup_key,
    extract_year,
    is_valid,
    normalize_whitespace,
    parse_attendance_mode,
    strip_html,
)


# ---------------------------------------------------------------------------
# strip_html
# ---------------------------------------------------------------------------

class TestStripHtml:
    def test_removes_basic_tags(self):
        assert strip_html("<p>Bonjour</p>") == "Bonjour"

    def test_inserts_space_between_blocks(self):
        # <p>foo</p><p>bar</p> ne doit pas devenir "foobar"
        out = strip_html("<p>foo</p><p>bar</p>")
        assert "foo" in out and "bar" in out
        assert "foobar" not in out

    def test_decodes_html_entities(self):
        assert strip_html("Caf&eacute; &amp; th&eacute;") == "Café & thé"

    def test_handles_nested_tags(self):
        out = strip_html("<div><strong>Concert</strong> de <em>jazz</em></div>")
        assert "Concert" in out and "jazz" in out
        assert "<" not in out and ">" not in out

    def test_preserves_none(self):
        assert strip_html(None) is None

    def test_handles_links(self):
        out = strip_html('Visitez <a href="https://example.com">notre site</a>')
        assert "notre site" in out
        assert "https://example.com" not in out  # texte seul, pas l'attribut href


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("Concert    jazz") == "Concert jazz"

    def test_strips_edges(self):
        assert normalize_whitespace("  hello  ") == "hello"

    def test_normalizes_newlines_and_tabs(self):
        assert normalize_whitespace("foo\n\tbar") == "foo bar"

    def test_empty_becomes_none(self):
        # Validé en planning : une chaîne vide après nettoyage = None
        # (préserve la distinction null vs vide en aval)
        assert normalize_whitespace("   ") is None
        assert normalize_whitespace("") is None

    def test_preserves_none(self):
        assert normalize_whitespace(None) is None

    def test_joins_list_of_strings(self):
        # keywords_fr et accessibility_label_fr arrivent en list dans Open Agenda
        assert normalize_whitespace(["jazz", "concert", "live"]) == "jazz, concert, live"

    def test_filters_empty_items_in_list(self):
        assert normalize_whitespace(["jazz", "", None, "concert"]) == "jazz, concert"

    def test_strips_lone_surrogates(self):
        # \ud835 isolé = caractère Unicode corrompu observé dans le dataset réel ;
        # planterait l'écriture UTF-8 si on le laissait passer.
        out = normalize_whitespace("Concert \ud835 jazz")
        assert out == "Concert jazz"
        # Le résultat doit pouvoir être réencodé en UTF-8 sans erreur.
        out.encode("utf-8")


# ---------------------------------------------------------------------------
# parse_attendance_mode
# ---------------------------------------------------------------------------

class TestParseAttendanceMode:
    SUR_PLACE = '{"id": 1, "label": {"fr": "Sur place"}}'
    EN_LIGNE = '{"id": 2, "label": {"fr": "En ligne"}}'
    MIXTE = '{"id": 3, "label": {"fr": "Mixte"}}'

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (SUR_PLACE, "sur_place"),
            (EN_LIGNE, "en_ligne"),
            (MIXTE, "mixte"),
        ],
    )
    def test_known_ids(self, raw: str, expected: str):
        assert parse_attendance_mode(raw) == expected

    def test_unknown_id_returns_none(self):
        assert parse_attendance_mode('{"id": 999}') is None

    def test_none_input(self):
        assert parse_attendance_mode(None) is None

    def test_malformed_json(self):
        assert parse_attendance_mode("not json at all") is None


# ---------------------------------------------------------------------------
# extract_year
# ---------------------------------------------------------------------------

class TestExtractYear:
    def test_iso_utc(self):
        assert extract_year("2024-06-15T14:00:00+00:00") == 2024

    def test_none(self):
        assert extract_year(None) is None

    def test_garbage(self):
        assert extract_year("not a date") is None


# ---------------------------------------------------------------------------
# is_valid
# ---------------------------------------------------------------------------

class TestIsValid:
    BASE: dict = {
        "title_fr": "Concert de jazz",
        "description_fr": "Un super concert",
        "event_year": 2024,
    }

    def test_valid_event(self):
        assert is_valid(self.BASE) is True

    def test_rejects_missing_title(self):
        assert is_valid({**self.BASE, "title_fr": None}) is False
        assert is_valid({**self.BASE, "title_fr": ""}) is False

    def test_rejects_missing_description(self):
        assert is_valid({**self.BASE, "description_fr": None}) is False

    def test_rejects_no_year(self):
        assert is_valid({**self.BASE, "event_year": None}) is False

    def test_rejects_aberrant_year(self):
        # 1900 et 2503 ont été observés dans le dataset réel
        assert is_valid({**self.BASE, "event_year": 1900}) is False
        assert is_valid({**self.BASE, "event_year": 2503}) is False
        assert is_valid({**self.BASE, "event_year": 2009}) is False  # juste sous le seuil

    def test_accepts_edges(self):
        assert is_valid({**self.BASE, "event_year": 2010}) is True
        assert is_valid({**self.BASE, "event_year": 2030}) is True


# ---------------------------------------------------------------------------
# clean_event (intégration des étapes)
# ---------------------------------------------------------------------------

class TestCleanEvent:
    def test_full_pipeline(self):
        raw = {
            "uid": "abc123",
            "title_fr": "  Concert   de jazz  ",
            "description_fr": "Un super concert",
            "longdescription_fr": "<p>Soir&eacute;e <strong>jazz</strong></p><p>Entr&eacute;e libre</p>",
            "conditions_fr": "<p>Gratuit</p>",
            "firstdate_begin": "2024-06-15T20:00:00+00:00",
            "location_name": "Le Bataclan",
            "attendancemode": '{"id": 1, "label": {"fr": "Sur place"}}',
        }
        out = clean_event(raw)

        assert out["title_fr"] == "Concert de jazz"
        assert "<" not in out["longdescription_fr"]
        assert "Soirée" in out["longdescription_fr"]
        assert "jazz" in out["longdescription_fr"]
        assert "Entrée libre" in out["longdescription_fr"]
        assert out["conditions_fr"] == "Gratuit"
        assert out["attendance_mode"] == "sur_place"
        assert out["event_year"] == 2024
        # Le champ brut a été remplacé par la version parsée
        assert "attendancemode" not in out

    def test_preserves_nulls_on_missing_text_fields(self):
        raw = {
            "uid": "x",
            "title_fr": "Titre",
            "description_fr": "desc",
            "longdescription_fr": None,
            "keywords_fr": None,
            "firstdate_begin": "2024-01-01T00:00:00+00:00",
        }
        out = clean_event(raw)
        assert out["longdescription_fr"] is None
        assert out["keywords_fr"] is None

    def test_does_not_mutate_input(self):
        raw = {"title_fr": "  espaces  ", "description_fr": "x"}
        clean_event(raw)
        # raw doit rester intact
        assert raw["title_fr"] == "  espaces  "

    def test_drops_constant_and_empty_fields(self):
        # country_fr est constant ("France (Métropole)") par construction du
        # filtre where=, et category est null à 100 % dans le snapshot
        # Opendatasoft. Les deux doivent être retirés au cleaning même si le
        # raw les contient encore (cas des fichiers téléchargés avant qu'on
        # les sorte du select=).
        raw = {
            "uid": "x",
            "title_fr": "Titre",
            "description_fr": "desc",
            "firstdate_begin": "2024-01-01T00:00:00+00:00",
            "country_fr": "France (Métropole)",
            "category": None,
        }
        out = clean_event(raw)
        assert "country_fr" not in out
        assert "category" not in out


# ---------------------------------------------------------------------------
# dedup_key
# ---------------------------------------------------------------------------

class TestDedupKey:
    def test_normal_case(self):
        ev = {
            "title_fr": "Concert",
            "firstdate_begin": "2024-06-15T20:00:00+00:00",
            "location_name": "Bataclan",
        }
        assert dedup_key(ev) == ("concert", "2024-06-15T20:00:00+00:00", "bataclan")

    def test_case_insensitive(self):
        ev1 = {"title_fr": "Concert", "firstdate_begin": "2024-06-15T20:00:00+00:00", "location_name": "Bataclan"}
        ev2 = {"title_fr": "CONCERT", "firstdate_begin": "2024-06-15T20:00:00+00:00", "location_name": "bataclan"}
        assert dedup_key(ev1) == dedup_key(ev2)

    def test_returns_none_when_missing_field(self):
        assert dedup_key({"title_fr": "x", "firstdate_begin": "2024-01-01T00:00:00+00:00"}) is None
        assert dedup_key({"title_fr": None, "firstdate_begin": "2024-01-01T00:00:00+00:00", "location_name": "a"}) is None


# ---------------------------------------------------------------------------
# Scénario bout-en-bout : rejet + dédup sur mini-fixture
# ---------------------------------------------------------------------------

def test_end_to_end_filtering_and_dedup():
    """Mini-fixture qui simule le pipeline complet : 5 events bruts dont on
    attend 2 conservés après cleaning + dédup."""
    raw_events = [
        # 1) valide
        {
            "uid": "a",
            "title_fr": "Concert jazz",
            "description_fr": "soirée",
            "firstdate_begin": "2024-06-15T20:00:00+00:00",
            "location_name": "Bataclan",
            "longdescription_fr": "<p>Foo</p>",
            "attendancemode": '{"id": 1, "label": {"fr": "Sur place"}}',
        },
        # 2) doublon strict de 1 (même titre + date + lieu)
        {
            "uid": "b",
            "title_fr": "Concert jazz",
            "description_fr": "Variante",
            "firstdate_begin": "2024-06-15T20:00:00+00:00",
            "location_name": "Bataclan",
            "longdescription_fr": "<p>Bar</p>",
        },
        # 3) invalide : pas de titre
        {
            "uid": "c",
            "title_fr": None,
            "description_fr": "x",
            "firstdate_begin": "2024-06-15T20:00:00+00:00",
        },
        # 4) invalide : année aberrante
        {
            "uid": "d",
            "title_fr": "Vieux truc",
            "description_fr": "x",
            "firstdate_begin": "1900-01-01T00:00:00+00:00",
        },
        # 5) valide, différent du 1
        {
            "uid": "e",
            "title_fr": "Expo peinture",
            "description_fr": "vernissage",
            "firstdate_begin": "2025-03-01T18:00:00+00:00",
            "location_name": "Musée d'Orsay",
        },
    ]

    seen: set = set()
    kept = []
    for raw in raw_events:
        cleaned = clean_event(raw)
        if not is_valid(cleaned):
            continue
        key = dedup_key(cleaned)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        kept.append(cleaned)

    assert len(kept) == 2
    titles = {ev["title_fr"] for ev in kept}
    assert titles == {"Concert jazz", "Expo peinture"}
