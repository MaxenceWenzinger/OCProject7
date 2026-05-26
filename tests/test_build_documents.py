"""Tests unitaires pour la conversion event clean → Document LangChain."""

from __future__ import annotations

from langchain_core.documents import Document

from src.indexing.build_documents import (
    build_metadata,
    build_page_content,
    event_to_chunks,
    event_to_document,
)


class _CharTokenizer:
    """Faux tokenizer déterministe : 1 caractère = 1 token.

    Évite de charger MiniLM (300+ MB) pour tester la logique de découpage.
    `encode` retourne les codes Unicode, `decode` les reconvertit en string."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in token_ids)


FULL_EVENT: dict = {
    "uid": "abc123",
    "slug": "concert-jazz-bataclan",
    "canonicalurl": "https://openagenda.com/x/events/concert-jazz",
    "title_fr": "Concert de jazz manouche",
    "description_fr": "Soirée jazz au Bataclan",
    "longdescription_fr": "Une soirée exceptionnelle de jazz manouche avec trois musiciens reconnus.",
    "keywords_fr": "Jazz, Concert, Manouche",
    "conditions_fr": "Tarif unique 15 €, gratuit pour les moins de 12 ans",
    "firstdate_begin": "2025-06-15T20:00:00+00:00",
    "firstdate_end": "2025-06-15T23:00:00+00:00",
    "lastdate_begin": "2025-06-15T20:00:00+00:00",
    "lastdate_end": "2025-06-15T23:00:00+00:00",
    "location_name": "Le Bataclan",
    "location_address": "50 Boulevard Voltaire",
    "location_city": "Paris",
    "location_postalcode": "75011",
    "location_department": "Paris",
    "location_region": "Île-de-France",
    "location_coordinates": [48.86, 2.37],
    "age_min": 0,
    "age_max": None,
    "accessibility_label_fr": "Accès PMR",
    "attendance_mode": "sur_place",
    "event_year": 2025,
}


# ---------------------------------------------------------------------------
# build_page_content
# ---------------------------------------------------------------------------

class TestBuildPageContent:
    def test_includes_all_present_blocks(self):
        content = build_page_content(FULL_EVENT)
        assert "Concert de jazz manouche" in content
        assert "Soirée jazz au Bataclan" in content
        assert "trois musiciens reconnus" in content
        assert "Mots-clés : Jazz, Concert, Manouche" in content
        assert "Conditions : Tarif unique" in content

    def test_blocks_separated_by_blank_line(self):
        content = build_page_content(FULL_EVENT)
        assert "\n\n" in content
        # Pas de bloc collé sans séparateur
        assert "BataclanUne soirée" not in content

    def test_title_appears_first(self):
        content = build_page_content(FULL_EVENT)
        idx_title = content.index("Concert de jazz manouche")
        idx_desc = content.index("Soirée jazz")
        idx_long = content.index("trois musiciens")
        idx_kw = content.index("Mots-clés")
        idx_cond = content.index("Conditions")
        assert idx_title < idx_desc < idx_long < idx_kw < idx_cond

    def test_omits_absent_fields(self):
        event = {
            "title_fr": "Concert",
            "description_fr": "Soirée",
            "longdescription_fr": None,
            "keywords_fr": None,
            "conditions_fr": None,
        }
        content = build_page_content(event)
        assert content == "Concert\n\nSoirée"
        # Pas de préfixes parasites pour les champs absents
        assert "Mots-clés" not in content
        assert "Conditions" not in content

    def test_omits_empty_strings(self):
        # Le cleaning normalise les chaînes vides en None, mais on prévoit
        # le cas robuste où l'event arriverait avec "" (cas non garanti).
        event = {
            "title_fr": "Concert",
            "description_fr": "",
            "longdescription_fr": "Détails",
            "keywords_fr": "",
            "conditions_fr": None,
        }
        content = build_page_content(event)
        assert content == "Concert\n\nDétails"

    def test_minimal_event_with_only_title(self):
        event = {"title_fr": "Concert"}
        assert build_page_content(event) == "Concert"

    def test_empty_event_returns_empty_string(self):
        assert build_page_content({}) == ""


# ---------------------------------------------------------------------------
# build_metadata
# ---------------------------------------------------------------------------

class TestBuildMetadata:
    EXPECTED_KEYS = {
        "uid",
        "title",
        "url",
        "first_date",
        "last_date",
        "location_name",
        "location_city",
        "location_region",
        "attendance_mode",
        "event_year",
    }

    def test_returns_exactly_the_10_expected_keys(self):
        meta = build_metadata(FULL_EVENT)
        assert set(meta.keys()) == self.EXPECTED_KEYS

    def test_maps_clean_fields_to_metadata_names(self):
        meta = build_metadata(FULL_EVENT)
        assert meta["uid"] == "abc123"
        assert meta["title"] == "Concert de jazz manouche"
        assert meta["url"] == "https://openagenda.com/x/events/concert-jazz"
        assert meta["first_date"] == "2025-06-15T20:00:00+00:00"
        assert meta["last_date"] == "2025-06-15T23:00:00+00:00"
        assert meta["location_name"] == "Le Bataclan"
        assert meta["location_city"] == "Paris"
        assert meta["location_region"] == "Île-de-France"
        assert meta["attendance_mode"] == "sur_place"
        assert meta["event_year"] == 2025

    def test_does_not_leak_excluded_fields(self):
        meta = build_metadata(FULL_EVENT)
        # Champs textuels déjà dans page_content : ne doivent PAS être dupliqués
        for excluded in ("description_fr", "longdescription_fr", "keywords_fr", "conditions_fr"):
            assert excluded not in meta
        # Champs jugés non-utiles au POC : pas en metadata non plus
        for excluded in (
            "slug",
            "location_address",
            "location_postalcode",
            "location_department",
            "location_coordinates",
            "age_min",
            "age_max",
            "accessibility_label_fr",
            "firstdate_end",
            "lastdate_begin",
        ):
            assert excluded not in meta

    def test_last_date_falls_back_on_firstdate_end(self):
        # lastdate_end absent → on retombe sur firstdate_end
        event = {
            "uid": "x",
            "lastdate_end": None,
            "firstdate_end": "2025-09-10T18:00:00+00:00",
        }
        meta = build_metadata(event)
        assert meta["last_date"] == "2025-09-10T18:00:00+00:00"

    def test_last_date_none_when_both_missing(self):
        meta = build_metadata({"uid": "x"})
        assert meta["last_date"] is None

    def test_preserves_none_values(self):
        # Un event minimal donne None partout sauf ce qui est passé
        event = {"uid": "x", "title_fr": "T"}
        meta = build_metadata(event)
        assert meta["uid"] == "x"
        assert meta["title"] == "T"
        assert meta["url"] is None
        assert meta["location_city"] is None


# ---------------------------------------------------------------------------
# event_to_document
# ---------------------------------------------------------------------------

class TestEventToDocument:
    def test_returns_langchain_document(self):
        doc = event_to_document(FULL_EVENT)
        assert isinstance(doc, Document)

    def test_page_content_and_metadata_attached(self):
        doc = event_to_document(FULL_EVENT)
        assert "Concert de jazz manouche" in doc.page_content
        assert doc.metadata["uid"] == "abc123"
        assert doc.metadata["event_year"] == 2025

    def test_metadata_does_not_mutate_input(self):
        event = dict(FULL_EVENT)
        event_to_document(event)
        # L'input event doit rester intact
        assert event == FULL_EVENT


# ---------------------------------------------------------------------------
# event_to_chunks
# ---------------------------------------------------------------------------

class TestEventToChunks:
    TOKENIZER = _CharTokenizer()

    def test_short_event_produces_single_chunk(self):
        # Un event qui tient sous la limite → 1 chunk avec le page_content
        # original (pas re-décodé).
        event = {"uid": "x", "title_fr": "Concert", "description_fr": "Soirée"}
        chunks = event_to_chunks(event, self.TOKENIZER, chunk_size=120, overlap=24)
        assert len(chunks) == 1
        assert chunks[0].page_content == "Concert\n\nSoirée"
        assert chunks[0].metadata["parent_uid"] == "x"
        assert chunks[0].metadata["chunk_index"] == 0

    def test_long_event_produces_multiple_chunks(self):
        # Texte total = "T\n\n" + "x"*300 = 303 chars
        # chunk_size=120, overlap=24 → step=96
        # Chunks à starts : 0, 96, 192 → 3 chunks (le dernier [192:303] = 111 chars,
        # end=312 ≥ 303 donc on break après).
        event = {"uid": "long", "title_fr": "T", "description_fr": "x" * 300}
        chunks = event_to_chunks(event, self.TOKENIZER, chunk_size=120, overlap=24)
        assert len(chunks) == 3
        assert [c.metadata["chunk_index"] for c in chunks] == [0, 1, 2]
        assert all(c.metadata["parent_uid"] == "long" for c in chunks)

    def test_overlap_creates_redundancy(self):
        # Vérifie que deux chunks consécutifs partagent bien `overlap` tokens
        event = {"uid": "x", "title_fr": "A" * 300, "description_fr": "B"}
        chunks = event_to_chunks(event, self.TOKENIZER, chunk_size=100, overlap=20)
        # Step = 80, premier chunk = [0:100], deuxième = [80:180]
        # Donc les chars [80:100] sont dans les deux
        if len(chunks) >= 2:
            tail_of_first = chunks[0].page_content[-20:]
            head_of_second = chunks[1].page_content[:20]
            assert tail_of_first == head_of_second

    def test_chunk_metadata_carries_all_parent_fields(self):
        chunks = event_to_chunks(FULL_EVENT, self.TOKENIZER, chunk_size=120, overlap=24)
        # Tous les champs metadata du parent doivent être présents dans chaque chunk
        for chunk in chunks:
            for parent_key in (
                "uid", "title", "url", "first_date", "last_date",
                "location_name", "location_city", "location_region",
                "attendance_mode", "event_year",
            ):
                assert parent_key in chunk.metadata
            # Plus les deux clés ajoutées
            assert "parent_uid" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_empty_event_returns_empty_list(self):
        assert event_to_chunks({}, self.TOKENIZER) == []
        # Un event sans aucun champ textuel → page_content vide → []
        assert event_to_chunks({"uid": "x"}, self.TOKENIZER) == []

    def test_does_not_mutate_input(self):
        event = dict(FULL_EVENT)
        event_to_chunks(event, self.TOKENIZER)
        assert event == FULL_EVENT

    def test_no_chunk_exceeds_chunk_size(self):
        # Aucun chunk produit ne doit dépasser chunk_size tokens
        event = {"uid": "x", "title_fr": "T", "description_fr": "z" * 500}
        chunks = event_to_chunks(event, self.TOKENIZER, chunk_size=80, overlap=10)
        for chunk in chunks:
            # Avec ce tokenizer 1 char = 1 token, donc len() en chars = len() en tokens
            assert len(chunk.page_content) <= 80

    def test_chunk_indices_are_sequential_starting_at_zero(self):
        event = {"uid": "x", "title_fr": "T", "description_fr": "x" * 1000}
        chunks = event_to_chunks(event, self.TOKENIZER, chunk_size=100, overlap=20)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))
