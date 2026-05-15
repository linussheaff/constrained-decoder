"""Tests for ``src.entity_extractor``.

Covers entity extraction, regex number fallback, multi-token tokenisation,
edge cases (empty / numeric-free input), and that the resulting allowlist
actually contains tokens for the spans we care about.
"""

from __future__ import annotations

import pytest

from src.entity_extractor import (
    Entity,
    Number,
    SourceFacts,
    _NUMBER_REGEX,
    _parse_number_value,
    _regex_numbers,
    _tokenize_variants,
    extract_facts,
)


# ---------------------------------------------------------------------------
# Pure / no-spaCy unit tests (run unconditionally)
# ---------------------------------------------------------------------------


class TestNumberRegex:
    def test_matches_plain_integer(self) -> None:
        matches = [m.group() for m in _NUMBER_REGEX.finditer("There were 42 cats.")]
        assert matches == ["42"]

    def test_matches_decimal(self) -> None:
        matches = [m.group() for m in _NUMBER_REGEX.finditer("price was 3.14 yesterday")]
        assert matches == ["3.14"]

    def test_matches_thousands_separator(self) -> None:
        matches = [m.group() for m in _NUMBER_REGEX.finditer("Loss of 1,250,000 dollars")]
        assert matches == ["1,250,000"]

    def test_matches_currency_and_percent(self) -> None:
        text = "Revenue rose 12.5% to $4,200 last quarter"
        matches = [m.group() for m in _NUMBER_REGEX.finditer(text)]
        assert "12.5%" in matches
        assert "$4,200" in matches


class TestParseNumberValue:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("42", 42.0),
            ("3.14", 3.14),
            ("1,000", 1000.0),
            ("$1,250.50", 1250.50),
            ("£250", 250.0),
            ("50%", 0.5),
            ("-7", -7.0),
        ],
    )
    def test_known_values(self, text: str, expected: float) -> None:
        assert _parse_number_value(text) == pytest.approx(expected)

    def test_unparseable_returns_none(self) -> None:
        assert _parse_number_value("not a number") is None


class TestRegexNumbersSkipEntities:
    def test_skips_numbers_inside_entity_spans(self) -> None:
        # Pretend "April 1, 2024" is a DATE entity covering chars 0..13.
        source = "April 1, 2024 was the deadline; only 7 people met it."
        ignore = [(0, 13)]
        nums = _regex_numbers(source, ignore)
        texts = [n.text for n in nums]
        assert "7" in texts
        assert "1" not in texts
        assert "2024" not in texts


class TestTokenizeVariants:
    def test_emits_with_and_without_leading_space(self, fake_tokenizer) -> None:
        seqs = _tokenize_variants("Lloyds", fake_tokenizer)
        # Two surface variants → two distinct token id sequences.
        assert len(seqs) == 2
        assert seqs[0] != seqs[1]

    def test_multi_token_entity_preserves_order(self, fake_tokenizer) -> None:
        seqs = _tokenize_variants("Lloyds Banking Group", fake_tokenizer)
        # Both variants tokenise into 3 pieces.
        for seq in seqs:
            assert len(seq) == 3

    def test_empty_text_returns_empty(self, fake_tokenizer) -> None:
        assert _tokenize_variants("", fake_tokenizer) == []

    def test_leading_space_is_not_duplicated(self, fake_tokenizer) -> None:
        seqs = _tokenize_variants(" Lloyds", fake_tokenizer)
        assert len(seqs) == 1


# ---------------------------------------------------------------------------
# SourceFacts dataclass behaviour
# ---------------------------------------------------------------------------


class TestSourceFacts:
    def test_factual_token_ids_is_union(self) -> None:
        facts = SourceFacts(
            source_text="x",
            entities=[],
            numbers=[],
            entity_tokens={1, 2, 3},
            number_tokens={3, 4, 5},
        )
        assert facts.factual_token_ids == {1, 2, 3, 4, 5}

    def test_all_spans_sorted(self) -> None:
        facts = SourceFacts(
            source_text="hello world 42",
            entities=[Entity("hello", "ORG", 0, 5)],
            numbers=[Number("42", 42.0, 12, 14)],
        )
        spans = facts.all_spans
        assert spans == [(0, 5, "entity"), (12, 14, "number")]


# ---------------------------------------------------------------------------
# Integration tests requiring spaCy + en_core_web_sm
# ---------------------------------------------------------------------------


class TestExtractFactsIntegration:
    """End-to-end tests that need a real spaCy NER pipeline."""

    def test_extracts_named_entities(self, fake_tokenizer, spacy_nlp) -> None:
        source = (
            "Lloyds Banking Group reported a profit of £2.1 billion in 2023, "
            "according to chief executive Charlie Nunn."
        )
        facts = extract_facts(source, fake_tokenizer)

        labels = {e.label for e in facts.entities}
        texts = {e.text for e in facts.entities}

        # spaCy may classify the org slightly differently across versions, so
        # we assert on the substring rather than exact label.
        assert any("Lloyds" in t for t in texts), texts
        assert any(t == "Charlie Nunn" or "Nunn" in t for t in texts), texts
        # Some factual labels must have been produced.
        assert labels & {"PERSON", "ORG", "MONEY", "DATE"}

    def test_entity_tokens_cover_entity_text(self, fake_tokenizer, spacy_nlp) -> None:
        source = "Lloyds Banking Group is based in London."
        facts = extract_facts(source, fake_tokenizer)

        # Re-tokenise the raw entity text and confirm every piece is in the
        # allowlist (across either surface variant).
        for ent in facts.entities:
            for seq in _tokenize_variants(ent.text, fake_tokenizer):
                assert set(seq).issubset(facts.entity_tokens), (
                    f"missing tokens for entity {ent.text!r}"
                )

    def test_numbers_fallback_picks_up_bare_numbers(
        self, fake_tokenizer, spacy_nlp
    ) -> None:
        # Stripped of contextual cues, spaCy often labels bare numbers as
        # CARDINAL — but the regex fallback should pick them up regardless.
        source = "The result was 42 and the ratio 0.71 was reported."
        facts = extract_facts(source, fake_tokenizer)
        numeric_strings = {n.text for n in facts.numbers}
        entity_strings = {e.text for e in facts.entities}
        all_numeric = numeric_strings | {
            e.text for e in facts.entities if e.label in {"CARDINAL", "QUANTITY"}
        }
        assert "42" in all_numeric or "42" in entity_strings
        assert "0.71" in all_numeric or "0.71" in entity_strings

    def test_empty_document_returns_empty_facts(self, fake_tokenizer, spacy_nlp) -> None:
        facts = extract_facts("", fake_tokenizer)
        assert facts.entities == []
        assert facts.numbers == []
        assert facts.entity_tokens == set()
        assert facts.number_tokens == set()

    def test_disabling_numbers_drops_regex_matches(
        self, fake_tokenizer, spacy_nlp
    ) -> None:
        source = "I have 3 apples."
        with_numbers = extract_facts(source, fake_tokenizer)
        without_numbers = extract_facts(source, fake_tokenizer, include_numbers=False)
        assert without_numbers.numbers == []
        # The "with" pass should contain the regex 3 unless spaCy claimed it.
        if not any(e.text == "3" for e in with_numbers.entities):
            assert any(n.text == "3" for n in with_numbers.numbers)

    def test_disabling_entity_labels(self, fake_tokenizer, spacy_nlp) -> None:
        source = "Apple Inc. is a company."
        facts = extract_facts(source, fake_tokenizer, entity_labels=frozenset())
        assert facts.entities == []

    def test_non_string_source_raises(self, fake_tokenizer, spacy_nlp) -> None:
        with pytest.raises(TypeError):
            extract_facts(None, fake_tokenizer)  # type: ignore[arg-type]
