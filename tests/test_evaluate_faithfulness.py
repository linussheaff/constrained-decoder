"""Tests for ``src.evaluate_faithfulness``.

The faithfulness pipeline depends on spaCy NER, so the integration tests are
gated on the ``spacy_nlp`` fixture (which skips if ``en_core_web_sm`` is not
installed). A small set of pure-function tests covers the matching helpers
without any model dependency.
"""

from __future__ import annotations

import pytest

from src.entity_extractor import Entity, Number, SourceFacts
from src.evaluate_faithfulness import (
    FaithfulnessReport,
    _entity_appears_in_source,
    _normalise_entity,
    _number_supported,
    evaluate_faithfulness,
)


# ---------------------------------------------------------------------------
# Helper-function unit tests (no spaCy)
# ---------------------------------------------------------------------------


class TestNormaliseEntity:
    def test_lowercases(self) -> None:
        assert _normalise_entity("Apple Inc") == "apple inc"

    def test_collapses_whitespace(self) -> None:
        assert _normalise_entity("  Apple   Inc  \n") == "apple inc"


class TestEntityAppearsInSource:
    def test_exact_match(self) -> None:
        assert _entity_appears_in_source("Apple", "I bought an apple yesterday")

    def test_case_insensitive(self) -> None:
        assert _entity_appears_in_source("APPLE", "i bought an apple")

    def test_word_boundary_blocks_substring_match(self) -> None:
        # "Apple" should not match inside "Pineapple".
        assert not _entity_appears_in_source("apple", "i ate a pineapple")

    def test_multi_word_entity(self) -> None:
        assert _entity_appears_in_source(
            "Apple Inc", "apple inc reported earnings"
        )

    def test_empty_entity_does_not_match(self) -> None:
        assert not _entity_appears_in_source("", "anything")

    def test_punctuation_in_source_does_not_block(self) -> None:
        # "London," in source should match "London" as an entity.
        assert _entity_appears_in_source("London", "she lives in london, uk")


class TestNumberSupported:
    def test_exact_match(self) -> None:
        assert _number_supported(42.0, [1.0, 42.0, 99.0])

    def test_no_match(self) -> None:
        assert not _number_supported(42.0, [1.0, 99.0])

    def test_none_value_unsupported(self) -> None:
        assert not _number_supported(None, [1.0])

    def test_floating_point_tolerance(self) -> None:
        # Reconstructed-via-cents values should still match.
        assert _number_supported(0.1 + 0.2, [0.3])


# ---------------------------------------------------------------------------
# FaithfulnessReport derived metrics
# ---------------------------------------------------------------------------


class TestReportProperties:
    def test_perfect_precision_recall(self) -> None:
        r = FaithfulnessReport(
            summary="x",
            source="y",
            summary_entities=["A", "B"],
            source_entities=["A", "B"],
            matched_entities=["A", "B"],
            recalled_entities=["A", "B"],
        )
        assert r.entity_precision == 1.0
        assert r.entity_recall == 1.0
        assert r.entity_f1 == 1.0

    def test_empty_summary_precision_is_one(self) -> None:
        # Convention: no entities → vacuous precision.
        r = FaithfulnessReport(summary="", source="x", summary_entities=[])
        assert r.entity_precision == 1.0

    def test_empty_source_recall_is_one(self) -> None:
        r = FaithfulnessReport(summary="x", source="", source_entities=[])
        assert r.entity_recall == 1.0

    def test_zero_recall(self) -> None:
        r = FaithfulnessReport(
            summary="x",
            source="y",
            source_entities=["A", "B"],
            recalled_entities=[],
        )
        assert r.entity_recall == 0.0

    def test_zero_pr_yields_zero_f1(self) -> None:
        r = FaithfulnessReport(
            summary="x",
            source="y",
            summary_entities=["A"],
            source_entities=["B"],
            matched_entities=[],
            recalled_entities=[],
        )
        assert r.entity_f1 == 0.0

    def test_number_accuracy_vacuous(self) -> None:
        r = FaithfulnessReport(summary="", source="x", summary_numbers=[])
        assert r.number_accuracy == 1.0

    def test_hallucinated_count(self) -> None:
        r = FaithfulnessReport(
            summary="x",
            source="y",
            hallucinated_entities=["Google", "Microsoft"],
        )
        assert r.hallucinated_entity_count == 2


# ---------------------------------------------------------------------------
# End-to-end (spaCy-backed) faithfulness evaluation
# ---------------------------------------------------------------------------


class TestEvaluateFaithfulnessIntegration:
    def test_perfect_copy_matches_everything(self, spacy_nlp) -> None:
        source = "Apple Inc reported a profit of $5 billion in 2023."
        summary = source
        r = evaluate_faithfulness(summary, source)
        assert r.entity_precision == 1.0
        # Source entities → all should be in the summary (since summary == source).
        assert r.entity_recall == 1.0
        assert r.hallucinated_entities == []
        assert r.number_accuracy == 1.0

    def test_hallucinated_entity_flagged(self, spacy_nlp) -> None:
        source = "Apple Inc reported a profit in 2023."
        summary = "Apple Inc and Microsoft reported a profit in 2023."
        r = evaluate_faithfulness(summary, source)
        # "Microsoft" should be flagged as hallucinated.
        hallucinated_lower = [e.lower() for e in r.hallucinated_entities]
        assert "microsoft" in hallucinated_lower
        assert r.hallucinated_entity_count >= 1
        assert r.entity_precision < 1.0

    def test_unsupported_number_flagged(self, spacy_nlp) -> None:
        source = "Apple reported a profit of $5 billion."
        summary = "Apple reported a profit of $7 billion."
        r = evaluate_faithfulness(summary, source)
        assert r.unsupported_number_count >= 1
        # And the matched-numbers list should not contain 7's text.
        assert "7" not in "".join(r.matched_numbers)
        assert r.number_accuracy < 1.0

    def test_number_value_match_ignores_formatting(self, spacy_nlp) -> None:
        # "$1,000" in source ↔ "1000" in summary — same value, different surface.
        source = "The fund holds $1,000."
        summary = "The fund holds 1000 dollars."
        r = evaluate_faithfulness(summary, source)
        # At least one summary number should be supported.
        assert r.matched_numbers, r

    def test_empty_summary_zero_recall(self, spacy_nlp) -> None:
        source = "Apple Inc reported earnings in London."
        r = evaluate_faithfulness("", source)
        assert r.entity_recall == 0.0
        # No summary entities → vacuous precision.
        assert r.entity_precision == 1.0
        assert r.hallucinated_entities == []

    def test_partial_match_partial_metrics(self, spacy_nlp) -> None:
        source = "Apple Inc reported earnings in 2023."
        summary = "Apple Inc reported earnings in 2024."  # date changed
        r = evaluate_faithfulness(summary, source)
        assert r.hallucinated_entity_count >= 1
        # Apple Inc should still be recalled and matched.
        assert any("apple" in e.lower() for e in r.matched_entities)

    def test_accepts_precomputed_facts(self, spacy_nlp) -> None:
        # Pre-build the source facts (typical use: same source, many summaries).
        from src.entity_extractor import extract_facts

        source = "Apple Inc reported earnings in 2023."
        summary = "Apple Inc reported earnings in 2023."
        source_facts = extract_facts(source, tokenizer=None)
        r = evaluate_faithfulness(
            summary, source, source_facts=source_facts
        )
        assert r.entity_precision == 1.0
