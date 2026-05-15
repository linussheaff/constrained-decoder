"""Tests for ``src.detector.HeuristicFactualSpanDetector``.

The pure-string heuristic is dependency-free and gets the bulk of coverage.
A small set of tests exercises the ``is_active`` tokenizer-decoding path
using the fake tokenizer from ``conftest``.
"""

from __future__ import annotations

import pytest

from src.detector import (
    DEFAULT_PRECEDER_WORDS,
    DEFAULT_TRIGGER_PHRASES,
    HeuristicFactualSpanDetector,
)


# ---------------------------------------------------------------------------
# Pure-string heuristic
# ---------------------------------------------------------------------------


@pytest.fixture
def detector() -> HeuristicFactualSpanDetector:
    return HeuristicFactualSpanDetector()


class TestEdgeCases:
    def test_empty_text_inactive(self, detector) -> None:
        assert detector.is_active_from_text("") is False

    def test_whitespace_only_inactive(self, detector) -> None:
        assert detector.is_active_from_text("    ") is False

    def test_plain_text_inactive(self, detector) -> None:
        # No triggers, no preceders, no terminators — should not fire.
        assert detector.is_active_from_text("hello world") is False

    def test_invalid_lookback_raises(self) -> None:
        with pytest.raises(ValueError):
            HeuristicFactualSpanDetector(lookback_chars=0)
        with pytest.raises(ValueError):
            HeuristicFactualSpanDetector(lookback_chars=-1)


class TestTriggerPhrases:
    def test_reported_fires(self, detector) -> None:
        assert detector.is_active_from_text("Apple Inc reported")

    def test_said_that_fires(self, detector) -> None:
        assert detector.is_active_from_text("She said that")

    def test_valued_at_fires(self, detector) -> None:
        assert detector.is_active_from_text("a portfolio valued at")

    def test_trigger_inside_clause_keeps_active(self, detector) -> None:
        # The trigger is mid-clause; the rest of the clause should remain
        # under constraint until a terminator appears.
        assert detector.is_active_from_text("She reported a profit of")

    def test_trigger_is_word_bounded(self, detector) -> None:
        # "reporter" should not match the "reported" trigger. We pick a
        # surrounding sentence that contains no other default trigger words
        # (no "wrote", "said", "ran outside", etc.).
        assert not detector.is_active_from_text("a friendly reporter")

    def test_case_insensitive(self, detector) -> None:
        assert detector.is_active_from_text("APPLE INC REPORTED")


class TestPrecederWords:
    def test_preposition_fires(self, detector) -> None:
        assert detector.is_active_from_text("He was born in")

    def test_honorific_fires(self, detector) -> None:
        assert detector.is_active_from_text("Then Mr")

    def test_preposition_not_last_does_not_fire(self, detector) -> None:
        # "in" appears but is not the last word.
        assert not detector.is_active_from_text("a cat in the hat sleeps")

    def test_determiner_does_not_fire(self, detector) -> None:
        # Determiners "the/a/an" are deliberately excluded by default.
        assert not detector.is_active_from_text("the")
        assert not detector.is_active_from_text("a")


class TestCurrencyTriggers:
    def test_dollar_at_end_fires(self, detector) -> None:
        assert detector.is_active_from_text("the cost was $")

    def test_pound_at_end_fires(self, detector) -> None:
        assert detector.is_active_from_text("worth £")

    def test_currency_with_trailing_space_fires(self, detector) -> None:
        # We rstrip before comparing, so trailing whitespace shouldn't matter.
        assert detector.is_active_from_text("the cost was $   ")

    def test_currency_in_middle_does_not_fire_via_currency_rule(
        self, detector
    ) -> None:
        # Currency-rule should require end-of-span; mid-span $ alone shouldn't
        # fire from this rule (though the trigger "worth" still might).
        d = HeuristicFactualSpanDetector(
            trigger_phrases=(), preceder_words=()
        )
        assert not d.is_active_from_text("$5 was the price of milk")


class TestTerminators:
    def test_period_deactivates(self, detector) -> None:
        assert not detector.is_active_from_text(
            "Apple Inc reported earnings today."
        )

    def test_comma_deactivates(self, detector) -> None:
        assert not detector.is_active_from_text(
            "Apple Inc reported earnings,"
        )

    def test_question_mark_deactivates(self, detector) -> None:
        assert not detector.is_active_from_text("Did they report? ")

    def test_conjunction_deactivates(self, detector) -> None:
        # "and" as a conjunction word splits the clause.
        assert not detector.is_active_from_text(
            "The CEO said hello and the world is fine"
        )

    def test_reactivates_after_new_trigger_post_terminator(
        self, detector
    ) -> None:
        # Terminator clears the span, then a fresh trigger fires.
        assert detector.is_active_from_text(
            "Earnings were strong. The CEO said"
        )

    def test_colon_deactivates(self, detector) -> None:
        assert not detector.is_active_from_text("Summary:")

    def test_newline_deactivates(self, detector) -> None:
        assert not detector.is_active_from_text("Apple reported earnings.\n")


class TestLookback:
    def test_short_lookback_skips_old_trigger(self) -> None:
        d = HeuristicFactualSpanDetector(lookback_chars=10)
        text = "reported " + ("x" * 100)
        # The trigger is more than lookback_chars before the end.
        assert not d.is_active_from_text(text)

    def test_long_lookback_catches_distant_trigger(self) -> None:
        d = HeuristicFactualSpanDetector(lookback_chars=200)
        text = "reported " + ("xxx " * 20)  # ≈ 80 chars trailing — within lookback
        assert d.is_active_from_text(text)


class TestCustomConfiguration:
    def test_custom_trigger_only(self) -> None:
        d = HeuristicFactualSpanDetector(
            trigger_phrases=["foobar"], preceder_words=(), currency_triggers=()
        )
        assert d.is_active_from_text("xx foobar")
        assert not d.is_active_from_text("xx reported")

    def test_disable_all_rules_never_fires(self) -> None:
        d = HeuristicFactualSpanDetector(
            trigger_phrases=(),
            preceder_words=(),
            currency_triggers=(),
        )
        assert not d.is_active_from_text("reported worth $ in Mr")

    def test_defaults_exposed_for_extension(self) -> None:
        assert "said" in DEFAULT_TRIGGER_PHRASES
        assert "in" in DEFAULT_PRECEDER_WORDS

    def test_reset_is_noop(self, detector) -> None:
        # Sanity check: state-free detector survives reset().
        detector.reset()
        assert detector.is_active_from_text("reported")


# ---------------------------------------------------------------------------
# Integration with a tokenizer.decode-style interface
# ---------------------------------------------------------------------------


class TestIsActiveWithTokenizer:
    def test_decoded_text_triggers(self, fake_tokenizer) -> None:
        d = HeuristicFactualSpanDetector()
        ids = fake_tokenizer.encode("Apple reported", add_special_tokens=False)
        assert d.is_active(ids, fake_tokenizer)

    def test_empty_ids_inactive(self, fake_tokenizer) -> None:
        d = HeuristicFactualSpanDetector()
        assert not d.is_active([], fake_tokenizer)

    def test_decoded_text_does_not_trigger(self, fake_tokenizer) -> None:
        d = HeuristicFactualSpanDetector()
        ids = fake_tokenizer.encode("a quiet field of grass.", add_special_tokens=False)
        assert not d.is_active(ids, fake_tokenizer)
