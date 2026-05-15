"""Tests for ``src.evaluate_quality``."""

from __future__ import annotations

import pytest

pytest.importorskip("rouge_score")

from src.evaluate_quality import (  # noqa: E402
    QualityReport,
    compute_rouge,
    evaluate_quality,
)


class TestComputeRouge:
    def test_identical_strings_score_one(self) -> None:
        scores = compute_rouge("the cat sat on the mat", "the cat sat on the mat")
        assert scores["rouge1"] == pytest.approx(1.0)
        assert scores["rouge2"] == pytest.approx(1.0)
        assert scores["rougeL"] == pytest.approx(1.0)

    def test_disjoint_strings_score_zero(self) -> None:
        scores = compute_rouge("hello world", "goodbye moon")
        assert scores["rouge1"] == 0.0
        assert scores["rouge2"] == 0.0
        assert scores["rougeL"] == 0.0

    def test_empty_prediction(self) -> None:
        scores = compute_rouge("", "the cat sat")
        assert scores == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def test_empty_reference(self) -> None:
        scores = compute_rouge("the cat sat", "")
        assert scores == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def test_partial_overlap_in_unit_interval(self) -> None:
        scores = compute_rouge(
            "the cat sat on the mat",
            "the dog sat on the rug",
        )
        for key in ("rouge1", "rouge2", "rougeL"):
            assert 0.0 < scores[key] < 1.0

    def test_rouge2_le_rouge1(self) -> None:
        # ROUGE-2 is bigram overlap and can't exceed unigram overlap.
        scores = compute_rouge(
            "a quick brown fox jumps over the lazy dog",
            "the quick brown fox vaults the lazy dog",
        )
        assert scores["rouge2"] <= scores["rouge1"]


class TestEvaluateQuality:
    def test_length_chars(self) -> None:
        report = evaluate_quality("hello world", reference=None)
        assert report.length_chars == 11

    def test_length_tokens_with_tokenizer(self, fake_tokenizer) -> None:
        report = evaluate_quality(
            "hello world", reference=None, tokenizer=fake_tokenizer
        )
        # FakeTokenizer splits on whitespace into 2 pieces.
        assert report.length_tokens == 2

    def test_length_tokens_none_without_tokenizer(self) -> None:
        report = evaluate_quality("hello world", reference=None)
        assert report.length_tokens is None

    def test_rouge_fields_none_without_reference(self) -> None:
        report = evaluate_quality("hello world", reference=None)
        assert report.rouge1 is None
        assert report.rouge2 is None
        assert report.rougeL is None

    def test_rouge_populated_with_reference(self) -> None:
        report = evaluate_quality(
            "the cat sat on the mat",
            reference="the cat sat on the mat",
        )
        assert report.rouge1 == pytest.approx(1.0)
        assert report.rouge2 == pytest.approx(1.0)
        assert report.rougeL == pytest.approx(1.0)

    def test_empty_summary(self) -> None:
        report = evaluate_quality("", reference="something")
        assert report.length_chars == 0
        # Empty prediction → zeros via compute_rouge guard.
        assert report.rouge1 == 0.0

    def test_returns_quality_report(self) -> None:
        assert isinstance(evaluate_quality("hi"), QualityReport)
