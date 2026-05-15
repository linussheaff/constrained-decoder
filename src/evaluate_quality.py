"""Quality metrics for generated summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol


class EncodableTokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]:
        ...


# ROUGE


@lru_cache(maxsize=1)
def _get_rouge_scorer():
    """Lazily-imported, cached ROUGE scorer"""
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:  # pragma: no cover — import guard
        raise RuntimeError(
            "rouge_score not installed; pip install rouge-score"
        ) from exc

    return rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )


def compute_rouge(prediction: str, reference: str) -> dict[str, float]:
    """Return F-measure for ROUGE-1, ROUGE-2 and ROUGE-L.
    """
    if not prediction or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scorer = _get_rouge_scorer()
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure),
    }


# Quality report


@dataclass
class QualityReport:
    """Quality metrics for a single summary."""

    summary: str
    reference: str | None
    length_chars: int
    length_tokens: int | None
    rouge1: float | None
    rouge2: float | None
    rougeL: float | None


def evaluate_quality(
    summary: str,
    reference: str | None = None,
    tokenizer: EncodableTokenizer | None = None,
) -> QualityReport:
    """Compute a :class:`QualityReport` for ``summary``.

    Args:
        summary: Generated summary text.
        reference: Optional gold reference; required for ROUGE.
        tokenizer: Optional tokenizer; required for length_tokens.

    Returns:
        A :class:QualityReport. ROUGE fields are None when no reference
        is provided. length_tokens is None when no tokenizer is given.
    """
    length_chars = len(summary)
    length_tokens: int | None = None
    if tokenizer is not None:
        try:
            length_tokens = len(tokenizer.encode(summary, add_special_tokens=False))
        except TypeError:
            length_tokens = len(tokenizer.encode(summary))

    rouge1 = rouge2 = rougeL = None
    if reference is not None:
        scores = compute_rouge(summary, reference)
        rouge1 = scores["rouge1"]
        rouge2 = scores["rouge2"]
        rougeL = scores["rougeL"]

    return QualityReport(
        summary=summary,
        reference=reference,
        length_chars=length_chars,
        length_tokens=length_tokens,
        rouge1=rouge1,
        rouge2=rouge2,
        rougeL=rougeL,
    )


__all__ = ["QualityReport", "compute_rouge", "evaluate_quality"]
