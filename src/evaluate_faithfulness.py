"""Faithfulness metrics for source-grounded summarisation.

Given a generated summary and the source document it was meant to summarise,
this module quantifies how much the summary stays faithful to the source.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .entity_extractor import (
    NUMERIC_ENTITY_LABELS,
    SourceFacts,
    _NUMBER_REGEX,
    _parse_number_value,
    extract_facts,
)


_NUMERIC_TOLERANCE: float = 1e-9

# Entity labels we expect to contain numeric values worth matching against source.
_NUMERIC_LABELS_FOR_EVAL: frozenset[str] = NUMERIC_ENTITY_LABELS | frozenset(
    {"DATE", "TIME"}
)


# Report


@dataclass
class FaithfulnessReport:
    """Faithfulness metrics for a single (summary, source) pair."""

    summary: str
    source: str
    summary_entities: list[str] = field(default_factory=list)
    source_entities: list[str] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    hallucinated_entities: list[str] = field(default_factory=list)
    recalled_entities: list[str] = field(default_factory=list)
    summary_numbers: list[str] = field(default_factory=list)
    source_numbers: list[str] = field(default_factory=list)
    matched_numbers: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)

    @property
    def entity_precision(self) -> float:
        """Of summary entities, fraction supported by the source.

        Conventionally 1.0 when the summary contains no entities (nothing
        to be wrong about). The hallucinated-entity count is the more
        informative signal in that case.
        """
        n = len(self.summary_entities)
        if n == 0:
            return 1.0
        return len(self.matched_entities) / n

    @property
    def entity_recall(self) -> float:
        """Of source entities, fraction that appear in the summary."""
        n = len(self.source_entities)
        if n == 0:
            return 1.0
        return len(self.recalled_entities) / n

    @property
    def entity_f1(self) -> float:
        p, r = self.entity_precision, self.entity_recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def number_accuracy(self) -> float:
        """Of summary numbers, fraction whose value is present in the source."""
        n = len(self.summary_numbers)
        if n == 0:
            return 1.0
        return len(self.matched_numbers) / n

    @property
    def hallucinated_entity_count(self) -> int:
        return len(self.hallucinated_entities)

    @property
    def unsupported_number_count(self) -> int:
        return len(self.unsupported_numbers)


# Matching helpers


def _normalise_entity(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_word_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _entity_appears_in_source(ent_text: str, source_lower: str) -> bool:
    """Return True if ent_text appears as a whole-word substring in source.

    Uses a manual boundary check rather than the regex ``\\b`` anchor because
    ``\\b`` only matches at word/non-word transitions, which fails for
    entities that start or end with a non-word character (the most common
    case being monetary entities like ""$5 billion"").
    """
    normalised = _normalise_entity(ent_text)
    if not normalised:
        return False
    n = len(source_lower)
    m = len(normalised)
    pos = 0
    while True:
        idx = source_lower.find(normalised, pos)
        if idx == -1:
            return False
        left = source_lower[idx - 1] if idx > 0 else " "
        right = source_lower[idx + m] if idx + m < n else " "
        left_ok = not (_is_word_char(normalised[0]) and _is_word_char(left))
        right_ok = not (_is_word_char(normalised[-1]) and _is_word_char(right))
        if left_ok and right_ok:
            return True
        pos = idx + 1


def _values_match(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=0.0, abs_tol=_NUMERIC_TOLERANCE)


def _collect_numeric_spans(facts: SourceFacts) -> list[tuple[str, float]]:
    """Return (surface, value) pairs for every numeric span in facts.

    Walks numeric-labelled entities (spaCy already classified them as
    MONEY/PERCENT/DATE/...) and runs the number regex over their text to
    pull out parseable values, then appends the regex-only number fallback.
    Deduplicates by (surface, value) so summaries that repeat the same
    fact aren't counted twice in the matched list.
    """
    out: list[tuple[str, float]] = []
    seen: set[tuple[str, float]] = set()
    for ent in facts.entities:
        if ent.label not in _NUMERIC_LABELS_FOR_EVAL:
            continue
        for match in _NUMBER_REGEX.finditer(ent.text):
            value = _parse_number_value(match.group())
            if value is None:
                continue
            key = (match.group(), value)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    for n in facts.numbers:
        if n.value is None:
            continue
        key = (n.text, n.value)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _number_supported(value: float | None, source_values: list[float]) -> bool:
    if value is None:
        return False
    return any(_values_match(value, v) for v in source_values)


# Public API


def evaluate_faithfulness(
    summary: str,
    source: str,
    *,
    summary_facts: SourceFacts | None = None,
    source_facts: SourceFacts | None = None,
    spacy_model: str = "en_core_web_sm",
) -> FaithfulnessReport:
    """Compare a summary to its source on entity and number faithfulness.

    Args:
        summary: Generated summary text.
        source: Source document the summary is meant to describe.
        summary_facts, source_facts: Optional pre-computed
            :class:SourceFacts for either side — pass these to avoid
            re-running spaCy when evaluating many summaries against the same
            source.
        spacy_model: spaCy pipeline used for NER if facts aren't provided.

    Returns:
        A :class: sFaithfulnessReport with per-metric values and the
        underlying entity/number lists for inspection.
    """
    if source_facts is None:
        source_facts = extract_facts(source, tokenizer=None, spacy_model=spacy_model)
    if summary_facts is None:
        summary_facts = extract_facts(summary, tokenizer=None, spacy_model=spacy_model)

    source_lower = source.lower()
    summary_lower = summary.lower()

    summary_entity_texts = [e.text for e in summary_facts.entities]
    source_entity_texts = [e.text for e in source_facts.entities]

    matched: list[str] = []
    hallucinated: list[str] = []
    for text in summary_entity_texts:
        if _entity_appears_in_source(text, source_lower):
            matched.append(text)
        else:
            hallucinated.append(text)

    recalled: list[str] = [
        text for text in source_entity_texts
        if _entity_appears_in_source(text, summary_lower)
    ]

    source_numeric_spans = _collect_numeric_spans(source_facts)
    summary_numeric_spans = _collect_numeric_spans(summary_facts)
    source_values = [v for (_, v) in source_numeric_spans]

    summary_number_texts: list[str] = []
    matched_numbers: list[str] = []
    unsupported_numbers: list[str] = []
    for surface, value in summary_numeric_spans:
        summary_number_texts.append(surface)
        if _number_supported(value, source_values):
            matched_numbers.append(surface)
        else:
            unsupported_numbers.append(surface)

    source_number_texts = [s for (s, _) in source_numeric_spans]

    return FaithfulnessReport(
        summary=summary,
        source=source,
        summary_entities=summary_entity_texts,
        source_entities=source_entity_texts,
        matched_entities=matched,
        hallucinated_entities=hallucinated,
        recalled_entities=recalled,
        summary_numbers=summary_number_texts,
        source_numbers=source_number_texts,
        matched_numbers=matched_numbers,
        unsupported_numbers=unsupported_numbers,
    )


__all__ = ["FaithfulnessReport", "evaluate_faithfulness"]
