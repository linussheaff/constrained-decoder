"""Extract factual spans (named entities and numbers) from source documents.

Given a source document and a tokenizer, this module produces a :class:`SourceFacts`
object summarising every span that the constrained decoder may later need to
recognise:

Designed to be lightweight and side-effect free: the spaCy model is loaded
lazily and cached, and the tokenizer is supplied by the caller.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Iterable, Protocol

if TYPE_CHECKING:  # pragma: no cover — typing only
    from spacy.language import Language

logger = logging.getLogger(__name__)


# spaCy entity labels we treat as "factual" — i.e. worth constraining to the
# source document during generation. 
FACTUAL_ENTITY_LABELS: frozenset[str] = frozenset(
    {
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "FAC",
        "NORP",
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "DATE",
        "TIME",
        "MONEY",
        "PERCENT",
        "QUANTITY",
        "CARDINAL",
        "ORDINAL",
    }
)

# Pure-numeric labels — useful for downstream filtering / weighting.
NUMERIC_ENTITY_LABELS: frozenset[str] = frozenset(
    {"MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL"}
)

# Permissive number pattern: matches "1", "1.5", "1,000", "1,000.50", "50%",
# "$100", "£1,200.50", optional leading sign..
_NUMBER_REGEX = re.compile(
    r"[-+]?[\$£€¥]?\d+(?:,\d{3})*(?:\.\d+)?%?"
)


class TokenizerLike(Protocol):
    """Minimal duck-typed interface we need from a tokenizer.

    Both HuggingFace ``PreTrainedTokenizerBase`` and any well-behaved stub
    satisfy this protocol.
    """

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]:
        ...


@dataclass(frozen=True)
class Entity:
    """A factual named-entity span in the source document."""

    text: str
    label: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class Number:
    """A numeric span in the source document.

    value is the parsed float when the span looks like a plain number
    (currency symbol and trailing percent stripped) and ``None`` otherwise.
    """

    text: str
    value: float | None
    start_char: int
    end_char: int


@dataclass
class SourceFacts:
    """Structured collection of factual spans extracted from a source document.

    Attributes:
        source_text: The original document text the facts were extracted from.
        entities: Named entities with factual labels.
        numbers: Numeric spans (regex fallback, deduplicated against entities).
        entity_tokens: Union of all token IDs appearing in any tokenised entity
            variant. Suitable for use as a token allowlist.
        number_tokens: Union of all token IDs appearing in any tokenised
            number variant.
        entity_token_sequences: Per-variant tokenised entity sequences; the raw
            material for building a multi-token prefix trie.
        number_token_sequences: Per-variant tokenised number sequences.
    """

    source_text: str
    entities: list[Entity]
    numbers: list[Number]
    entity_tokens: set[int] = field(default_factory=set)
    number_tokens: set[int] = field(default_factory=set)
    entity_token_sequences: list[list[int]] = field(default_factory=list)
    number_token_sequences: list[list[int]] = field(default_factory=list)

    @property
    def factual_token_ids(self) -> set[int]:
        """Union of all token IDs across entities and numbers."""
        return self.entity_tokens | self.number_tokens

    @property
    def all_spans(self) -> list[tuple[int, int, str]]:
        """All factual character spans as (start, end, kind) triples."""
        spans: list[tuple[int, int, str]] = [
            (e.start_char, e.end_char, "entity") for e in self.entities
        ]
        spans.extend((n.start_char, n.end_char, "number") for n in self.numbers)
        spans.sort()
        return spans


@lru_cache(maxsize=4)
def _load_spacy(model_name: str) -> "Language":
    """Load and cache a spaCy pipeline by name."""
    try:
        import spacy
    except ImportError as exc:  # no cover — import guard
        raise RuntimeError(
            "spaCy is required for entity extraction. Install with: "
            "pip install spacy && python -m spacy download en_core_web_sm"
        ) from exc

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. Install with: "
            f"python -m spacy download {model_name}"
        ) from exc


def _parse_number_value(text: str) -> float | None:
    """Best-effort parse of a numeric span to a float."""
    cleaned = text.strip()
    for sym in ("$", "£", "€", "¥"):
        cleaned = cleaned.replace(sym, "")
    cleaned = cleaned.replace(",", "")
    percent = cleaned.endswith("%")
    if percent:
        cleaned = cleaned[:-1]
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value / 100.0 if percent else value


def _tokenize_variants(
    text: str, tokenizer: TokenizerLike
) -> list[list[int]]:
    """Tokenise text in surface-form variants the model might emit."""
    if not text:
        return []

    variants = [text, " " + text]
    # Avoid duplicating if the input already starts with whitespace.
    if text.startswith(" "):
        variants = [text]

    sequences: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for variant in variants:
        try:
            ids = list(tokenizer.encode(variant, add_special_tokens=False))
        except TypeError:
            # Older tokenizer stubs may not accept the kwarg.
            ids = list(tokenizer.encode(variant))
        if not ids:
            continue
        key = tuple(ids)
        if key in seen:
            continue
        sequences.append(ids)
        seen.add(key)
    return sequences


def _regex_numbers(
    source: str, ignore_spans: Iterable[tuple[int, int]]
) -> list[Number]:
    """Find numeric spans by regex, skipping those already inside an entity."""
    ignore = sorted(ignore_spans)

    def inside_entity(start: int, end: int) -> bool:
        for s, e in ignore:
            if e <= start:
                continue
            if s >= end:
                break
            if s <= start and end <= e:
                return True
        return False

    numbers: list[Number] = []
    for match in _NUMBER_REGEX.finditer(source):
        start, end = match.start(), match.end()
        if inside_entity(start, end):
            continue
        text = match.group()
        # Reject matches that are purely a stray sign / currency with no digits.
        if not any(ch.isdigit() for ch in text):
            continue
        numbers.append(
            Number(
                text=text,
                value=_parse_number_value(text),
                start_char=start,
                end_char=end,
            )
        )
    return numbers


def extract_facts(
    source: str,
    tokenizer: TokenizerLike | None = None,
    spacy_model: str = "en_core_web_sm",
    include_numbers: bool = True,
    entity_labels: frozenset[str] | None = None,
) -> SourceFacts:
    """Extract entities and numbers from a source document."""
    if not isinstance(source, str):
        raise TypeError(f"source must be str, got {type(source).__name__}")

    labels = entity_labels if entity_labels is not None else FACTUAL_ENTITY_LABELS

    entities: list[Entity] = []
    if source.strip() and labels:
        nlp = _load_spacy(spacy_model)
        doc = nlp(source)
        for ent in doc.ents:
            if ent.label_ not in labels:
                continue
            entities.append(
                Entity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                )
            )

    numbers: list[Number] = []
    if include_numbers and source:
        ignore_spans = [(e.start_char, e.end_char) for e in entities]
        numbers = _regex_numbers(source, ignore_spans)

    entity_token_sequences: list[list[int]] = []
    entity_tokens: set[int] = set()
    number_token_sequences: list[list[int]] = []
    number_tokens: set[int] = set()
    if tokenizer is not None:
        for ent in entities:
            for seq in _tokenize_variants(ent.text, tokenizer):
                entity_token_sequences.append(seq)
                entity_tokens.update(seq)
        for n in numbers:
            for seq in _tokenize_variants(n.text, tokenizer):
                number_token_sequences.append(seq)
                number_tokens.update(seq)

    logger.debug(
        "Extracted %d entities, %d numbers from %d chars; %d unique factual token IDs",
        len(entities),
        len(numbers),
        len(source),
        len(entity_tokens | number_tokens),
    )

    return SourceFacts(
        source_text=source,
        entities=entities,
        numbers=numbers,
        entity_tokens=entity_tokens,
        number_tokens=number_tokens,
        entity_token_sequences=entity_token_sequences,
        number_token_sequences=number_token_sequences,
    )


__all__ = [
    "Entity",
    "Number",
    "SourceFacts",
    "TokenizerLike",
    "FACTUAL_ENTITY_LABELS",
    "NUMERIC_ENTITY_LABELS",
    "extract_facts",
]
