"""Detect when the decoder is generating a factual span.

The point of a factual-span detector is to turn the source-grounded constraint
*on* only at high-hallucination-risk timesteps — typically immediately before
the model emits a named entity, a number, or a date — and leave the model
otherwise unconstrained. This is the difference between the "fully
constrained" baseline (constrain every step → grounded but degraded) and the
selective variant the research question is really about.

This module implements a heuristic detector 
"""

from __future__ import annotations

import re
from typing import Iterable, Protocol, Sequence


class DecodableTokenizer(Protocol):
    """Minimal tokenizer protocol: we just need ``decode``."""

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = ...) -> str:
        ...


# Defaults

# Trigger phrases — words/phrases whose appearance in the current clause
# suggests the rest of the clause will contain a fact (entity, date, number).
DEFAULT_TRIGGER_PHRASES: tuple[str, ...] = (
    "said that",
    "said",
    "told",
    "told reporters",
    "reported",
    "announced",
    "according to",
    "stated",
    "claimed",
    "confirmed",
    "denied",
    "warned",
    "added",
    "wrote",
    "tweeted",
    "noted",
    "explained",
    "worth",
    "valued at",
    "priced at",
    "estimated at",
    "amounted to",
    "named",
    "called",
    "born in",
    "born on",
    "based in",
    "headquartered in",
    "located in",
    "founded in",
    "founded by",
)

# Last-word preceders: prepositions and titles that *typically* sit
# immediately before a named entity. Deliberately excludes the determiners
# "the/a/an" — these precede entities sometimes but precede ordinary nouns
# overwhelmingly more often, so including them produced an over-eager
# detector during development.
DEFAULT_PRECEDER_WORDS: frozenset[str] = frozenset(
    {
        # prepositions
        "in", "at", "of", "from", "to", "by", "near", "with", "on", "into",
        "onto", "across", "between", "among", "via",
        # honorifics / titles (often appear right before a personal name)
        "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "lord", "lady",
        "president", "ceo", "chairman", "chairwoman", "minister", "premier",
        "judge", "officer", "captain", "general", "colonel", "lieutenant",
    }
)

DEFAULT_CURRENCY_TRIGGERS: tuple[str, ...] = ("$", "£", "€", "¥")

DEFAULT_TERMINATOR_CHARS: tuple[str, ...] = (".", ";", "!", "?", ":", ",", "\n")

# Conjunctions that mark a clause break for deactivation
DEFAULT_TERMINATOR_WORDS: frozenset[str] = frozenset({"and", "or", "but"})


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class HeuristicFactualSpanDetector:
    """Heuristic detector for factual spans during generation.

    Stateless across timesteps: every call inspects only the input text/IDs.

    Args:
        trigger_phrases: Phrases whose appearance in the current clause
            activates constraints. Lower-cased internally.
        preceder_words: Words that activate constraints when they are the
            last token of the current clause (typically prepositions and
            honorifics).
        currency_triggers: Single-character cues that activate constraints
            when they end the current clause (the model is about to emit
            a monetary amount).
        terminator_chars: Characters that close the current clause.
        terminator_words: Whole-word terminators (conjunctions).
        lookback_chars: Maximum number of characters from the end of the
            generated text to consider. Bounds the cost of running the
            detector on very long outputs.
    """

    def __init__(
        self,
        trigger_phrases: Iterable[str] | None = None,
        preceder_words: Iterable[str] | None = None,
        currency_triggers: Iterable[str] | None = None,
        terminator_chars: Iterable[str] | None = None,
        terminator_words: Iterable[str] | None = None,
        lookback_chars: int = 128,
    ) -> None:
        if lookback_chars <= 0:
            raise ValueError(f"lookback_chars must be positive; got {lookback_chars}")

        self.trigger_phrases: tuple[str, ...] = tuple(
            t.lower()
            for t in (
                trigger_phrases
                if trigger_phrases is not None
                else DEFAULT_TRIGGER_PHRASES
            )
        )
        self.preceder_words: frozenset[str] = frozenset(
            w.lower()
            for w in (
                preceder_words
                if preceder_words is not None
                else DEFAULT_PRECEDER_WORDS
            )
        )
        self.currency_triggers: tuple[str, ...] = tuple(
            currency_triggers
            if currency_triggers is not None
            else DEFAULT_CURRENCY_TRIGGERS
        )
        self.terminator_chars: tuple[str, ...] = tuple(
            terminator_chars
            if terminator_chars is not None
            else DEFAULT_TERMINATOR_CHARS
        )
        self.terminator_words: frozenset[str] = frozenset(
            w.lower()
            for w in (
                terminator_words
                if terminator_words is not None
                else DEFAULT_TERMINATOR_WORDS
            )
        )
        self.lookback_chars: int = lookback_chars

        # Pre-compile conjunction regex once.
        if self.terminator_words:
            pattern = r"\b(?:" + "|".join(
                re.escape(w) for w in self.terminator_words
            ) + r")\b"
            self._terminator_word_re: re.Pattern[str] | None = re.compile(pattern)
        else:
            self._terminator_word_re = None

    # public API

    def reset(self) -> None:
        """No-op for the stateless heuristic; reserved for future detectors."""
        return None

    def is_active_from_text(self, text: str) -> bool:
        """Return True if constraints should fire for the *next* token,
        given the text generated so far.

        Pure string heuristic so does not depend on a tokenizer or model.
        """
        if not text:
            return False

        snippet = text[-self.lookback_chars :].lower()
        span_start = self._span_start_index(snippet)
        span = snippet[span_start:]

        if self._ends_with_currency(span):
            return True
        if self._contains_trigger_phrase(span):
            return True
        if self._last_word_is_preceder(span):
            return True
        return False

    def is_active(
        self,
        token_ids: Sequence[int],
        tokenizer: DecodableTokenizer,
    ) -> bool:
        """Decode token_ids via tokenizer and run the heuristic."""
        if not token_ids:
            return False
        text = tokenizer.decode(list(token_ids), skip_special_tokens=False)
        return self.is_active_from_text(text)

    # -- internals -----------------------------------------------------------

    def _span_start_index(self, snippet: str) -> int:
        """Index in snippet after the most recent terminator (char or word)."""
        last_term = -1
        for ch in self.terminator_chars:
            idx = snippet.rfind(ch)
            if idx > last_term:
                last_term = idx

        if self._terminator_word_re is not None:
            for match in self._terminator_word_re.finditer(snippet):
                end = match.end() - 1  # treat the conjunction end as a terminator pos
                if end > last_term:
                    last_term = end

        return last_term + 1

    def _ends_with_currency(self, span: str) -> bool:
        stripped = span.rstrip()
        if not stripped:
            return False
        return any(stripped.endswith(sym) for sym in self.currency_triggers)

    def _contains_trigger_phrase(self, span: str) -> bool:
        for trigger in self.trigger_phrases:
            # Word-boundary match so "report" doesn't fire on "reporter" — but
            # phrases with spaces ("said that") don't have a trailing \b
            # problem.
            if re.search(rf"\b{re.escape(trigger)}\b", span):
                return True
        return False

    def _last_word_is_preceder(self, span: str) -> bool:
        words = re.findall(r"[A-Za-z']+", span)
        if not words:
            return False
        return words[-1] in self.preceder_words


__all__ = [
    "DecodableTokenizer",
    "HeuristicFactualSpanDetector",
    "DEFAULT_TRIGGER_PHRASES",
    "DEFAULT_PRECEDER_WORDS",
    "DEFAULT_CURRENCY_TRIGGERS",
    "DEFAULT_TERMINATOR_CHARS",
    "DEFAULT_TERMINATOR_WORDS",
]
