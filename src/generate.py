"""End-to-end generation pipeline: run all decoding conditions on one source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from transformers import LogitsProcessorList

from .constraint_builder import GroundedConstraint, build_constraint
from .detector import HeuristicFactualSpanDetector
from .entity_extractor import SourceFacts, extract_facts
from .grounded_logits_processor import (
    FlatAllowlistLogitsProcessor,
    TrieSelectiveLogitsProcessor,
)


CONDITION_NAMES: tuple[str, ...] = (
    "unconstrained",
    "fully_constrained_hard",
    "selective_hard",
    "selective_soft",
)


def soft_condition_name(penalty: float) -> str:
    """Canonical condition name for a soft-penalty sweep entry."""
    if float(penalty).is_integer():
        return f"selective_soft_p{int(penalty)}"
    # e.g. 2.5 -> "selective_soft_p2_5" (no dots in identifier-like names)
    return f"selective_soft_p{str(penalty).replace('.', '_')}"


def expand_condition_names(
    soft_penalty_sweep: Sequence[float] | None,
) -> tuple[str, ...]:
    """Return the full ordered condition list given an optional sweep.

    Without a sweep this returns :data:`CONDITION_NAMES` unchanged. With a
    sweep the single selective_soft slot is replaced by the per-penalty
    sweep variants, preserving the order of the input.
    """
    if not soft_penalty_sweep:
        return CONDITION_NAMES
    head = (
        "unconstrained",
        "fully_constrained_hard",
        "selective_hard",
    )
    return head + tuple(soft_condition_name(p) for p in soft_penalty_sweep)


@dataclass
class ConditionalSummary:
    """A single (condition, summary) result for one source document."""

    name: str
    text: str
    new_token_ids: list[int]
    fraction_constrained: float | None = None  # None for non-selective conditions

    @property
    def n_new_tokens(self) -> int:
        return len(self.new_token_ids)


def _default_extra_token_ids(tokenizer: Any) -> list[int]:
    """Tokens that should always be permitted in the flat allowlist.

    EOS + a handful of common structural characters. Without these the
    fully-constrained baseline can't terminate or produce sentence structure.
    """
    extras: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        extras.add(int(eos))
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        extras.add(int(bos))
    for piece in (".", ",", ";", ":", "!", "?", " ", "\n", "-", "'", '"'):
        try:
            for tok_id in tokenizer.encode(piece, add_special_tokens=False):
                extras.add(int(tok_id))
        except Exception:
            continue
    return sorted(extras)


def _greedy_generate(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    processors: LogitsProcessorList,
    max_new_tokens: int,
) -> tuple[str, list[int]]:
    """Run greedy generation with the given logits processors.

    Returns (decoded_continuation, new_token_ids).
    """
    prompt_len = inputs["input_ids"].shape[1]
    pad_token_id = (
        tokenizer.pad_token_id
        if getattr(tokenizer, "pad_token_id", None) is not None
        else getattr(tokenizer, "eos_token_id", None)
    )
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=processors,
            pad_token_id=pad_token_id,
        )
    new_ids = out[0, prompt_len:].tolist()
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
    return decoded, new_ids


def generate_summaries(
    source: str,
    prompt: str,
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 64,
    soft_penalty: float = 5.0,
    soft_penalty_sweep: Sequence[float] | None = None,
    extra_token_ids: Iterable[int] | None = None,
    detector: HeuristicFactualSpanDetector | None = None,
    constraint: GroundedConstraint | None = None,
    source_facts: SourceFacts | None = None,
    spacy_model: str = "en_core_web_sm",
) -> dict[str, ConditionalSummary]:
    """Run the decoding conditions on one source document.

    Args:
        source: Source document to be summarised.
        prompt: Full text prompt fed to the model. Construction is the
            caller's responsibility (chat template, instruction format, …).
        model: A HuggingFace causal LM.
        tokenizer: The model's tokenizer.
        max_new_tokens: Max tokens to generate per condition.
        soft_penalty: Penalty subtracted from disallowed-token scores in the
            selective_soft condition (inf would make it identical to
            selective_hard). Ignored if soft_penalty_sweep is given.
        soft_penalty_sweep: Optional list of soft penalties. When provided, the
            single selective_soft condition is replaced by one condition
            per value (e.g. selective_soft_p2, selective_soft_p5, …).
            Use a multi-value sweep to characterise the faithfulness/quality
            curve; the single-value default is sufficient for spot checks.
        extra_token_ids: Additional token IDs to include in the flat allowlist
            used by the fully-constrained baseline — EOS and common
            punctuation by default. Not used by the trie-gated selective
            conditions, which constrain strictly to trie continuations.
        detector: Override the heuristic detector (e.g. for a custom trigger
            list). Default :class:HeuristicFactualSpanDetector.
        constraint: Override the constraint construction. Useful when the
            same source is used many times.
        source_facts: Pre-computed source facts. Avoids re-running spaCy.
        spacy_model: spaCy pipeline used if source_facts is None.

    Returns:
        Dict keyed by condition name, each containing a :class:`ConditionalSummary`.
    """
    if constraint is None:
        if source_facts is None:
            source_facts = extract_facts(
                source, tokenizer=tokenizer, spacy_model=spacy_model
            )
        extras = (
            list(extra_token_ids)
            if extra_token_ids is not None
            else _default_extra_token_ids(tokenizer)
        )
        constraint = build_constraint(source_facts, extra_token_ids=extras)

    if detector is None:
        detector = HeuristicFactualSpanDetector()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    results: dict[str, ConditionalSummary] = {}

    # 1. Unconstrained
    text, ids = _greedy_generate(
        model, tokenizer, inputs, LogitsProcessorList(), max_new_tokens
    )
    results["unconstrained"] = ConditionalSummary(
        name="unconstrained", text=text, new_token_ids=ids
    )

    # 2. Fully constrained, hard mask (flat allowlist)
    full_proc = FlatAllowlistLogitsProcessor.from_constraint(constraint)
    text, ids = _greedy_generate(
        model, tokenizer, inputs, LogitsProcessorList([full_proc]), max_new_tokens
    )
    results["fully_constrained_hard"] = ConditionalSummary(
        name="fully_constrained_hard", text=text, new_token_ids=ids
    )

    # 3. Selective, trie-gated hard mask
    sel_hard = TrieSelectiveLogitsProcessor(
        constraint, detector=detector, tokenizer=tokenizer
    )
    text, ids = _greedy_generate(
        model, tokenizer, inputs, LogitsProcessorList([sel_hard]), max_new_tokens
    )
    results["selective_hard"] = ConditionalSummary(
        name="selective_hard",
        text=text,
        new_token_ids=ids,
        fraction_constrained=sel_hard.fraction_constrained,
    )

    # 4. Selective, trie-gated soft penalty (single value or sweep)
    sweep: tuple[tuple[str, float], ...]
    if soft_penalty_sweep:
        sweep = tuple(
            (soft_condition_name(p), float(p)) for p in soft_penalty_sweep
        )
    else:
        sweep = (("selective_soft", float(soft_penalty)),)

    for name, penalty in sweep:
        # A fresh detector per condition so its (no-op) state can't leak.
        cond_detector = (
            detector
            if len(sweep) == 1
            else HeuristicFactualSpanDetector(
                trigger_phrases=detector.trigger_phrases,
                preceder_words=detector.preceder_words,
                currency_triggers=detector.currency_triggers,
                terminator_chars=detector.terminator_chars,
                terminator_words=detector.terminator_words,
                lookback_chars=detector.lookback_chars,
            )
        )
        sel_soft = TrieSelectiveLogitsProcessor(
            constraint,
            detector=cond_detector,
            tokenizer=tokenizer,
            penalty=penalty,
        )
        text, ids = _greedy_generate(
            model,
            tokenizer,
            inputs,
            LogitsProcessorList([sel_soft]),
            max_new_tokens,
        )
        results[name] = ConditionalSummary(
            name=name,
            text=text,
            new_token_ids=ids,
            fraction_constrained=sel_soft.fraction_constrained,
        )

    return results


__all__ = [
    "ConditionalSummary",
    "CONDITION_NAMES",
    "expand_condition_names",
    "generate_summaries",
    "soft_condition_name",
]
