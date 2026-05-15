"""HuggingFace ``LogitsProcessor`` variants for source-grounded constrained decoding.

Three process are provided that share mask-construction and caching machinery; the selective
variants add a per-step gate plus diagnostic counters.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from transformers.generation.logits_process import LogitsProcessor

from .constraint_builder import GroundedConstraint
from .detector import DecodableTokenizer, HeuristicFactualSpanDetector


class FlatAllowlistLogitsProcessor(LogitsProcessor):
    """Mask logits to a flat token allowlist at every decoding step.

    Implements the "fully constrained" baseline.


    Args:
        allowed_token_ids: Iterable of vocabulary IDs permitted in the output.
            Must be non-empty.
        penalty: Amount to subtract from disallowed-token scores. The
            default (inf) gives a hard mask via masked_fill; a finite
            positive value gives a soft penalty, preserving the relative
            ordering of disallowed tokens.

    Notes:
        * The same allowlist is applied to every element of the batch. Per-
          example allowlists will require a different processor (planned).
        * The vocab-sized mask is cached after the first call and reused while
          scores.shape[-1] and scores.device stay the same.
    """

    def __init__(
        self,
        allowed_token_ids: Iterable[int],
        penalty: float = float("inf"),
    ) -> None:
        ids = sorted({int(t) for t in allowed_token_ids})
        if not ids:
            raise ValueError("allowed_token_ids must be non-empty")
        if min(ids) < 0:
            raise ValueError(f"allowed_token_ids must be non-negative; got {min(ids)}")
        if penalty < 0:
            raise ValueError(f"penalty must be non-negative; got {penalty}")
        self._allowed_ids: list[int] = ids
        self._penalty: float = float(penalty)
        self._mask_cache: torch.Tensor | None = None
        self._mask_vocab: int | None = None
        self._mask_device: torch.device | None = None

    @classmethod
    def from_constraint(
        cls,
        constraint: GroundedConstraint,
        penalty: float = float("inf"),
    ) -> "FlatAllowlistLogitsProcessor":
        """Build a processor directly from a :class:`GroundedConstraint`."""
        return cls(constraint.allowlist.token_ids, penalty=penalty)

    @property
    def num_allowed(self) -> int:
        return len(self._allowed_ids)

    @property
    def is_hard_mask(self) -> bool:
        """True iff disallowed tokens are forced to ``-inf``."""
        return math.isinf(self._penalty)

    @property
    def penalty(self) -> float:
        return self._penalty

    def _build_mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        """Return a boolean mask of shape (vocab_size,) where True == BLOCK."""
        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        allowed_idx = torch.tensor(self._allowed_ids, dtype=torch.long, device=device)
        allowed_idx = allowed_idx[allowed_idx < vocab_size]
        if allowed_idx.numel() == 0:
            raise ValueError(
                f"No allowed token IDs fit within vocab_size={vocab_size}; "
                f"max allowed id was {max(self._allowed_ids)}"
            )
        mask[allowed_idx] = False
        return mask

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        vocab_size = scores.shape[-1]
        if (
            self._mask_cache is None
            or self._mask_vocab != vocab_size
            or self._mask_device != scores.device
        ):
            self._mask_cache = self._build_mask(vocab_size, scores.device)
            self._mask_vocab = vocab_size
            self._mask_device = scores.device
        mask = self._mask_cache
        if self.is_hard_mask:
            # masked_fill broadcasts (vocab,) over (batch, vocab).
            return scores.masked_fill(mask, -float("inf"))
        # Subtract the penalty from disallowed positions; allowed scores are
        # untouched and the relative ordering among disallowed tokens is
        # preserved.
        return scores - self._penalty * mask.to(scores.dtype)


class SelectiveLogitsProcessor(LogitsProcessor):
    """Apply a flat source-grounded allowlist only during detected factual spans.

    Args:
        allowed_token_ids: Vocabulary IDs permitted inside factual spans.
        detector: Object exposing
            ``is_active(token_ids: Sequence[int], tokenizer) -> bool``.
        tokenizer: Tokenizer with a ``decode`` method; passed to the detector.
        penalty: Forwarded to the underlying flat mask.
            ``inf`` (default) → hard mask; finite positive value → soft penalty.

    Notes:
        Currently supports batch size 1 only — selective constraining needs
        a per-example active flag, which would require a more involved
        mask construction. The fully-constrained baseline supports batching;
        this one will be extended when the experiment driver needs it.
    """

    def __init__(
        self,
        allowed_token_ids: Iterable[int],
        detector: HeuristicFactualSpanDetector,
        tokenizer: DecodableTokenizer,
        penalty: float = float("inf"),
    ) -> None:
        self._inner = FlatAllowlistLogitsProcessor(
            allowed_token_ids, penalty=penalty
        )
        self._detector = detector
        self._tokenizer = tokenizer
        self._prompt_length: int | None = None
        self._prompt_fingerprint: torch.Tensor | None = None
        self._n_steps: int = 0
        self._n_constrained: int = 0

    @classmethod
    def from_constraint(
        cls,
        constraint: GroundedConstraint,
        detector: HeuristicFactualSpanDetector,
        tokenizer: DecodableTokenizer,
        penalty: float = float("inf"),
    ) -> "SelectiveLogitsProcessor":
        return cls(
            constraint.allowlist.token_ids,
            detector=detector,
            tokenizer=tokenizer,
            penalty=penalty,
        )

    # -- state 

    def reset(self) -> None:
        """Forget prompt length and counters; call between generations."""
        self._prompt_length = None
        self._prompt_fingerprint = None
        self._n_steps = 0
        self._n_constrained = 0
        self._detector.reset()

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def n_constrained(self) -> int:
        return self._n_constrained

    @property
    def fraction_constrained(self) -> float:
        """Proportion of generation steps at which the mask was applied."""
        return self._n_constrained / self._n_steps if self._n_steps else 0.0

    @property
    def is_hard_mask(self) -> bool:
        return self._inner.is_hard_mask

    @property
    def num_allowed(self) -> int:
        return self._inner.num_allowed

    # -- core ---------------------------------------------------------------

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.shape[0] != 1:
            raise NotImplementedError(
                "SelectiveLogitsProcessor currently supports batch_size=1 only; "
                f"got batch_size={input_ids.shape[0]}"
            )

        cur_len = int(input_ids.shape[1])
        # A new generation is detected when (a) we've never been called,
        # (b) the new input is shorter than the previously stored prompt, or
        # (c) the new input's prefix doesn't match the stored prompt
        # fingerprint. Comparing the prefix is robust to the edge case where
        # the new prompt happens to be the same length as the previous final
        # output — the length-delta heuristic mishandles that.
        new_generation = (
            self._prompt_length is None
            or self._prompt_fingerprint is None
            or cur_len < self._prompt_length
            or not torch.equal(
                input_ids[0, : self._prompt_length],
                self._prompt_fingerprint,
            )
        )
        if new_generation:
            self._prompt_length = cur_len
            self._prompt_fingerprint = input_ids[0, :cur_len].detach().clone()
            self._n_steps = 0
            self._n_constrained = 0
            self._detector.reset()

        assert self._prompt_length is not None  # set above
        generated_ids = input_ids[0, self._prompt_length :].tolist()
        active = self._detector.is_active(generated_ids, self._tokenizer)

        self._n_steps += 1
        if active:
            self._n_constrained += 1
            return self._inner(input_ids, scores)
        return scores


class TrieSelectiveLogitsProcessor(LogitsProcessor):
    """Trie-gated selective constrained-decoding processor.

    When the detector says we are entering a factual span, the processor
    initialises a live set on the root of a prefix trie built from the source's
    tokenised entities and numbers. On each subsequent step it advances the
    live set with the most recently emitted token; the allowed-next set is the
    union of trie children of the live nodes. With hard masking this forces
    multi-token entities ("Lloyds Banking Group", "£2.1 billion", "200,000") to
    be reproduced verbatim from the source, fixing the failure mode of the
    flat-allowlist :class:`SelectiveLogitsProcessor` where each subword token
    is checked independently.

    Args:
        constraint: A :class:`GroundedConstraint` whose trie field is the
            tokenised-entity trie. The flat allowlist is **not** used by this
            processor — when active we constrain *strictly* to trie
            continuations.
        detector: Factual-span detector with the same is_active/reset
            protocol used by :class:SelectiveLogitsProcessor.
        tokenizer: Tokenizer with a decode method; passed to the detector.
        penalty: inf (default) → hard mask; finite positive value → soft
            penalty subtracted from disallowed token scores. Hard mode is the
            faithful-by-construction setting; soft mode is for sweeps that
            characterise the faithfulness/quality trade-off.

    Notes:
        Currently supports batch size 1 only (same constraint as
        :class:`SelectiveLogitsProcessor` — selective constraining is
        per-example by nature).
    """

    def __init__(
        self,
        constraint: GroundedConstraint,
        detector: HeuristicFactualSpanDetector,
        tokenizer: DecodableTokenizer,
        penalty: float = float("inf"),
    ) -> None:
        if penalty < 0:
            raise ValueError(f"penalty must be non-negative; got {penalty}")
        self._constraint = constraint
        self._detector = detector
        self._tokenizer = tokenizer
        self._penalty = float(penalty)

        self._prompt_length: int | None = None
        self._prompt_fingerprint: torch.Tensor | None = None
        self._live: set = set()
        self._n_steps: int = 0
        self._n_constrained: int = 0
        self._n_entities_started: int = 0
        self._n_entities_completed: int = 0

    # -- state 

    def reset(self) -> None:
        """Forget prompt length, live trie state, and counters."""
        self._prompt_length = None
        self._prompt_fingerprint = None
        self._live = set()
        self._n_steps = 0
        self._n_constrained = 0
        self._n_entities_started = 0
        self._n_entities_completed = 0
        self._detector.reset()

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def n_constrained(self) -> int:
        return self._n_constrained

    @property
    def fraction_constrained(self) -> float:
        return self._n_constrained / self._n_steps if self._n_steps else 0.0

    @property
    def n_entities_started(self) -> int:
        return self._n_entities_started

    @property
    def n_entities_completed(self) -> int:
        return self._n_entities_completed

    @property
    def is_hard_mask(self) -> bool:
        return math.isinf(self._penalty)

    @property
    def penalty(self) -> float:
        return self._penalty

    # -- core 

    def _apply_mask(
        self, scores: torch.FloatTensor, allowed: set[int]
    ) -> torch.FloatTensor:
        vocab_size = scores.shape[-1]
        device = scores.device
        # True == BLOCK. Build a boolean mask once per step; sets are tiny.
        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        allowed_in_range = [t for t in allowed if 0 <= t < vocab_size]
        if not allowed_in_range:
            # Every allowed continuation is out of vocab — passthrough rather
            # than wedge the decoder with -inf everywhere.
            return scores
        idx = torch.tensor(allowed_in_range, dtype=torch.long, device=device)
        mask[idx] = False
        if math.isinf(self._penalty):
            return scores.masked_fill(mask, -float("inf"))
        return scores - self._penalty * mask.to(scores.dtype)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.shape[0] != 1:
            raise NotImplementedError(
                "TrieSelectiveLogitsProcessor supports batch_size=1 only; "
                f"got batch_size={input_ids.shape[0]}"
            )

        cur_len = int(input_ids.shape[1])
        new_generation = (
            self._prompt_length is None
            or self._prompt_fingerprint is None
            or cur_len < self._prompt_length
            or not torch.equal(
                input_ids[0, : self._prompt_length],
                self._prompt_fingerprint,
            )
        )
        if new_generation:
            self._prompt_length = cur_len
            self._prompt_fingerprint = input_ids[0, :cur_len].detach().clone()
            self._live = set()
            self._n_steps = 0
            self._n_constrained = 0
            self._n_entities_started = 0
            self._n_entities_completed = 0
            self._detector.reset()

        assert self._prompt_length is not None
        generated_ids = input_ids[0, self._prompt_length :].tolist()

        trie = self._constraint.trie

        # If we were inside an entity, advance the live set by the last
        # emitted token. A terminal node means the entity has been completed;
        # we deactivate per the design (caller's request) and let the detector
        # re-fire on a *subsequent* step (an always-active detector must not
        # immediately re-seed in the same call — that would defeat the
        # "deactivate on terminal" semantic).
        # Took me so sodding long to get this right godamnit
        just_completed = False
        if self._live and generated_ids:
            last_tok = int(generated_ids[-1])
            advanced = trie.step(self._live, last_tok)
            if advanced and trie.any_terminal(advanced):
                self._live = set()
                self._n_entities_completed += 1
                just_completed = True
            else:
                self._live = advanced  # may be empty if soft mode went off-trie

        # 2. Decide whether constraints fire this step.
        active = self._detector.is_active(generated_ids, self._tokenizer)

        # 3. Enter a fresh factual span if active and not currently inside one,
        # but never on the same step that completed an entity — give the
        # decoder one free step before reactivating.
        if active and not self._live and not just_completed:
            start_tokens = trie.start_tokens
            if start_tokens:
                self._live = trie.new_run()
                self._n_entities_started += 1
            else:
                # No entities in the source so nothing to constrain to. Pass
                # through rather than block every token.
                active = False

        self._n_steps += 1

        if not active or not self._live:
            return scores

        allowed = trie.allowed_next(self._live)
        if not allowed:
            # Trie exhausted without a terminal (shouldn't happen for a
            # well-formed trie, but be defensive). Drop out.
            self._live = set()
            return scores

        self._n_constrained += 1
        return self._apply_mask(scores, allowed)


__all__ = [
    "FlatAllowlistLogitsProcessor",
    "SelectiveLogitsProcessor",
    "TrieSelectiveLogitsProcessor",
]
