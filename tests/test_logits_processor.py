"""Tests for ``src.grounded_logits_processor.FlatAllowlistLogitsProcessor``.

Covers: hard masking, soft penalty, batched application, argmax-in-allowlist
invariant, mask caching, from_constraint helper, edge cases, and a tiny HF
model end-to-end integration test (gated on network access).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.constraint_builder import (  # noqa: E402  (import after importorskip)
    EntityTrie,
    GroundedConstraint,
    TokenAllowlist,
    build_constraint,
)
from src.detector import HeuristicFactualSpanDetector  # noqa: E402
from src.entity_extractor import SourceFacts  # noqa: E402
from src.grounded_logits_processor import (  # noqa: E402
    FlatAllowlistLogitsProcessor,
    SelectiveLogitsProcessor,
    TrieSelectiveLogitsProcessor,
)


# ---------------------------------------------------------------------------
# Hard masking
# ---------------------------------------------------------------------------


class TestHardMask:
    def test_disallowed_tokens_become_neg_inf(self) -> None:
        proc = FlatAllowlistLogitsProcessor([1, 3, 5])
        scores = torch.zeros(1, 10)
        input_ids = torch.zeros(1, 1, dtype=torch.long)
        out = proc(input_ids, scores)
        # Allowed positions stay at 0, blocked positions become -inf.
        for tok in range(10):
            if tok in {1, 3, 5}:
                assert out[0, tok].item() == 0.0, f"token {tok} should be allowed"
            else:
                assert math.isinf(out[0, tok].item()), f"token {tok} should be blocked"
                assert out[0, tok].item() < 0

    def test_argmax_always_in_allowlist(self) -> None:
        torch.manual_seed(0)
        proc = FlatAllowlistLogitsProcessor([2, 7])
        scores = torch.randn(8, 32)
        out = proc(torch.zeros(8, 1, dtype=torch.long), scores)
        chosen = out.argmax(dim=-1).tolist()
        assert all(c in {2, 7} for c in chosen), chosen

    def test_batched_application_uses_same_mask(self) -> None:
        proc = FlatAllowlistLogitsProcessor([0, 4])
        scores = torch.zeros(3, 6)
        out = proc(torch.zeros(3, 1, dtype=torch.long), scores)
        # All batch elements identical.
        assert torch.equal(out[0], out[1])
        assert torch.equal(out[1], out[2])
        # And the mask pattern is correct.
        for tok in range(6):
            allowed = tok in {0, 4}
            for b in range(3):
                if allowed:
                    assert out[b, tok].item() == 0.0
                else:
                    assert math.isinf(out[b, tok].item())

    def test_does_not_mutate_input_scores(self) -> None:
        proc = FlatAllowlistLogitsProcessor([1, 2])
        scores = torch.ones(1, 5)
        original = scores.clone()
        _ = proc(torch.zeros(1, 1, dtype=torch.long), scores)
        # masked_fill returns a new tensor, so the input should be untouched.
        assert torch.equal(scores, original)

    def test_is_hard_mask_property(self) -> None:
        assert FlatAllowlistLogitsProcessor([1]).is_hard_mask
        assert not FlatAllowlistLogitsProcessor([1], penalty=5.0).is_hard_mask


# ---------------------------------------------------------------------------
# Soft penalty variant
# ---------------------------------------------------------------------------


class TestSoftPenalty:
    def test_penalty_subtracts_from_disallowed_only(self) -> None:
        proc = FlatAllowlistLogitsProcessor([1], penalty=5.0)
        scores = torch.tensor([[2.0, 1.0, 3.0, 0.0]])
        out = proc(torch.zeros(1, 1, dtype=torch.long), scores)
        assert out[0, 1].item() == 1.0  # allowed → unchanged
        assert out[0, 0].item() == 2.0 - 5.0
        assert out[0, 2].item() == 3.0 - 5.0
        assert out[0, 3].item() == 0.0 - 5.0

    def test_relative_order_of_disallowed_preserved(self) -> None:
        # A finite penalty preserves the relative ordering of disallowed
        # tokens (unlike a hard mask, which collapses them all to -inf).
        proc = FlatAllowlistLogitsProcessor([1], penalty=2.0)
        scores = torch.tensor([[0.5, 0.0, -1.0, 1.5]])
        out = proc(torch.zeros(1, 1, dtype=torch.long), scores)
        disallowed_after = [out[0, i].item() for i in (0, 2, 3)]
        disallowed_before = [scores[0, i].item() for i in (0, 2, 3)]
        assert sorted(range(3), key=lambda i: disallowed_after[i]) == sorted(
            range(3), key=lambda i: disallowed_before[i]
        )

    def test_strong_score_can_beat_soft_penalty(self) -> None:
        # If a disallowed token has a much higher raw score than the penalty
        # can overcome, soft-mask decoding will still pick it. Confirms the
        # soft variant is *not* a hard mask.
        proc = FlatAllowlistLogitsProcessor([1], penalty=1.0)
        scores = torch.tensor([[0.0, -2.0, 10.0, 0.0]])  # token 2 dominates
        out = proc(torch.zeros(1, 1, dtype=torch.long), scores)
        assert out.argmax(dim=-1).item() == 2

    def test_negative_penalty_raises(self) -> None:
        with pytest.raises(ValueError):
            FlatAllowlistLogitsProcessor([1], penalty=-1.0)


# ---------------------------------------------------------------------------
# Mask caching
# ---------------------------------------------------------------------------


class TestMaskCaching:
    def test_mask_cached_across_calls(self) -> None:
        proc = FlatAllowlistLogitsProcessor([1, 2])
        scores = torch.zeros(1, 8)
        _ = proc(torch.zeros(1, 1, dtype=torch.long), scores)
        first_mask = proc._mask_cache
        assert first_mask is not None
        _ = proc(torch.zeros(1, 1, dtype=torch.long), scores)
        assert proc._mask_cache is first_mask

    def test_mask_rebuilt_when_vocab_size_changes(self) -> None:
        proc = FlatAllowlistLogitsProcessor([1, 2])
        _ = proc(torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 8))
        first_mask = proc._mask_cache
        _ = proc(torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 16))
        assert proc._mask_cache is not first_mask
        assert proc._mask_cache.shape == (16,)


# ---------------------------------------------------------------------------
# Construction edge cases
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_allowlist_raises(self) -> None:
        with pytest.raises(ValueError):
            FlatAllowlistLogitsProcessor([])

    def test_negative_token_id_raises(self) -> None:
        with pytest.raises(ValueError):
            FlatAllowlistLogitsProcessor([-1, 0, 1])

    def test_deduplicates_and_sorts_ids(self) -> None:
        proc = FlatAllowlistLogitsProcessor([3, 1, 3, 1, 2])
        assert proc._allowed_ids == [1, 2, 3]
        assert proc.num_allowed == 3

    def test_out_of_range_ids_are_dropped_silently(self) -> None:
        proc = FlatAllowlistLogitsProcessor([1, 999])
        scores = torch.zeros(1, 5)  # vocab_size=5 → id 999 ignored
        out = proc(torch.zeros(1, 1, dtype=torch.long), scores)
        # Token 1 still allowed; the rest blocked. No error raised.
        assert out[0, 1].item() == 0.0
        assert math.isinf(out[0, 0].item())

    def test_all_ids_out_of_range_raises(self) -> None:
        proc = FlatAllowlistLogitsProcessor([100, 200])
        with pytest.raises(ValueError):
            _ = proc(torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 8))


# ---------------------------------------------------------------------------
# from_constraint integration with the rest of the pipeline
# ---------------------------------------------------------------------------


class TestFromConstraint:
    def test_round_trip_from_grounded_constraint(self) -> None:
        facts = SourceFacts(
            source_text="",
            entities=[],
            numbers=[],
            entity_tokens={2, 4, 6},
            number_tokens={4, 8},
            entity_token_sequences=[[2, 4, 6]],
            number_token_sequences=[[4, 8]],
        )
        constraint = build_constraint(facts, extra_token_ids=[0])
        proc = FlatAllowlistLogitsProcessor.from_constraint(constraint)

        # The processor should permit exactly the constraint's allowlist.
        scores = torch.zeros(1, 10)
        out = proc(torch.zeros(1, 1, dtype=torch.long), scores)
        for tok in range(10):
            if tok in constraint.allowlist.token_ids:
                assert out[0, tok].item() == 0.0, f"{tok} should be allowed"
            else:
                assert math.isinf(out[0, tok].item()), f"{tok} should be blocked"


# ---------------------------------------------------------------------------
# End-to-end with a tiny HF model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_lm():
    """Load a tiny HF causal LM. Skips if the download/load fails."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "hf-internal-testing/tiny-random-gpt2"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForCausalLM.from_pretrained(name)
    except Exception as exc:
        pytest.skip(f"could not load tiny HF model {name}: {exc}")
    mdl.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, mdl


class TestEndToEndGeneration:
    def test_generated_tokens_all_in_allowlist(self, tiny_lm) -> None:
        from transformers import LogitsProcessorList

        tok, mdl = tiny_lm
        prompt = tok("Hello", return_tensors="pt")
        prompt_len = prompt["input_ids"].shape[1]

        # Allow a small handful of token IDs plus EOS so generation can terminate.
        allowed = {5, 10, 15, 20, 25, int(tok.eos_token_id)}
        proc = FlatAllowlistLogitsProcessor(allowed)

        out = mdl.generate(
            **prompt,
            max_new_tokens=8,
            do_sample=False,
            logits_processor=LogitsProcessorList([proc]),
            pad_token_id=tok.pad_token_id,
        )
        new_tokens = out[0, prompt_len:].tolist()
        # Every generated token must be in the allowlist (or EOS, which is in it).
        assert all(t in allowed for t in new_tokens), (new_tokens, allowed)

    def test_greedy_reproducible(self, tiny_lm) -> None:
        from transformers import LogitsProcessorList

        tok, mdl = tiny_lm
        prompt = tok("Hello", return_tensors="pt")
        allowed = list({5, 10, 15, int(tok.eos_token_id)})

        def run() -> list[int]:
            proc = FlatAllowlistLogitsProcessor(allowed)
            o = mdl.generate(
                **prompt,
                max_new_tokens=6,
                do_sample=False,
                logits_processor=LogitsProcessorList([proc]),
                pad_token_id=tok.pad_token_id,
            )
            return o[0].tolist()

        assert run() == run()


# ===========================================================================
# SelectiveLogitsProcessor
# ===========================================================================


class _AlwaysActive:
    """Stand-in detector that always returns True. No tokenizer needed."""

    def reset(self) -> None:
        return None

    def is_active(self, token_ids, tokenizer) -> bool:  # noqa: ARG002
        return True


class _NeverActive:
    """Stand-in detector that always returns False."""

    def reset(self) -> None:
        return None

    def is_active(self, token_ids, tokenizer) -> bool:  # noqa: ARG002
        return False


class _ResetSpy:
    """Records reset() calls so we can verify the processor forwards them."""

    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def is_active(self, token_ids, tokenizer) -> bool:  # noqa: ARG002
        return False


class TestSelectiveGate:
    def test_passes_scores_through_when_inactive(self, fake_tokenizer) -> None:
        proc = SelectiveLogitsProcessor(
            [1, 2], detector=_NeverActive(), tokenizer=fake_tokenizer
        )
        scores = torch.tensor([[0.0, 1.0, 0.5, 2.0]])
        out = proc(torch.tensor([[7, 8]]), scores)
        assert torch.equal(out, scores)
        assert proc.fraction_constrained == 0.0
        assert proc.n_steps == 1
        assert proc.n_constrained == 0

    def test_applies_mask_when_active(self, fake_tokenizer) -> None:
        proc = SelectiveLogitsProcessor(
            [1, 2], detector=_AlwaysActive(), tokenizer=fake_tokenizer
        )
        scores = torch.zeros(1, 5)
        out = proc(torch.tensor([[7]]), scores)
        # Tokens 1 and 2 allowed, the rest blocked.
        assert out[0, 1].item() == 0.0
        assert out[0, 2].item() == 0.0
        for t in (0, 3, 4):
            assert math.isinf(out[0, t].item())
        assert proc.n_steps == 1
        assert proc.n_constrained == 1
        assert proc.fraction_constrained == 1.0


class TestPromptLengthAndAutoReset:
    def test_prompt_length_captured_on_first_call(self, fake_tokenizer) -> None:
        proc = SelectiveLogitsProcessor(
            [1], detector=_NeverActive(), tokenizer=fake_tokenizer
        )
        proc(torch.zeros(1, 5, dtype=torch.long), torch.zeros(1, 4))
        assert proc._prompt_length == 5

    def test_continuing_generation_does_not_reset(self, fake_tokenizer) -> None:
        proc = SelectiveLogitsProcessor(
            [1], detector=_NeverActive(), tokenizer=fake_tokenizer
        )
        prompt = torch.tensor([[1, 2, 3, 4, 5]])
        proc(prompt, torch.zeros(1, 4))
        proc(torch.tensor([[1, 2, 3, 4, 5, 6]]), torch.zeros(1, 4))
        proc(torch.tensor([[1, 2, 3, 4, 5, 6, 7]]), torch.zeros(1, 4))
        assert proc.n_steps == 3
        assert proc._prompt_length == 5

    def test_shorter_input_resets_state(self, fake_tokenizer) -> None:
        proc = SelectiveLogitsProcessor(
            [1], detector=_NeverActive(), tokenizer=fake_tokenizer
        )
        proc(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), torch.zeros(1, 4))
        proc(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]), torch.zeros(1, 4))
        # Second generation with a shorter prompt.
        proc(torch.tensor([[20, 21, 22]]), torch.zeros(1, 4))
        assert proc._prompt_length == 3
        assert proc.n_steps == 1

    def test_different_prompt_same_length_resets(self, fake_tokenizer) -> None:
        # The previous failure mode: gen-2 prompt happens to have the same
        # length as gen-1's final input. The fingerprint catches it.
        proc = SelectiveLogitsProcessor(
            [1], detector=_NeverActive(), tokenizer=fake_tokenizer
        )
        proc(torch.tensor([[1, 2, 3]]), torch.zeros(1, 4))
        proc(torch.tensor([[1, 2, 3, 4]]), torch.zeros(1, 4))
        proc(torch.tensor([[1, 2, 3, 4, 5]]), torch.zeros(1, 4))
        # Now a fresh prompt of length 5 — must be detected as a new gen
        # even though length == previous final length.
        proc(torch.tensor([[9, 8, 7, 6, 5]]), torch.zeros(1, 4))
        assert proc._prompt_length == 5
        assert proc.n_steps == 1

    def test_explicit_reset(self, fake_tokenizer) -> None:
        spy = _ResetSpy()
        proc = SelectiveLogitsProcessor([1], detector=spy, tokenizer=fake_tokenizer)
        proc(torch.zeros(1, 5, dtype=torch.long), torch.zeros(1, 4))
        proc.reset()
        assert proc._prompt_length is None
        assert proc.n_steps == 0
        assert spy.reset_count >= 1

    def test_batched_input_raises(self, fake_tokenizer) -> None:
        proc = SelectiveLogitsProcessor(
            [1], detector=_NeverActive(), tokenizer=fake_tokenizer
        )
        with pytest.raises(NotImplementedError):
            proc(torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 4))


class TestDetectorReceivesGeneratedOnly:
    def test_detector_sees_only_generated_tokens(self, fake_tokenizer) -> None:
        seen: list[list[int]] = []

        class Recorder:
            def reset(self) -> None:
                pass

            def is_active(self, token_ids, tokenizer) -> bool:  # noqa: ARG002
                seen.append(list(token_ids))
                return False

        proc = SelectiveLogitsProcessor(
            [1], detector=Recorder(), tokenizer=fake_tokenizer
        )
        # First call: input = prompt only; detector should see [].
        prompt = torch.tensor([[100, 101, 102]])
        proc(prompt, torch.zeros(1, 5))
        # Second call: input = prompt + 1 generated token.
        proc(torch.tensor([[100, 101, 102, 200]]), torch.zeros(1, 5))
        # Third call: input = prompt + 2 generated tokens.
        proc(torch.tensor([[100, 101, 102, 200, 201]]), torch.zeros(1, 5))

        assert seen == [[], [200], [200, 201]]


class TestSelectiveWithHeuristicDetector:
    def test_passthrough_until_trigger_then_constrain(self, fake_tokenizer) -> None:
        det = HeuristicFactualSpanDetector()
        proc = SelectiveLogitsProcessor(
            [1, 2], detector=det, tokenizer=fake_tokenizer
        )

        # Prompt is 3 tokens, with no factual context in the generated prefix yet.
        prompt = torch.tensor([[10, 11, 12]])
        out = proc(prompt, torch.zeros(1, 6))
        # Generated prefix empty → detector inactive → passthrough.
        assert torch.equal(out, torch.zeros(1, 6))

        # Append "reported" so the detector should fire.
        trigger_ids = fake_tokenizer.encode(" reported", add_special_tokens=False)
        full_ids = torch.tensor([[10, 11, 12] + trigger_ids])
        # First grow the input by one token so the unit-step detector kicks in;
        # we'll fake intermediate calls.
        for k in range(1, len(trigger_ids) + 1):
            partial = torch.tensor([[10, 11, 12] + trigger_ids[:k]])
            out = proc(partial, torch.zeros(1, 6))

        # The last call had the full "reported" suffix → constraints on.
        assert math.isinf(out[0, 0].item())  # 0 not in allowlist
        assert out[0, 1].item() == 0.0
        assert proc.n_constrained >= 1


class TestSelectiveFromConstraint:
    def test_round_trip(self, fake_tokenizer) -> None:
        facts = SourceFacts(
            source_text="",
            entities=[],
            numbers=[],
            entity_tokens={3, 5},
            number_tokens={7},
            entity_token_sequences=[[3, 5]],
            number_token_sequences=[[7]],
        )
        constraint = build_constraint(facts)
        proc = SelectiveLogitsProcessor.from_constraint(
            constraint,
            detector=_AlwaysActive(),
            tokenizer=fake_tokenizer,
        )
        assert proc.num_allowed == 3
        out = proc(torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 10))
        for tok in range(10):
            if tok in {3, 5, 7}:
                assert out[0, tok].item() == 0.0
            else:
                assert math.isinf(out[0, tok].item())


# ---------------------------------------------------------------------------
# End-to-end with a tiny HF model + heuristic detector
# ---------------------------------------------------------------------------


class TestSelectiveEndToEnd:
    def test_runs_with_tiny_model(self, tiny_lm) -> None:
        from transformers import LogitsProcessorList

        tok, mdl = tiny_lm
        detector = HeuristicFactualSpanDetector()
        # Pick a tiny allowlist; we only care that generation runs and that
        # constrained timesteps respect the mask, not that output is sensible.
        allowed = {5, 10, 15, int(tok.eos_token_id)}
        proc = SelectiveLogitsProcessor(
            allowed, detector=detector, tokenizer=tok
        )

        prompt = tok("Hello", return_tensors="pt")
        prompt_len = prompt["input_ids"].shape[1]
        out = mdl.generate(
            **prompt,
            max_new_tokens=12,
            do_sample=False,
            logits_processor=LogitsProcessorList([proc]),
            pad_token_id=tok.pad_token_id,
        )
        new_tokens = out[0, prompt_len:].tolist()

        # n_steps should equal the number of generated tokens.
        assert proc.n_steps == len(new_tokens)
        # The fraction of constrained steps is in [0, 1].
        assert 0.0 <= proc.fraction_constrained <= 1.0

    def test_soft_selective_preserves_disallowed_ordering(self, fake_tokenizer) -> None:
        # A finite penalty in selective mode should still produce the
        # "subtract, don't replace" semantics inside active timesteps.
        proc = SelectiveLogitsProcessor(
            [1], detector=_AlwaysActive(), tokenizer=fake_tokenizer, penalty=2.0
        )
        scores = torch.tensor([[0.5, 0.0, -1.0, 1.5]])
        out = proc(torch.tensor([[7]]), scores)
        assert out[0, 1].item() == 0.0  # allowed → unchanged
        # Disallowed positions: 0.5-2=-1.5, -1-2=-3, 1.5-2=-0.5 — order preserved.
        disallowed_before = [scores[0, i].item() for i in (0, 2, 3)]
        disallowed_after = [out[0, i].item() for i in (0, 2, 3)]
        assert sorted(range(3), key=lambda i: disallowed_after[i]) == sorted(
            range(3), key=lambda i: disallowed_before[i]
        )
        assert not proc.is_hard_mask

    def test_soft_selective_passthrough_unchanged_when_inactive(
        self, fake_tokenizer
    ) -> None:
        # Even with a finite penalty, an inactive detector means no change.
        proc = SelectiveLogitsProcessor(
            [1], detector=_NeverActive(), tokenizer=fake_tokenizer, penalty=5.0
        )
        scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
        out = proc(torch.tensor([[7]]), scores)
        assert torch.equal(out, scores)

    def test_hard_vs_soft_differ_qualitatively_on_controlled_scores(
        self, fake_tokenizer
    ) -> None:
        # Logits-level comparison: identical inputs, an always-active detector,
        # only token 1 in the allowlist. With a strong disallowed score for
        # token 2, hard mode forces token 1; soft mode (small penalty) leaves
        # token 2 the argmax. This is the qualitative behaviour the soft
        # variant exists to enable.
        scores = torch.tensor([[0.0, 0.5, 10.0, 0.0]])
        input_ids = torch.tensor([[7]])

        hard = SelectiveLogitsProcessor(
            [1], detector=_AlwaysActive(), tokenizer=fake_tokenizer
        )
        soft = SelectiveLogitsProcessor(
            [1], detector=_AlwaysActive(), tokenizer=fake_tokenizer, penalty=1.0
        )

        hard_argmax = hard(input_ids, scores.clone()).argmax(dim=-1).item()
        soft_argmax = soft(input_ids, scores.clone()).argmax(dim=-1).item()

        assert hard_argmax == 1  # forced into allowlist
        assert soft_argmax == 2  # 10.0 - 1.0 still beats 0.5
        assert hard_argmax != soft_argmax

    def test_two_consecutive_generations_auto_reset(self, tiny_lm) -> None:
        from transformers import LogitsProcessorList

        tok, mdl = tiny_lm
        proc = SelectiveLogitsProcessor(
            {5, 10, int(tok.eos_token_id)},
            detector=HeuristicFactualSpanDetector(),
            tokenizer=tok,
        )

        for prompt_text in ["Hello", "World news today"]:
            prompt = tok(prompt_text, return_tensors="pt")
            n_steps_before = proc.n_steps
            mdl.generate(
                **prompt,
                max_new_tokens=4,
                do_sample=False,
                logits_processor=LogitsProcessorList([proc]),
                pad_token_id=tok.pad_token_id,
            )
            # Counter should have reset between generations.
            assert proc.n_steps <= 4
            assert proc.n_steps != n_steps_before + 4 or n_steps_before == 0


# ===========================================================================
# TrieSelectiveLogitsProcessor
# ===========================================================================


def _build_trie_constraint(
    entity_sequences: list[list[int]],
    number_sequences: list[list[int]] | None = None,
) -> GroundedConstraint:
    """Construct a GroundedConstraint with a non-trivial trie for tests."""
    entity_tokens: set[int] = set()
    for seq in entity_sequences:
        entity_tokens.update(seq)
    number_tokens: set[int] = set()
    for seq in number_sequences or []:
        number_tokens.update(seq)
    facts = SourceFacts(
        source_text="",
        entities=[],
        numbers=[],
        entity_tokens=entity_tokens,
        number_tokens=number_tokens,
        entity_token_sequences=entity_sequences,
        number_token_sequences=number_sequences or [],
    )
    return build_constraint(facts)


class TestTrieSelectiveGate:
    def test_passes_scores_through_when_inactive(self, fake_tokenizer) -> None:
        constraint = _build_trie_constraint([[1, 2, 3]])
        proc = TrieSelectiveLogitsProcessor(
            constraint, detector=_NeverActive(), tokenizer=fake_tokenizer
        )
        scores = torch.tensor([[0.0, 1.0, 0.5, 2.0]])
        out = proc(torch.tensor([[7, 8]]), scores)
        assert torch.equal(out, scores)
        assert proc.fraction_constrained == 0.0
        assert proc.n_entities_started == 0

    def test_active_seeds_live_set_to_start_tokens(self, fake_tokenizer) -> None:
        # Two source entities: [1,2,3] and [5,6]. Start tokens are {1, 5}.
        constraint = _build_trie_constraint([[1, 2, 3], [5, 6]])
        proc = TrieSelectiveLogitsProcessor(
            constraint, detector=_AlwaysActive(), tokenizer=fake_tokenizer
        )
        scores = torch.zeros(1, 8)
        out = proc(torch.tensor([[99]]), scores)
        # 1 and 5 allowed, everything else blocked.
        for tok in range(8):
            if tok in {1, 5}:
                assert out[0, tok].item() == 0.0, f"token {tok} should be allowed"
            else:
                assert math.isinf(out[0, tok].item()), f"token {tok} should be blocked"
        assert proc.n_entities_started == 1
        assert proc.n_constrained == 1


class TestTrieAdvancement:
    def test_advances_through_a_single_entity_path(self, fake_tokenizer) -> None:
        # Entity tokenises to [1, 2, 3]. Always-active detector.
        constraint = _build_trie_constraint([[1, 2, 3]])
        proc = TrieSelectiveLogitsProcessor(
            constraint, detector=_AlwaysActive(), tokenizer=fake_tokenizer
        )
        prompt = torch.tensor([[99]])

        # Step 1: prompt only, no generated tokens. Should allow start = {1}.
        out = proc(prompt, torch.zeros(1, 6))
        assert out[0, 1].item() == 0.0
        for t in (0, 2, 3, 4, 5):
            assert math.isinf(out[0, t].item())

        # Step 2: model picked 1. Live = {child for 1}; allowed = {2}.
        out = proc(torch.tensor([[99, 1]]), torch.zeros(1, 6))
        assert out[0, 2].item() == 0.0
        for t in (0, 1, 3, 4, 5):
            assert math.isinf(out[0, t].item())

        # Step 3: model picked 2. Live = {child for 2}; allowed = {3}.
        out = proc(torch.tensor([[99, 1, 2]]), torch.zeros(1, 6))
        assert out[0, 3].item() == 0.0
        for t in (0, 1, 2, 4, 5):
            assert math.isinf(out[0, t].item())

        # Step 4: model picked 3. The trie node is terminal → deactivate.
        scores = torch.tensor([[0.0, 0.5, 1.0, 0.5, 0.0, 2.0]])
        out = proc(torch.tensor([[99, 1, 2, 3]]), scores)
        # Passthrough on terminal.
        assert torch.equal(out, scores)
        assert proc.n_entities_completed == 1

    def test_branching_paths_allow_either_continuation(self, fake_tokenizer) -> None:
        # "Lloyds Banking Group" = [1,2,3] and "Lloyds Bank" = [1,2,4].
        constraint = _build_trie_constraint([[1, 2, 3], [1, 2, 4]])
        proc = TrieSelectiveLogitsProcessor(
            constraint, detector=_AlwaysActive(), tokenizer=fake_tokenizer
        )

        # Walk through to the branching point.
        proc(torch.tensor([[99]]), torch.zeros(1, 6))         # start
        proc(torch.tensor([[99, 1]]), torch.zeros(1, 6))      # after 1, allow 2
        out = proc(torch.tensor([[99, 1, 2]]), torch.zeros(1, 6))
        # After 1, 2, both 3 and 4 are valid continuations.
        for t in (3, 4):
            assert out[0, t].item() == 0.0
        for t in (0, 1, 2, 5):
            assert math.isinf(out[0, t].item())


class TestTrieFixesNumberFailure:
    def test_two_hundred_thousand_cannot_become_two_hundred_one(
        self, fake_tokenizer
    ) -> None:
        # Source contains "200,000" tokenised as [10, 11, 12], and a separate
        # number "001" elsewhere tokenised to [13]. A flat allowlist would
        # admit [10, 13] sequentially — the trie must reject this and force
        # 11 after 10.
        constraint = _build_trie_constraint(
            entity_sequences=[],
            number_sequences=[[10, 11, 12], [13]],
        )
        # Sanity: flat allowlist would include both 11 and 13.
        assert {11, 13}.issubset(constraint.allowlist.token_ids)

        proc = TrieSelectiveLogitsProcessor(
            constraint, detector=_AlwaysActive(), tokenizer=fake_tokenizer
        )
        # Step 1: prompt only. Start tokens = {10, 13}.
        out = proc(torch.tensor([[99]]), torch.zeros(1, 16))
        assert out[0, 10].item() == 0.0
        assert out[0, 13].item() == 0.0
        # Step 2: model picked 10. Now ONLY 11 is allowed — even though 13
        # is in the flat allowlist, the trie forces continuation of "200,000".
        out = proc(torch.tensor([[99, 10]]), torch.zeros(1, 16))
        assert out[0, 11].item() == 0.0
        assert math.isinf(out[0, 13].item()), "13 leaks through the trie"
        for t in range(16):
            if t == 11:
                continue
            assert math.isinf(out[0, t].item()), f"token {t} should be blocked"


class TestTrieDeactivationAfterTerminal:
    def test_passthrough_after_entity_completes(self, fake_tokenizer) -> None:
        constraint = _build_trie_constraint([[1, 2]])
        proc = TrieSelectiveLogitsProcessor(
            constraint, detector=_AlwaysActive(), tokenizer=fake_tokenizer
        )
        proc(torch.tensor([[99]]), torch.zeros(1, 5))      # start, allow {1}
        proc(torch.tensor([[99, 1]]), torch.zeros(1, 5))   # allow {2}
        # After completing [1, 2] the entity terminates; this call should
        # pass through even with an always-active detector.
        scores = torch.tensor([[1.0, 0.5, 0.2, 3.0, 0.0]])
        out = proc(torch.tensor([[99, 1, 2]]), scores)
        assert torch.equal(out, scores)
        assert proc.n_entities_completed == 1


class TestTrieReentryAfterTerminal:
    def test_detector_can_refire_for_a_new_entity(self, fake_tokenizer) -> None:
        # Two distinct entities. Detector toggles: active, active, then a
        # passthrough step (entity done), then active again for a new entity.
        constraint = _build_trie_constraint([[1, 2], [5, 6]])

        class Toggling:
            """Active when generated_ids is short or has length >= 3."""
            def reset(self) -> None: pass
            def is_active(self, ids, tok) -> bool:
                # Always active — we want to verify reentry through terminal logic.
                return True

        proc = TrieSelectiveLogitsProcessor(
            constraint, detector=Toggling(), tokenizer=fake_tokenizer
        )

        # Step 1: start of entity 1.
        out = proc(torch.tensor([[99]]), torch.zeros(1, 8))
        # Either start token allowed.
        assert out[0, 1].item() == 0.0 and out[0, 5].item() == 0.0

        # Step 2: model picked 1 → only 2.
        out = proc(torch.tensor([[99, 1]]), torch.zeros(1, 8))
        assert out[0, 2].item() == 0.0
        assert math.isinf(out[0, 5].item())

        # Step 3: terminal hit. Passthrough.
        scores = torch.zeros(1, 8)
        out = proc(torch.tensor([[99, 1, 2]]), scores)
        assert torch.equal(out, scores)

        # Step 4: detector still active → reseed → start tokens allowed again.
        out = proc(torch.tensor([[99, 1, 2, 7]]), torch.zeros(1, 8))
        assert out[0, 1].item() == 0.0
        assert out[0, 5].item() == 0.0
        assert proc.n_entities_started == 2


class TestTrieNoFactsPassthrough:
    def test_active_detector_with_empty_trie_passes_through(
        self, fake_tokenizer
    ) -> None:
        # No source facts ⇒ trie has no entities. The processor must not
        # produce all-blocked logits; it should fall back to passthrough.
        constraint = _build_trie_constraint([])
        assert constraint.trie.num_entities == 0
        proc = TrieSelectiveLogitsProcessor(
            constraint, detector=_AlwaysActive(), tokenizer=fake_tokenizer
        )
        scores = torch.tensor([[0.5, 0.2, 0.9]])
        out = proc(torch.tensor([[99]]), scores)
        assert torch.equal(out, scores)
        assert proc.n_constrained == 0


class TestTrieSoftPenalty:
    def test_soft_allows_off_trie_when_logit_is_large(self, fake_tokenizer) -> None:
        constraint = _build_trie_constraint([[1, 2]])
        proc = TrieSelectiveLogitsProcessor(
            constraint,
            detector=_AlwaysActive(),
            tokenizer=fake_tokenizer,
            penalty=1.0,
        )
        # Start step. Trie allows {1}; soft mode subtracts 1.0 from others.
        # Token 3 has logit 10.0; even after the 1.0 penalty it dominates.
        scores = torch.tensor([[0.0, 0.5, 0.0, 10.0]])
        out = proc(torch.tensor([[99]]), scores.clone())
        # Allowed unchanged, others penalised.
        assert out[0, 1].item() == 0.5
        assert out[0, 0].item() == -1.0
        assert out[0, 3].item() == 9.0
        # argmax still on disallowed.
        assert out.argmax(dim=-1).item() == 3
        assert not proc.is_hard_mask

    def test_soft_passthrough_when_inactive(self, fake_tokenizer) -> None:
        constraint = _build_trie_constraint([[1]])
        proc = TrieSelectiveLogitsProcessor(
            constraint,
            detector=_NeverActive(),
            tokenizer=fake_tokenizer,
            penalty=5.0,
        )
        scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
        out = proc(torch.tensor([[99]]), scores)
        assert torch.equal(out, scores)


class TestTriePromptResetAndCounters:
    def test_new_generation_resets_state(self, fake_tokenizer) -> None:
        constraint = _build_trie_constraint([[1, 2]])
        proc = TrieSelectiveLogitsProcessor(
            constraint, detector=_AlwaysActive(), tokenizer=fake_tokenizer
        )
        proc(torch.tensor([[10, 11, 12]]), torch.zeros(1, 5))
        proc(torch.tensor([[10, 11, 12, 1]]), torch.zeros(1, 5))
        assert proc.n_steps == 2

        # A fresh prompt with no shared prefix forces a reset.
        proc(torch.tensor([[20, 21]]), torch.zeros(1, 5))
        assert proc._prompt_length == 2
        assert proc.n_steps == 1

    def test_batched_input_raises(self, fake_tokenizer) -> None:
        constraint = _build_trie_constraint([[1, 2]])
        proc = TrieSelectiveLogitsProcessor(
            constraint, detector=_NeverActive(), tokenizer=fake_tokenizer
        )
        with pytest.raises(NotImplementedError):
            proc(torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 4))

    def test_negative_penalty_raises(self, fake_tokenizer) -> None:
        constraint = _build_trie_constraint([[1]])
        with pytest.raises(ValueError):
            TrieSelectiveLogitsProcessor(
                constraint,
                detector=_NeverActive(),
                tokenizer=fake_tokenizer,
                penalty=-1.0,
            )


class TestTrieEndToEnd:
    def test_runs_with_tiny_model(self, tiny_lm) -> None:
        from transformers import LogitsProcessorList

        tok, mdl = tiny_lm
        # Build a trie with a couple of token sequences and confirm generation
        # runs end-to-end without errors.
        eos = int(tok.eos_token_id)
        constraint = _build_trie_constraint(
            entity_sequences=[[5, 10], [15, 20, 25]],
        )
        # Manually add EOS so generation can terminate even when constraints
        # never fire (the heuristic detector probably won't fire on "Hello").
        proc = TrieSelectiveLogitsProcessor(
            constraint,
            detector=HeuristicFactualSpanDetector(),
            tokenizer=tok,
        )

        prompt = tok("Hello", return_tensors="pt")
        prompt_len = prompt["input_ids"].shape[1]
        out = mdl.generate(
            **prompt,
            max_new_tokens=8,
            do_sample=False,
            logits_processor=LogitsProcessorList([proc]),
            pad_token_id=tok.pad_token_id,
        )
        new_tokens = out[0, prompt_len:].tolist()
        assert proc.n_steps == len(new_tokens)
        assert 0.0 <= proc.fraction_constrained <= 1.0
        _ = eos  # silence unused warning
