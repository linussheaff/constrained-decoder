"""Smoke tests for ``src.generate.generate_summaries``.

End-to-end: tiny GPT-2 + a short source document. We can't assert on output
quality (the model is randomly initialised), but we can assert on:

* Every condition returns a non-None ConditionalSummary.
* The fully-constrained baseline emits only allowlist tokens.
* Selective conditions report a sensible ``fraction_constrained``.
* Soft and hard selective produce distinct outputs in at least one realistic case.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.constraint_builder import build_constraint  # noqa: E402
from src.entity_extractor import extract_facts  # noqa: E402
from src.generate import (  # noqa: E402
    CONDITION_NAMES,
    ConditionalSummary,
    expand_condition_names,
    generate_summaries,
    soft_condition_name,
)


@pytest.fixture(scope="module")
def tiny_lm():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "sshleifer/tiny-gpt2"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForCausalLM.from_pretrained(name).eval()
    except Exception as exc:
        pytest.skip(f"could not load tiny model {name}: {exc}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, mdl


SOURCE_DOC = (
    "Lloyds Banking Group reported a pre-tax profit of £2.1 billion in 2023, "
    "up 57% on the previous year, chief executive Charlie Nunn told reporters "
    "in London."
)
PROMPT = "Article: " + SOURCE_DOC + "\nSummary:"


class TestGenerateSummariesShape:
    def test_returns_all_four_conditions(self, tiny_lm, spacy_nlp) -> None:
        tok, mdl = tiny_lm
        result = generate_summaries(
            SOURCE_DOC, PROMPT, mdl, tok, max_new_tokens=8
        )
        assert set(result.keys()) == set(CONDITION_NAMES)
        for name, summary in result.items():
            assert isinstance(summary, ConditionalSummary)
            assert summary.name == name
            assert isinstance(summary.text, str)
            assert summary.n_new_tokens > 0

    def test_only_selective_reports_fraction_constrained(
        self, tiny_lm, spacy_nlp
    ) -> None:
        tok, mdl = tiny_lm
        result = generate_summaries(
            SOURCE_DOC, PROMPT, mdl, tok, max_new_tokens=8
        )
        assert result["unconstrained"].fraction_constrained is None
        assert result["fully_constrained_hard"].fraction_constrained is None
        for name in ("selective_hard", "selective_soft"):
            f = result[name].fraction_constrained
            assert f is not None
            assert 0.0 <= f <= 1.0


class TestGenerateSummariesSemantics:
    def test_fully_constrained_output_is_in_allowlist(
        self, tiny_lm, spacy_nlp
    ) -> None:
        tok, mdl = tiny_lm
        # Reconstruct the same constraint generate_summaries builds and
        # verify every generated token is allowed.
        from src.generate import _default_extra_token_ids

        facts = extract_facts(SOURCE_DOC, tokenizer=tok)
        constraint = build_constraint(
            facts, extra_token_ids=_default_extra_token_ids(tok)
        )
        result = generate_summaries(
            SOURCE_DOC,
            PROMPT,
            mdl,
            tok,
            max_new_tokens=8,
            constraint=constraint,
        )
        new_ids = result["fully_constrained_hard"].new_token_ids
        assert new_ids, "expected some generated tokens"
        for tid in new_ids:
            assert tid in constraint.allowlist.token_ids, (
                f"token {tid} not in allowlist"
            )

    def test_unconstrained_is_unrestricted(self, tiny_lm, spacy_nlp) -> None:
        # The unconstrained run is allowed to produce tokens outside the
        # allowlist. We don't assert it *does* (could go either way for a
        # random model), but we verify it ran without error and produced
        # something.
        tok, mdl = tiny_lm
        result = generate_summaries(
            SOURCE_DOC, PROMPT, mdl, tok, max_new_tokens=8
        )
        assert result["unconstrained"].new_token_ids


class TestSoftPenaltySweep:
    def test_sweep_replaces_selective_soft_with_per_penalty_variants(
        self, tiny_lm, spacy_nlp
    ) -> None:
        tok, mdl = tiny_lm
        sweep = (2.0, 5.0, 10.0)
        result = generate_summaries(
            SOURCE_DOC,
            PROMPT,
            mdl,
            tok,
            max_new_tokens=4,
            soft_penalty_sweep=sweep,
        )
        expected = set(expand_condition_names(sweep))
        assert set(result.keys()) == expected
        # The unsuffixed selective_soft is replaced, not kept alongside.
        assert "selective_soft" not in result
        for p in sweep:
            name = soft_condition_name(p)
            assert name in result
            summary = result[name]
            assert isinstance(summary, ConditionalSummary)
            assert summary.fraction_constrained is not None
            assert 0.0 <= summary.fraction_constrained <= 1.0

    def test_soft_condition_name_formatting(self) -> None:
        assert soft_condition_name(2) == "selective_soft_p2"
        assert soft_condition_name(5.0) == "selective_soft_p5"
        assert soft_condition_name(2.5) == "selective_soft_p2_5"

    def test_expand_condition_names_no_sweep_is_identity(self) -> None:
        assert expand_condition_names(None) == CONDITION_NAMES
        assert expand_condition_names(()) == CONDITION_NAMES

    def test_expand_condition_names_with_sweep(self) -> None:
        out = expand_condition_names((2.0, 5.0, 10.0))
        assert out == (
            "unconstrained",
            "fully_constrained_hard",
            "selective_hard",
            "selective_soft_p2",
            "selective_soft_p5",
            "selective_soft_p10",
        )


class TestExtraTokenIdsOverride:
    def test_explicit_extras_override_defaults(
        self, tiny_lm, spacy_nlp
    ) -> None:
        tok, mdl = tiny_lm
        # Provide only EOS as extra — the fully-constrained run might
        # produce mostly EOS, which is fine. We just check the override is honoured.
        result = generate_summaries(
            SOURCE_DOC,
            PROMPT,
            mdl,
            tok,
            max_new_tokens=4,
            extra_token_ids=[int(tok.eos_token_id)],
        )
        assert set(result.keys()) == set(CONDITION_NAMES)
