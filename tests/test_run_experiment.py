"""Smoke tests for ``scripts/run_experiment.py``.

Runs the full driver against a synthetic 2-example dataset and the tiny
random GPT-2 model. The point is to verify the JSON has the expected
structure end-to-end, not to assert on summary quality. Includes a
``plot_all`` round-trip that renders the figures into a temp directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("rouge_score")

import json  # noqa: E402

from scripts.plot_results import plot_all  # noqa: E402
from scripts.run_experiment import (  # noqa: E402
    CONDITIONS,
    ExperimentConfig,
    DEFAULT_INSTRUCTION,
    aggregate_metrics,
    build_prompt,
    run_experiment,
)


SOURCE = (
    "Lloyds Banking Group reported a pre-tax profit of £2.1 billion in 2023, "
    "up 57% on the previous year, chief executive Charlie Nunn told reporters "
    "in London."
)
REFERENCE = "Lloyds Banking Group made a £2.1 billion profit in 2023."


@pytest.fixture(scope="module")
def tiny_lm():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "sshleifer/tiny-gpt2"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForCausalLM.from_pretrained(name).eval()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"could not load {name}: {exc}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, mdl


def _synthetic_examples() -> list[dict[str, str]]:
    return [
        {"idx": 0, "article": SOURCE, "reference": REFERENCE},
        {
            "idx": 1,
            "article": "Apple Inc announced new products in California last week.",
            "reference": "Apple revealed new products in California.",
        },
    ]


class TestBuildPrompt:
    def test_falls_back_to_plain_prompt_without_chat_template(self, tiny_lm) -> None:
        tok, _ = tiny_lm
        # tiny-gpt2 has no chat template.
        assert not getattr(tok, "chat_template", None)
        prompt = build_prompt("body text", tok, instruction="Summarise it.")
        assert "body text" in prompt
        assert "Summary:" in prompt

    def test_uses_chat_template_when_available(self, tiny_lm) -> None:
        tok, _ = tiny_lm
        original_template = tok.chat_template
        try:
            tok.chat_template = (
                "{% for m in messages %}<<<{{ m['role'] }}: {{ m['content'] }}>>>"
                "{% endfor %}"
            )
            prompt = build_prompt("hello", tok, instruction="Summarise:")
            assert "<<<user:" in prompt
            assert "hello" in prompt
        finally:
            tok.chat_template = original_template


class TestRunExperimentSmoke:
    def test_writes_expected_json_structure(self, tiny_lm, spacy_nlp, tmp_path) -> None:
        tok, mdl = tiny_lm
        output = tmp_path / "smoke.json"
        config = ExperimentConfig(
            model="sshleifer/tiny-gpt2",  # informational only since we pass model in
            dataset="(synthetic)",
            split="(none)",
            n=2,
            seed=0,
            max_new_tokens=6,
            soft_penalty=2.0,
            dtype="float32",
            device="cpu",
            output=str(output),
            instruction=DEFAULT_INSTRUCTION,
        )
        result = run_experiment(
            config,
            examples=_synthetic_examples(),
            model=mdl,
            tokenizer=tok,
            progress=False,
        )

        assert output.exists()
        on_disk = json.loads(output.read_text())
        for blob in (result, on_disk):
            assert blob["n_examples_processed"] == 2
            assert blob["conditions"] == list(CONDITIONS)
            assert len(blob["per_example"]) == 2
            for record in blob["per_example"]:
                assert set(record["conditions"]) == set(CONDITIONS)
                for cond, payload in record["conditions"].items():
                    assert isinstance(payload["text"], str)
                    assert payload["n_new_tokens"] > 0
                    assert payload["entity_precision"] is not None
                    if cond.startswith("selective"):
                        assert 0.0 <= payload["fraction_constrained"] <= 1.0
                    else:
                        assert payload["fraction_constrained"] is None
            # Aggregate covers every condition.
            assert set(blob["aggregate"]) == set(CONDITIONS)
            for cond_metrics in blob["aggregate"].values():
                assert "entity_precision" in cond_metrics
                assert "mean" in cond_metrics["entity_precision"]

    def test_per_example_failure_does_not_abort(self, tiny_lm, spacy_nlp, tmp_path) -> None:
        # Drop in one well-formed example and one with a non-string article;
        # the driver should log + skip the bad one and still write a JSON
        # containing the good one.
        tok, mdl = tiny_lm
        output = tmp_path / "with_failure.json"
        config = ExperimentConfig(
            model="sshleifer/tiny-gpt2",
            dataset="(synthetic)",
            split="(none)",
            n=2,
            seed=0,
            max_new_tokens=4,
            soft_penalty=1.0,
            dtype="float32",
            device="cpu",
            output=str(output),
            instruction=DEFAULT_INSTRUCTION,
        )
        bad_examples = [
            {"idx": 0, "article": SOURCE, "reference": REFERENCE},
            {"idx": 1, "article": None, "reference": "x"},  # will fail in extract_facts
        ]
        result = run_experiment(
            config,
            examples=bad_examples,
            model=mdl,
            tokenizer=tok,
            progress=False,
        )
        assert result["n_examples_processed"] == 1
        assert result["per_example"][0]["idx"] == 0


class TestSoftPenaltySweep:
    def test_sweep_produces_per_penalty_conditions(
        self, tiny_lm, spacy_nlp, tmp_path
    ) -> None:
        tok, mdl = tiny_lm
        sweep = (2.0, 5.0, 10.0)
        config = ExperimentConfig(
            model="sshleifer/tiny-gpt2",
            dataset="(synthetic)",
            split="(none)",
            n=2,
            seed=0,
            max_new_tokens=4,
            soft_penalty=5.0,
            dtype="float32",
            device="cpu",
            output=str(tmp_path / "sweep.json"),
            instruction=DEFAULT_INSTRUCTION,
            soft_penalty_sweep=sweep,
        )
        # Conditions list should reflect the sweep.
        expected = (
            "unconstrained",
            "fully_constrained_hard",
            "selective_hard",
            "selective_soft_p2",
            "selective_soft_p5",
            "selective_soft_p10",
        )
        assert config.conditions == expected

        result = run_experiment(
            config,
            examples=_synthetic_examples(),
            model=mdl,
            tokenizer=tok,
            progress=False,
        )

        on_disk = json.loads(Path(config.output).read_text())
        for blob in (result, on_disk):
            assert tuple(blob["conditions"]) == expected
            assert "selective_soft" not in blob["per_example"][0]["conditions"]
            for cond in expected:
                assert cond in blob["per_example"][0]["conditions"]
            # Aggregate covers each sweep condition.
            assert set(blob["aggregate"]) == set(expected)
            assert blob["config"]["soft_penalty_sweep"] == list(sweep)


class TestAggregateMetrics:
    def test_handles_empty_input(self) -> None:
        agg = aggregate_metrics([])
        assert set(agg) == set(CONDITIONS)
        for cond_metrics in agg.values():
            for stats in cond_metrics.values():
                assert stats["mean"] is None
                assert stats["stdev"] is None

    def test_mean_and_stdev_on_simple_payload(self) -> None:
        per_example = [
            {
                "conditions": {
                    cond: {"entity_precision": v, "rouge1": v}
                    for cond, v in zip(CONDITIONS, [0.6, 0.7, 0.8, 0.9])
                }
            },
            {
                "conditions": {
                    cond: {"entity_precision": v, "rouge1": v}
                    for cond, v in zip(CONDITIONS, [0.8, 0.9, 1.0, 1.0])
                }
            },
        ]
        agg = aggregate_metrics(per_example)
        # unconstrained: mean(0.6, 0.8) = 0.7
        assert agg["unconstrained"]["entity_precision"]["mean"] == pytest.approx(0.7)
        assert agg["unconstrained"]["entity_precision"]["stdev"] >= 0.0


class TestPlotAllRoundTrip:
    def test_plot_all_writes_expected_pngs(self, tiny_lm, spacy_nlp, tmp_path) -> None:
        tok, mdl = tiny_lm
        config = ExperimentConfig(
            model="sshleifer/tiny-gpt2",
            dataset="(synthetic)",
            split="(none)",
            n=2,
            seed=0,
            max_new_tokens=4,
            soft_penalty=1.0,
            dtype="float32",
            device="cpu",
            output=str(tmp_path / "for_plots.json"),
            instruction=DEFAULT_INSTRUCTION,
        )
        result = run_experiment(
            config,
            examples=_synthetic_examples(),
            model=mdl,
            tokenizer=tok,
            progress=False,
        )

        output_dir = tmp_path / "figs"
        written = plot_all(result, output_dir)
        # We expect at least the four bar charts + the scatter; the activation
        # plot may or may not be produced depending on fraction_constrained.
        names = {Path(p).name for p in written if Path(p).exists()}
        assert "entity_precision.png" in names
        assert "hallucinated_entities.png" in names
        assert "rouge1.png" in names
        assert "faithfulness_vs_quality.png" in names
        for path in written:
            if path.exists():
                assert path.stat().st_size > 0
