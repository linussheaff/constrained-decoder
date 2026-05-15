"""Run the full source-grounded-decoding experiment.

For each example sampled from the dataset, generates a summary under four
decoding conditions.

Designed to be both a runnable CLI script and an importable Python module so
the Colab notebook can call :func:`run_experiment` directly with custom args.

Example:

    python scripts/run_experiment.py \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --n 50 \\
        --output results/raw/xsum_50.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Make src importable when the script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.entity_extractor import extract_facts  
from src.evaluate_faithfulness import evaluate_faithfulness  
from src.evaluate_quality import evaluate_quality  
from src.generate import (  
    CONDITION_NAMES,
    expand_condition_names,
    generate_summaries,
)

logger = logging.getLogger("run_experiment")


# Default condition list (single-soft-penalty mode). When a sweep is in use
# the active list is computed via :func:`expand_condition_names`.
CONDITIONS: tuple[str, ...] = CONDITION_NAMES

DEFAULT_INSTRUCTION: str = (
    "Summarise the following article in a single concise sentence. "
    "Only state facts that appear in the article."
)


# Config


@dataclass
class ExperimentConfig:
    model: str
    dataset: str
    split: str
    n: int
    seed: int
    max_new_tokens: int
    soft_penalty: float
    dtype: str
    device: str | None
    output: str
    instruction: str
    soft_penalty_sweep: tuple[float, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Normalise the sweep to a plain list so the JSON dump is portable.
        if d.get("soft_penalty_sweep") is not None:
            d["soft_penalty_sweep"] = list(d["soft_penalty_sweep"])
        return d

    @property
    def conditions(self) -> tuple[str, ...]:
        """Active condition list, accounting for the soft-penalty sweep."""
        return expand_condition_names(self.soft_penalty_sweep)


# Model & data loading


def _resolve_dtype(name: str) -> "torch.dtype":
    import torch

    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(f"Unknown dtype {name!r}. Choices: {sorted(table)}")
    return table[name]


def _resolve_device(name: str | None) -> str:
    import torch

    if name is not None:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    model_name: str, dtype: str = "float16", device: str | None = None
):
    """Load a HuggingFace causal LM and tokenizer onto the chosen device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = _resolve_device(device)
    if dtype != "float32" and resolved_device == "cpu":
        logger.warning(
            "fp16/bf16 on CPU is slow; falling back to float32 for the CPU run."
        )
        torch_dtype = torch.float32
    else:
        torch_dtype = _resolve_dtype(dtype)

    logger.info("Loading tokenizer %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        "Loading model %s (dtype=%s, device=%s)", model_name, torch_dtype, resolved_device
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype
    )
    model.to(resolved_device)
    model.eval()
    return tokenizer, model


def load_dataset_examples(
    dataset: str,
    split: str,
    n: int,
    seed: int,
) -> list[dict[str, str]]:
    """Sample n examples from dataset[split] and normalise the fields.

    Returns dicts with article, reference, and idx. Compatible with
    XSum (document/summary) and the closely-related SAMSum
    (dialogue/summary); for other datasets, override field detection.
    """
    from datasets import load_dataset

    logger.info("Loading dataset %s split=%s", dataset, split)
    ds = load_dataset(dataset, split=split)
    logger.info("Loaded %d examples; sampling %d (seed=%d)", len(ds), n, seed)
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    sampled = indices[:n]

    examples: list[dict[str, str]] = []
    for idx in sampled:
        row = ds[idx]
        article = (
            row.get("document")
            or row.get("article")
            or row.get("dialogue")
            or row.get("input")
        )
        reference = (
            row.get("summary")
            or row.get("highlights")
            or row.get("target")
            or row.get("output")
        )
        if article is None or reference is None:
            logger.warning("Skipping idx=%d: missing article/reference fields", idx)
            continue
        examples.append(
            {"idx": int(idx), "article": str(article), "reference": str(reference)}
        )
    return examples


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


def build_prompt(article: str, tokenizer: Any, instruction: str) -> str:
    """Build a model-ready prompt using the chat template when available."""
    user_content = f"{instruction}\n\nArticle:\n{article}"
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        messages = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{user_content}\n\nSummary:"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


_METRIC_KEYS: tuple[str, ...] = (
    "entity_precision",
    "entity_recall",
    "entity_f1",
    "number_accuracy",
    "hallucinated_entity_count",
    "unsupported_number_count",
    "rouge1",
    "rouge2",
    "rougeL",
    "length_chars",
    "length_tokens",
    "fraction_constrained",
    "n_new_tokens",
)


def _safe_mean(values: list[float]) -> float | None:
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return statistics.fmean(cleaned)


def _safe_stdev(values: list[float]) -> float | None:
    cleaned = [v for v in values if v is not None]
    if len(cleaned) < 2:
        return 0.0 if cleaned else None
    return statistics.stdev(cleaned)


def aggregate_metrics(
    per_example: list[dict[str, Any]],
    conditions: Iterable[str] = CONDITIONS,
) -> dict[str, dict[str, dict[str, float | None]]]:
    """For each condition, compute mean and stdev of each metric."""
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for cond in conditions:
        cond_metrics: dict[str, list[float]] = {k: [] for k in _METRIC_KEYS}
        for record in per_example:
            payload = record["conditions"].get(cond)
            if payload is None:
                continue
            for key in _METRIC_KEYS:
                value = payload.get(key)
                if value is None:
                    continue
                cond_metrics[key].append(float(value))
        out[cond] = {
            key: {"mean": _safe_mean(values), "stdev": _safe_stdev(values)}
            for key, values in cond_metrics.items()
        }
    return out


# Per-example processing


def _flatten_metrics(
    condition_name: str,
    text: str,
    new_token_ids: list[int],
    fraction_constrained: float | None,
    faith: Any,
    quality: Any,
) -> dict[str, Any]:
    """Merge the dataclass reports into a flat dict for JSON serialisation."""
    return {
        "name": condition_name,
        "text": text,
        "n_new_tokens": len(new_token_ids),
        "fraction_constrained": fraction_constrained,
        # Faithfulness
        "entity_precision": faith.entity_precision,
        "entity_recall": faith.entity_recall,
        "entity_f1": faith.entity_f1,
        "number_accuracy": faith.number_accuracy,
        "hallucinated_entity_count": faith.hallucinated_entity_count,
        "unsupported_number_count": faith.unsupported_number_count,
        "summary_entities": faith.summary_entities,
        "hallucinated_entities": faith.hallucinated_entities,
        "matched_numbers": faith.matched_numbers,
        "unsupported_numbers": faith.unsupported_numbers,
        # Quality
        "rouge1": quality.rouge1,
        "rouge2": quality.rouge2,
        "rougeL": quality.rougeL,
        "length_chars": quality.length_chars,
        "length_tokens": quality.length_tokens,
    }


def process_example(
    example: dict[str, str],
    model: Any,
    tokenizer: Any,
    instruction: str,
    max_new_tokens: int,
    soft_penalty: float,
    soft_penalty_sweep: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run all conditions on a single example and return a serialisable record."""
    article = example["article"]
    reference = example["reference"]
    prompt = build_prompt(article, tokenizer, instruction)

    # Pre-compute source facts once (shared across all generations + eval).
    source_facts = extract_facts(article, tokenizer=tokenizer)

    summaries = generate_summaries(
        article,
        prompt,
        model,
        tokenizer,
        max_new_tokens=max_new_tokens,
        soft_penalty=soft_penalty,
        soft_penalty_sweep=soft_penalty_sweep,
        source_facts=source_facts,
    )

    cond_records: dict[str, dict[str, Any]] = {}
    for name, summary in summaries.items():
        faith = evaluate_faithfulness(summary.text, article, source_facts=source_facts)
        quality = evaluate_quality(summary.text, reference=reference, tokenizer=tokenizer)
        cond_records[name] = _flatten_metrics(
            condition_name=name,
            text=summary.text,
            new_token_ids=summary.new_token_ids,
            fraction_constrained=summary.fraction_constrained,
            faith=faith,
            quality=quality,
        )

    return {
        "idx": example["idx"],
        "article": article,
        "reference": reference,
        "prompt": prompt,
        "conditions": cond_records,
    }


# Driver


def _set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(
    config: ExperimentConfig,
    examples: list[dict[str, str]] | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the experiment and return the result dict (also written to disk).

    Args:
        config: Experiment configuration.
        examples: Pre-loaded examples (skip dataset loading). Used by tests.
        model, tokenizer: Pre-loaded model/tokenizer (skip HF load). Used by tests.
        progress: Show tqdm progress bar (set False for clean test output).
    """
    _set_seeds(config.seed)

    if model is None or tokenizer is None:
        tokenizer, model = load_model_and_tokenizer(
            config.model, dtype=config.dtype, device=config.device
        )
    if examples is None:
        examples = load_dataset_examples(
            config.dataset, config.split, config.n, config.seed
        )
    examples = list(examples)[: config.n]
    if not examples:
        raise ValueError("No examples available to run.")

    logger.info(
        "Running %d examples × %d conditions = %d generations",
        len(examples),
        len(CONDITIONS),
        len(examples) * len(CONDITIONS),
    )

    iterator: Iterable[dict[str, str]] = examples
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(examples, desc="examples")
        except ImportError:  # pragma: no cover
            iterator = examples

    conditions = config.conditions

    per_example: list[dict[str, Any]] = []
    start = time.time()
    for example in iterator:
        try:
            record = process_example(
                example,
                model=model,
                tokenizer=tokenizer,
                instruction=config.instruction,
                max_new_tokens=config.max_new_tokens,
                soft_penalty=config.soft_penalty,
                soft_penalty_sweep=config.soft_penalty_sweep,
            )
        except Exception as exc:  # one bad example shouldn't kill the run
            logger.exception("Example idx=%s failed: %s", example.get("idx"), exc)
            continue
        per_example.append(record)
    elapsed = time.time() - start

    aggregate = aggregate_metrics(per_example, conditions=conditions)

    result = {
        "config": config.as_dict(),
        "elapsed_seconds": elapsed,
        "n_examples_processed": len(per_example),
        "conditions": list(conditions),
        "per_example": per_example,
        "aggregate": aggregate,
    }

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s (%d examples in %.1fs)", output_path, len(per_example), elapsed)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset", default="EdinburghNLP/xsum")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--soft-penalty", type=float, default=5.0)
    parser.add_argument(
        "--soft-penalty-sweep",
        type=str,
        default=None,
        help=(
            "Comma-separated soft penalties to sweep, e.g. '2,5,10,20,40'. "
            "When set, replaces the single 'selective_soft' condition with one "
            "'selective_soft_p{p}' condition per value."
        ),
    )
    parser.add_argument(
        "--dtype", default="float16", choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="results/raw/experiment.json")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--quiet", action="store_true", help="suppress tqdm progress bar"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    sweep: tuple[float, ...] | None = None
    if args.soft_penalty_sweep:
        sweep = tuple(
            float(s.strip()) for s in args.soft_penalty_sweep.split(",") if s.strip()
        )
    config = ExperimentConfig(
        model=args.model,
        dataset=args.dataset,
        split=args.split,
        n=args.n,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        soft_penalty=args.soft_penalty,
        dtype=args.dtype,
        device=args.device,
        output=args.output,
        instruction=args.instruction,
        soft_penalty_sweep=sweep,
    )
    run_experiment(config, progress=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
