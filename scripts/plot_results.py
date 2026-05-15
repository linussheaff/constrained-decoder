"""Render the experiment figures from a results JSON.

Reads the JSON produced by :mod:`scripts.run_experiment` and writes a set of
PNG figures into ``results/figures/`` (or a custom ``--output-dir``):
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("plot_results")


CONDITION_ORDER: tuple[str, ...] = (
    "unconstrained",
    "fully_constrained_hard",
    "selective_hard",
    "selective_soft",
)
CONDITION_COLOURS: dict[str, str] = {
    "unconstrained": "#4c72b0",
    "fully_constrained_hard": "#c44e52",
    "selective_hard": "#55a868",
    "selective_soft": "#8172b2",
}

# Colour ramp for selective_soft_pN sweep conditions,
# larger penalties get warmer colours.
_SWEEP_COLOUR_RAMP: tuple[str, ...] = (
    "#beb6d9", 
    "#8172b2",  
    "#6b4f93",
    "#9e3b76",
    "#c44e52", 
)


def _resolve_conditions(result: dict[str, Any]) -> tuple[str, ...]:
    """Pick the condition list from a result blob"""
    raw = result.get("conditions")
    if isinstance(raw, list) and raw:
        return tuple(raw)
    # Fall back to enumerating conditions seen on the first per-example record.
    per_ex = result.get("per_example", [])
    if per_ex:
        first = per_ex[0].get("conditions", {})
        if isinstance(first, dict) and first:
            return tuple(first.keys())
    return CONDITION_ORDER


def _colour_for(cond: str, sweep_index: dict[str, int]) -> str:
    if cond in CONDITION_COLOURS:
        return CONDITION_COLOURS[cond]
    if cond in sweep_index:
        i = sweep_index[cond]
        return _SWEEP_COLOUR_RAMP[min(i, len(_SWEEP_COLOUR_RAMP) - 1)]
    return "#888888"


def _sweep_index(conditions: tuple[str, ...]) -> dict[str, int]:
    """Map each condition to its position in the sweep."""
    sweep = [c for c in conditions if c.startswith("selective_soft_p")]
    return {c: i for i, c in enumerate(sweep)}


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  
    except ImportError as exc:  
        raise RuntimeError(
            "matplotlib is required for plotting; pip install matplotlib"
        ) from exc


def _per_condition_metric(
    per_example: list[dict[str, Any]],
    metric: str,
    conditions: tuple[str, ...] = CONDITION_ORDER,
) -> dict[str, list[float]]:
    """Collect per-example values of metric for each condition."""
    out: dict[str, list[float]] = {c: [] for c in conditions}
    for record in per_example:
        for cond, payload in record.get("conditions", {}).items():
            if cond not in out:
                continue
            value = payload.get(metric)
            if value is None:
                continue
            out[cond].append(float(value))
    return out


def _bar_plot(
    title: str,
    ylabel: str,
    values_per_condition: dict[str, list[float]],
    output_path: Path,
    conditions: tuple[str, ...] = CONDITION_ORDER,
    sweep_index: dict[str, int] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    sweep_index = sweep_index or {}
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(conditions)), 4))
    means = []
    stds = []
    colours = []
    for cond in conditions:
        values = values_per_condition.get(cond, [])
        if not values:
            means.append(0.0)
            stds.append(0.0)
        else:
            mean = sum(values) / len(values)
            if len(values) > 1:
                var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                std = var**0.5
            else:
                std = 0.0
            means.append(mean)
            stds.append(std)
        colours.append(_colour_for(cond, sweep_index))

    xs = list(range(len(conditions)))
    ax.bar(xs, means, yerr=stds, color=colours, alpha=0.85, capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("_", "\n") for c in conditions], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", output_path)


def _scatter_faithfulness_vs_quality(
    per_example: list[dict[str, Any]],
    output_path: Path,
    conditions: tuple[str, ...] = CONDITION_ORDER,
    sweep_index: dict[str, int] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    sweep_index = sweep_index or {}
    fig, ax = plt.subplots(figsize=(7, 5))
    for cond in conditions:
        xs: list[float] = []
        ys: list[float] = []
        for record in per_example:
            payload = record.get("conditions", {}).get(cond)
            if not payload:
                continue
            r = payload.get("rouge1")
            p = payload.get("entity_precision")
            if r is None or p is None:
                continue
            xs.append(float(r))
            ys.append(float(p))
        if not xs:
            continue
        colour = _colour_for(cond, sweep_index)
        ax.scatter(xs, ys, alpha=0.4, color=colour, label=cond, s=30)
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        ax.scatter([mean_x], [mean_y], color=colour, s=180, marker="X",
                   edgecolors="black", linewidths=1.2)

    ax.set_xlabel("ROUGE-1 (quality)")
    ax.set_ylabel("Entity precision (faithfulness)")
    ax.set_title("Faithfulness vs quality, per example.\nLarge X = condition mean.")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", output_path)


def _selective_activation_plot(
    per_example: list[dict[str, Any]],
    output_path: Path,
    conditions: tuple[str, ...] = CONDITION_ORDER,
    sweep_index: dict[str, int] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    sweep_index = sweep_index or {}
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    selective = tuple(c for c in conditions if c.startswith("selective"))
    if not selective:
        selective = ("selective_hard", "selective_soft")
    for cond in selective:
        values: list[float] = []
        for record in per_example:
            payload = record.get("conditions", {}).get(cond)
            if not payload:
                continue
            f = payload.get("fraction_constrained")
            if f is None:
                continue
            values.append(float(f))
        if not values:
            continue
        ax.hist(
            values,
            bins=20,
            alpha=0.55,
            color=_colour_for(cond, sweep_index),
            label=cond,
            edgecolor="white",
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        logger.info("no fraction_constrained data; skipping %s", output_path)
        return
    ax.set_xlabel("fraction of steps with constraints active")
    ax.set_ylabel("examples")
    ax.set_title("Selective detector activation rate")
    ax.set_xlim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", output_path)


def _soft_sweep_curve(
    per_example: list[dict[str, Any]],
    output_path: Path,
    conditions: tuple[str, ...],
) -> Path | None:
    """Plot mean entity_precision and rouge1 across selective_soft_pN penalties."""
    import matplotlib.pyplot as plt

    sweep = [c for c in conditions if c.startswith("selective_soft_p")]
    if len(sweep) < 2:
        return None

    # Recover the numeric penalty from the condition name.
    def parse_penalty(name: str) -> float:
        tail = name[len("selective_soft_p") :].replace("_", ".")
        try:
            return float(tail)
        except ValueError:
            return float("nan")

    penalties = [parse_penalty(c) for c in sweep]
    order = sorted(range(len(sweep)), key=lambda i: penalties[i])
    sweep = [sweep[i] for i in order]
    penalties = [penalties[i] for i in order]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    precs = [
        _per_condition_metric(per_example, "entity_precision", (c,)).get(c, [])
        for c in sweep
    ]
    rouges = [
        _per_condition_metric(per_example, "rouge1", (c,)).get(c, [])
        for c in sweep
    ]

    def mean(vs: list[float]) -> float | None:
        return (sum(vs) / len(vs)) if vs else None

    prec_means = [mean(v) for v in precs]
    rouge_means = [mean(v) for v in rouges]

    ax1.plot(
        penalties,
        prec_means,
        marker="o",
        color="#55a868",
        label="entity precision",
    )
    ax2.plot(
        penalties,
        rouge_means,
        marker="s",
        color="#4c72b0",
        label="ROUGE-1",
    )
    ax1.set_xlabel("soft penalty (subtracted from off-trie logits)")
    ax1.set_ylabel("mean entity precision (faithfulness)", color="#55a868")
    ax2.set_ylabel("mean ROUGE-1 F (quality)", color="#4c72b0")
    ax1.set_title("Soft-penalty sweep: faithfulness/quality trade-off")
    ax1.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", output_path)
    return output_path


def plot_all(result: dict[str, Any], output_dir: Path) -> list[Path]:
    """Render all figures into ``output_dir``. Returns the list of paths written."""
    _ensure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_example = result.get("per_example", [])
    conditions = _resolve_conditions(result)
    sweep_index = _sweep_index(conditions)
    written: list[Path] = []

    for metric, title, ylabel, filename in [
        (
            "entity_precision",
            "Entity precision — higher is more faithful",
            "mean entity precision",
            "entity_precision.png",
        ),
        (
            "hallucinated_entity_count",
            "Hallucinated entities per summary",
            "mean count",
            "hallucinated_entities.png",
        ),
        (
            "rouge1",
            "ROUGE-1 against reference summary",
            "mean ROUGE-1 F",
            "rouge1.png",
        ),
        (
            "number_accuracy",
            "Number accuracy — fraction of summary numbers found in source",
            "mean accuracy",
            "number_accuracy.png",
        ),
    ]:
        path = output_dir / filename
        _bar_plot(
            title=title,
            ylabel=ylabel,
            values_per_condition=_per_condition_metric(
                per_example, metric, conditions
            ),
            output_path=path,
            conditions=conditions,
            sweep_index=sweep_index,
        )
        written.append(path)

    scatter_path = output_dir / "faithfulness_vs_quality.png"
    _scatter_faithfulness_vs_quality(
        per_example, scatter_path, conditions=conditions, sweep_index=sweep_index
    )
    written.append(scatter_path)

    activation_path = output_dir / "selective_activation.png"
    _selective_activation_plot(
        per_example,
        activation_path,
        conditions=conditions,
        sweep_index=sweep_index,
    )
    written.append(activation_path)

    sweep_path = output_dir / "soft_penalty_sweep.png"
    if _soft_sweep_curve(per_example, sweep_path, conditions) is not None:
        written.append(sweep_path)

    return written


# CLI


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="Path to the experiment JSON")
    parser.add_argument(
        "--output-dir",
        default="results/figures",
        help="Directory to write PNG figures into",
    )
    args = parser.parse_args(argv)

    result = json.loads(Path(args.results).read_text())
    plot_all(result, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
