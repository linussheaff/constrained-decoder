"""Pre-cache the evaluation dataset.

Usage:

    python scripts/download_data.py
    python scripts/download_data.py --dataset EdinburghNLP/xsum --split test
    python scripts/download_data.py --dataset Samsung/samsum --split test
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger("download_data")


def download(dataset: str, splits: list[str]) -> None:
    from datasets import load_dataset

    for split in splits:
        logger.info("Caching %s split=%s", dataset, split)
        ds = load_dataset(dataset, split=split)
        logger.info("  %d examples", len(ds))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="EdinburghNLP/xsum")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test", "validation"],
        help="Which splits to pre-cache",
    )
    args = parser.parse_args(argv)
    download(args.dataset, args.splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
