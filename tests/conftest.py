"""Shared pytest fixtures.

We keep the project tests dependency-light: a deterministic fake tokenizer
covers the cases the entity extractor needs without pulling in HuggingFace,
and a spaCy model fixture skips cleanly if ``en_core_web_sm`` isn't installed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# Make ``src`` importable without an installed package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeTokenizer:
    """Deterministic whitespace/punctuation tokenizer for tests.

    Splits on whitespace and punctuation, preserving a leading-space marker so
    that ``"Lloyds"`` and ``" Lloyds"`` map to different token IDs (mirroring
    real BPE behaviour). IDs are assigned in first-seen order so tests can
    reason about them concretely.
    """

    _TOKEN_RE = re.compile(r"\s+|[^\w\s]|\w+")

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._next_id: int = 0

    def _intern(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = self._next_id
            self._next_id += 1
        return self._vocab[tok]

    def _split(self, text: str) -> list[str]:
        # Treat each whitespace run as a "leading-space" marker prefixing the
        # following word, e.g. " Banking" -> "▁Banking".
        out: list[str] = []
        pending_space = False
        for match in self._TOKEN_RE.finditer(text):
            piece = match.group()
            if piece.isspace():
                pending_space = True
                continue
            if pending_space:
                out.append("▁" + piece)
                pending_space = False
            else:
                out.append(piece)
        return out

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text:
            return []
        return [self._intern(tok) for tok in self._split(text)]

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        """Inverse of ``encode``: ``encode`` then ``decode`` reproduces the
        input text up to whitespace normalisation."""
        inv = {v: k for k, v in self._vocab.items()}
        out: list[str] = []
        for tok_id in token_ids:
            piece = inv.get(int(tok_id), "")
            if piece.startswith("▁"):
                out.append(" " + piece[1:])
            else:
                out.append(piece)
        return "".join(out)

    @property
    def vocab(self) -> dict[str, int]:
        return dict(self._vocab)


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture(scope="session")
def spacy_nlp():
    """Return a loaded ``en_core_web_sm`` pipeline or skip the test."""
    spacy = pytest.importorskip("spacy")
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip(
            "spaCy model 'en_core_web_sm' not installed; "
            "run: python -m spacy download en_core_web_sm"
        )
