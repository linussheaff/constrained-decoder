"""SummaC-ZS: NLI-based inconsistency detection for summaries.

The algorithm:

1. Split the source document into sentences (premises) and the summary into
   sentences (hypotheses).
2. Run the NLI model over every (source sentence, summary sentence) pair,
   producing an entailment/contradiction "image" of shape
   ``(2, n_source_sents, n_summary_sents)``.
3. For each summary sentence take the **max** entailment and the **max**
   contradiction over source sentences, and score it ``ent - con``.
4. The document-level score is the **mean** over summary sentences.

Scores live in ``[-1, 1]``: 1.0 means every summary sentence is entailed by
some source sentence, negative means contradiction dominates.

Unlike the entity/number metrics in :mod:`src.evaluate_faithfulness`, this is
a model-based metric — it catches relational hallucinations ("X acquired Y"
when the source says the reverse) that copy-based checks are blind to.

The NLI model is loaded lazily and cached per (model, device) pair, so
scoring thousands of summaries only pays the load cost once.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover — typing only
    import numpy as np

logger = logging.getLogger(__name__)


# The `vitc` weights from the SummaC release: ALBERT-xlarge fine-tuned on
# VitaminC + MNLI. ~235M params, ~900MB download.
DEFAULT_NLI_MODEL: str = "tals/albert-xlarge-vitaminc-mnli"

# Label names differ across NLI checkpoints. The vitc weights come from
# VitaminC/FEVER and use SUPPORTS/REFUTES; MNLI-trained models use
# entailment/contradiction.
_ENTAILMENT_ALIASES: frozenset[str] = frozenset(
    {"entailment", "entail", "supports", "support"}
)
_CONTRADICTION_ALIASES: frozenset[str] = frozenset(
    {"contradiction", "contradict", "refutes", "refute"}
)

# Fallback label positions, used only when the model config carries no usable
# ``id2label`` mapping (e.g. placeholder LABEL_0/LABEL_1 names).
_FALLBACK_LABEL_INDICES: dict[str, tuple[int, int]] = {
    # model_name -> (entailment_idx, contradiction_idx)
    "tals/albert-xlarge-vitaminc-mnli": (0, 1),
    "roberta-large-mnli": (2, 0),
    "microsoft/deberta-base-mnli": (2, 0),
}

# Documents get long; SummaC truncates the premise list rather than paying
# quadratic cost on a 100-sentence article.
_DEFAULT_MAX_SOURCE_SENTS: int = 100

# Naive sentence splitter used when spaCy isn't available. Splits on
# sentence-final punctuation followed by whitespace and a capital/digit.
_SENTENCE_FALLBACK_REGEX = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


# Report


@dataclass
class SummaCReport:
    """SummaC-ZS result for a single (summary, source) pair."""

    summary: str
    source: str
    score: float
    sentence_scores: list[float] = field(default_factory=list)
    summary_sentences: list[str] = field(default_factory=list)
    n_source_sentences: int = 0

    @property
    def n_summary_sentences(self) -> int:
        return len(self.summary_sentences)

    @property
    def normalised_score(self) -> float:
        """``score`` rescaled from ``[-1, 1]`` to ``[0, 1]``.

        Handy for plotting on the same axes as the precision/ROUGE metrics;
        the raw ``score`` is what the SummaC paper reports.
        """
        return (self.score + 1.0) / 2.0

    @property
    def min_sentence_score(self) -> float:
        """Score of the least-supported summary sentence.

        More sensitive than the mean for short summaries where a single
        hallucinated clause is the whole failure.
        """
        if not self.sentence_scores:
            return 0.0
        return min(self.sentence_scores)


# Sentence splitting


def split_sentences(text: str, spacy_model: str = "en_core_web_sm") -> list[str]:
    """Split ``text`` into sentences, preferring spaCy over the regex fallback.

    spaCy is already a hard dependency for entity extraction, so we reuse its
    cached pipeline. If the model isn't installed we degrade to a punctuation
    heuristic rather than failing — the metric stays computable in the
    dependency-light test environment.
    """
    stripped = text.strip()
    if not stripped:
        return []
    try:
        from .entity_extractor import _load_spacy

        nlp = _load_spacy(spacy_model)
    except Exception:  # spaCy or the model unavailable
        logger.debug("spaCy unavailable for sentence splitting; using regex fallback")
        return [s.strip() for s in _SENTENCE_FALLBACK_REGEX.split(stripped) if s.strip()]

    doc = nlp(stripped)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences or [stripped]


# NLI model loading


@lru_cache(maxsize=2)
def _load_nli(model_name: str, device: str) -> tuple[Any, Any, int, int]:
    """Load and cache an NLI model, returning (tokenizer, model, ent_idx, con_idx)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading NLI model %s onto %s", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    ent_idx, con_idx = _resolve_label_indices(model.config, model_name)
    logger.info(
        "%s label indices: entailment=%d contradiction=%d",
        model_name,
        ent_idx,
        con_idx,
    )
    return tokenizer, model, ent_idx, con_idx


def _resolve_label_indices(config: Any, model_name: str) -> tuple[int, int]:
    """Find the entailment and contradiction logit positions for a model.

    Label ordering *and* naming differ between NLI checkpoints (vitc puts
    SUPPORTS first, ``roberta-large-mnli`` puts entailment last), and getting
    it wrong silently inverts the metric. Read ``id2label`` when it's
    meaningful and only fall back to a hard-coded table for known checkpoints.
    """
    id2label = getattr(config, "id2label", None) or {}
    ent_idx: int | None = None
    con_idx: int | None = None
    for idx, label in id2label.items():
        if not isinstance(label, str):
            continue
        name = label.strip().lower()
        if name in _ENTAILMENT_ALIASES:
            ent_idx = int(idx)
        elif name in _CONTRADICTION_ALIASES:
            con_idx = int(idx)
    if ent_idx is not None and con_idx is not None:
        return ent_idx, con_idx

    fallback = _FALLBACK_LABEL_INDICES.get(model_name)
    if fallback is not None:
        logger.warning(
            "%s has no usable id2label mapping; using known indices %s",
            model_name,
            fallback,
        )
        return fallback

    raise ValueError(
        f"Cannot determine entailment/contradiction label positions for "
        f"{model_name!r}: id2label={id2label!r}. Pass a model whose config "
        f"labels the classes, or add it to _FALLBACK_LABEL_INDICES."
    )


# Scorer


class SummaCZS:
    """Zero-shot SummaC scorer.

    Args:
        model_name: HuggingFace NLI checkpoint. Defaults to the ``vitc``
            weights used by the SummaC paper.
        device: ``"cuda"``, ``"cpu"``, ... Auto-detected when None.
        batch_size: NLI pairs scored per forward pass.
        use_ent, use_con: Which channels feed the score. The paper's default
            (both) scores ``entailment - contradiction``.
        max_source_sentences: Truncate the premise list to this many sentences.
        max_length: Token truncation length for each (premise, hypothesis) pair.
        spacy_model: Pipeline used for sentence splitting.

    Example:
        >>> scorer = SummaCZS()                          # doctest: +SKIP
        >>> scorer.score(summary, article).score         # doctest: +SKIP
        0.82
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
        use_ent: bool = True,
        use_con: bool = True,
        max_source_sentences: int = _DEFAULT_MAX_SOURCE_SENTS,
        max_length: int = 512,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        if not use_ent and not use_con:
            raise ValueError("At least one of use_ent/use_con must be True.")
        self.model_name = model_name
        self.device = device or _auto_device()
        self.batch_size = batch_size
        self.use_ent = use_ent
        self.use_con = use_con
        self.max_source_sentences = max_source_sentences
        self.max_length = max_length
        self.spacy_model = spacy_model
        self._loaded: tuple[Any, Any, int, int] | None = None

    # Lazy load so constructing a scorer is free and importing this module
    # never touches the network.
    def _ensure_loaded(self) -> tuple[Any, Any, int, int]:
        if self._loaded is None:
            self._loaded = _load_nli(self.model_name, self.device)
        return self._loaded

    def nli_probabilities(self, pairs: Sequence[tuple[str, str]]) -> "np.ndarray":
        """Return an ``(n_pairs, 2)`` array of (entailment, contradiction) probs."""
        import numpy as np
        import torch

        if not pairs:
            return np.zeros((0, 2), dtype=float)

        tokenizer, model, ent_idx, con_idx = self._ensure_loaded()
        out = np.zeros((len(pairs), 2), dtype=float)
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            batch = tokenizer(
                [premise for premise, _ in chunk],
                [hypothesis for _, hypothesis in chunk],
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**batch).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            out[start : start + len(chunk), 0] = probs[:, ent_idx]
            out[start : start + len(chunk), 1] = probs[:, con_idx]
        return out

    def build_image(self, summary: str, source: str) -> "np.ndarray":
        """Build the SummaC image of shape ``(2, n_source_sents, n_summary_sents)``.

        Channel 0 is entailment probability, channel 1 contradiction.
        """
        import numpy as np

        source_sents = split_sentences(source, self.spacy_model)[
            : self.max_source_sentences
        ]
        summary_sents = split_sentences(summary, self.spacy_model)
        if not source_sents or not summary_sents:
            return np.zeros((2, len(source_sents), len(summary_sents)), dtype=float)

        pairs = [(src, hyp) for hyp in summary_sents for src in source_sents]
        probs = self.nli_probabilities(pairs)
        # `pairs` varies source fastest within each summary sentence, so the
        # reshape lands as (n_summary, n_source, 2) before the transpose.
        image = probs.reshape(len(summary_sents), len(source_sents), 2)
        return np.transpose(image, (2, 1, 0))

    def image_to_scores(self, image: "np.ndarray") -> list[float]:
        """Reduce an image to one score per summary sentence.

        Takes the max over source sentences independently per channel (the
        "does *any* source sentence support this?" question) and combines the
        enabled channels.
        """
        import numpy as np

        if image.size == 0 or image.shape[1] == 0 or image.shape[2] == 0:
            return [0.0] * (image.shape[2] if image.ndim == 3 else 0)

        ent = np.max(image[0], axis=0)
        con = np.max(image[1], axis=0)
        if self.use_ent and self.use_con:
            scores = ent - con
        elif self.use_ent:
            scores = ent
        else:
            scores = -con
        return [float(s) for s in scores]

    def score(self, summary: str, source: str) -> SummaCReport:
        """Score one (summary, source) pair.

        An empty summary or source yields 0.0 — neutral rather than
        rewarded, since there's nothing to entail either way.
        """
        summary_sents = split_sentences(summary, self.spacy_model)
        source_sents = split_sentences(source, self.spacy_model)[
            : self.max_source_sentences
        ]
        if not summary_sents or not source_sents:
            return SummaCReport(
                summary=summary,
                source=source,
                score=0.0,
                sentence_scores=[],
                summary_sentences=summary_sents,
                n_source_sentences=len(source_sents),
            )

        image = self.build_image(summary, source)
        sentence_scores = self.image_to_scores(image)
        mean_score = sum(sentence_scores) / len(sentence_scores)
        return SummaCReport(
            summary=summary,
            source=source,
            score=float(mean_score),
            sentence_scores=sentence_scores,
            summary_sentences=summary_sents,
            n_source_sentences=len(source_sents),
        )

    def score_many(self, pairs: Iterable[tuple[str, str]]) -> list[SummaCReport]:
        """Score an iterable of ``(summary, source)`` pairs."""
        return [self.score(summary, source) for summary, source in pairs]


def _auto_device() -> str:
    try:
        import torch
    except ImportError:  # pragma: no cover — import guard
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=2)
def get_default_scorer(
    model_name: str = DEFAULT_NLI_MODEL, device: str | None = None
) -> SummaCZS:
    """Return a process-wide cached :class:`SummaCZS` for the given weights."""
    return SummaCZS(model_name=model_name, device=device)


def evaluate_summac(
    summary: str,
    source: str,
    *,
    scorer: SummaCZS | None = None,
    model_name: str = DEFAULT_NLI_MODEL,
    device: str | None = None,
) -> SummaCReport:
    """Convenience wrapper mirroring :func:`src.evaluate_faithfulness.evaluate_faithfulness`.

    Pass ``scorer`` when evaluating many summaries so the NLI model is loaded
    once; otherwise a cached default scorer is used.
    """
    if scorer is None:
        scorer = get_default_scorer(model_name, device)
    return scorer.score(summary, source)


__all__ = [
    "DEFAULT_NLI_MODEL",
    "SummaCReport",
    "SummaCZS",
    "evaluate_summac",
    "get_default_scorer",
    "split_sentences",
]
