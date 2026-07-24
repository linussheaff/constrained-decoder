"""Build token allowlists and prefix tries from extracted source facts.

Consumes a :class:src.entity_extractor.SourceFacts object and produces the
data structures the constrained decoder needs at generation time:

* :class:TokenAllowlist — a flat frozenset[int] of token IDs permitted
  inside a factual span. Cheap to apply (one boolean mask over the vocab) but
  ignores entity-level sequence structure.

* :class:EntityTrie — a prefix trie over the tokenised source-entity
  sequences. At every step we track the set of trie nodes consistent with the 
  tokens generated so far. 
  This lets the decoder constrain multi-token entities to actual continuations of
  entities that appeared in the source (e.g. allow "Ll" followed only by
  "oyds" if "Lloyds Banking Group" appeared in the source).

* :class:`NegativeTrie` — the inverse construction. Rather than *forcing*
  generation onto source material, it identifies the vocabulary tokens that
  could only ever *begin* an entity mention absent from the source, and
  penalises exactly those. See :func:`build_negative_trie`.
"""

from __future__ import annotations

import re
import weakref
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

from .entity_extractor import SourceFacts


@dataclass(frozen=True)
class TokenAllowlist:
    """Flat set of token IDs permitted in factual spans.

    Typically the union of every token ID appearing in any tokenised entity or
    number variant from the source document, plus optional structural tokens
    (EOS, common punctuation, newline). Use :class:`EntityTrie` when token
    order matters.
    """

    token_ids: frozenset[int]

    def __contains__(self, token_id: int) -> bool:
        return token_id in self.token_ids

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids)

    @property
    def is_empty(self) -> bool:
        return not self.token_ids


class TrieNode:
    """Node in an entity prefix trie."""

    # lickle space and time save
    __slots__ = ("children", "is_terminal", "depth")

    def __init__(self, depth: int = 0) -> None:
        self.children: dict[int, "TrieNode"] = {}
        self.is_terminal: bool = False
        self.depth: int = depth

    def __repr__(self) -> str:  # no cover!!
        return (
            f"TrieNode(depth={self.depth}, "
            f"n_children={len(self.children)}, terminal={self.is_terminal})"
        )


class EntityTrie:
    """Prefix trie over tokenised source-entity sequences.

    Supports non-deterministic traversal. At each step we track the *set* of
    trie nodes consistent with the tokens generated so far (a token may
    simultaneously continue one partial entity and start another). The trie
    answers "what tokens are allowed next?" without committing to a single
    interpretation of the prefix.
    """

    def __init__(self) -> None:
        self.root: TrieNode = TrieNode(depth=0)
        self._size: int = 0  # number of distinct entity sequences inserted

    def insert(self, token_ids: Sequence[int]) -> None:
        """Insert a tokenised entity sequence into the trie. Idempotent."""
        if not token_ids:
            return
        node = self.root
        for token_id in token_ids:
            child = node.children.get(token_id)
            if child is None:
                child = TrieNode(depth=node.depth + 1)
                node.children[token_id] = child
            node = child
        if not node.is_terminal:
            node.is_terminal = True
            self._size += 1

    def insert_many(self, sequences: Iterable[Sequence[int]]) -> None:
        for seq in sequences:
            self.insert(seq)

    @property
    def num_entities(self) -> int:
        """Number of distinct entity sequences inserted."""
        return self._size

    @property
    def start_tokens(self) -> set[int]:
        """Tokens that begin some entity in the trie."""
        return set(self.root.children.keys())

    def new_run(self) -> set[TrieNode]:
        """Initial live set for a new factual span (just the root)."""
        return {self.root}

    def allowed_next(self, live: set[TrieNode]) -> set[int]:
        """Tokens permitted given a set of live trie nodes.

        Returns the union of children of all live nodes. If a live node is
        terminal, the caller is responsible for adding the root's children
        (i.e. start tokens of a new entity) 
        """
        allowed: set[int] = set()
        for node in live:
            allowed.update(node.children.keys())
        return allowed

    def step(self, live: set[TrieNode], token_id: int) -> set[TrieNode]:
        """Advance the live set by one token.

        Returns the set of trie nodes reachable from any live node via
        token_id. If the result is empty the prefix has gone off-trie; the
        caller can decide whether to reset (call :meth:`new_run`) or fail.
        """
        next_live: set[TrieNode] = set()
        for node in live:
            child = node.children.get(token_id)
            if child is not None:
                next_live.add(child)
        return next_live

    def any_terminal(self, live: set[TrieNode]) -> bool:
        """True if any live node marks the end of a complete entity."""
        return any(node.is_terminal for node in live)

    def paths(self) -> list[list[int]]:
        """All complete entity token-id sequences in the trie (debug helper)."""
        result: list[list[int]] = []

        def dfs(node: TrieNode, prefix: list[int]) -> None:
            if node.is_terminal:
                result.append(list(prefix))
            for tok_id, child in node.children.items():
                prefix.append(tok_id)
                dfs(child, prefix)
                prefix.pop()

        dfs(self.root, [])
        return result


def extend_live_after_terminal(
    trie: EntityTrie, live: set[TrieNode]
) -> set[TrieNode]:
    """Add root-as-live whenever the current live set contains a terminal node.

    Use this when the decoder allows a new entity to start immediately after
    one ends (the common case in selective decoding).
    """
    if trie.any_terminal(live):
        return live | {trie.root}
    return live


@dataclass(frozen=True)
class GroundedConstraint:
    """Bundle of allowlist + trie used by the grounded logits processors."""

    allowlist: TokenAllowlist
    trie: EntityTrie

    @property
    def num_entities(self) -> int:
        return self.trie.num_entities

    @property
    def vocab_size_lower_bound(self) -> int:
        """Smallest vocab size that can hold this allowlist (max id + 1)."""
        if not self.allowlist.token_ids:
            return 0
        return max(self.allowlist.token_ids) + 1


def build_constraint(
    facts: SourceFacts,
    extra_token_ids: Iterable[int] | None = None,
) -> GroundedConstraint:
    """Build an allowlist + entity trie from extracted source facts.

    Args:
        facts: Extracted source facts (entities, numbers, tokenised sequences).
        extra_token_ids: Optional additional token IDs to include in the
            allowlist only. Typical examples: eos_token_id,
            bos_token_id, common punctuation, newline. These are *not*
            inserted into the trie — they're outside the entity structure.

    Returns:
        A :class:GroundedConstraint combining a flat allowlist with a prefix
        trie suitable for tracking multi-token entity generation.
    """
    token_ids: set[int] = set(facts.factual_token_ids)
    if extra_token_ids is not None:
        token_ids.update(int(t) for t in extra_token_ids)

    trie = EntityTrie()
    trie.insert_many(facts.entity_token_sequences)
    trie.insert_many(facts.number_token_sequences)

    return GroundedConstraint(
        allowlist=TokenAllowlist(token_ids=frozenset(token_ids)),
        trie=trie,
    )


def build_allowlist(
    facts: SourceFacts,
    extra_token_ids: Iterable[int] | None = None,
) -> TokenAllowlist:
    """Convenience wrapper returning just the flat allowlist."""
    return build_constraint(facts, extra_token_ids).allowlist


# -----------------------
#      Negative Trie
# -----------------------


# Characters that start a monetary amount, treated like a digit.
_CURRENCY_CHARS: frozenset[str] = frozenset({"$", "£", "€", "¥"})

# Sub-word markers. SentencePiece and byte-level BPE mark the *start* of a
# word; WordPiece marks the continuation of one.
_WORD_START_MARKERS: tuple[str, ...] = ("Ġ", "▁", " ")  # "Ġ", "▁"
_CONTINUATION_MARKERS: tuple[str, ...] = ("##",)

# Closed-class words can be capitalised (sentence-initially) but can never
# *begin* a named entity, so penalising them would only cost fluency. Matched
# against the whole lower-cased token surface, so sub-word prefixes like "Th"
# are unaffected.
CLOSED_CLASS_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "this", "that", "these", "those",
        "and", "but", "or", "nor", "so", "yet", "if", "then", "than",
        "as", "because", "while", "when", "where", "after", "before",
        "he", "she", "it", "they", "we", "you", "i", "him", "her", "them",
        "his", "hers", "its", "their", "our", "your", "my",
        "is", "are", "was", "were", "be", "been", "being", "has", "have",
        "had", "do", "does", "did", "will", "would", "can", "could",
        "shall", "should", "may", "might", "must",
        "in", "on", "at", "of", "for", "from", "to", "by", "with", "about",
        "into", "over", "under", "between", "during", "against",
        "not", "no", "there", "here", "what", "which", "who", "how", "why",
    }
)

# Word-ish runs in the source: alphabetic runs, numbers (with separators and
# an optional percent sign), and currency-prefixed amounts. Each is inserted
# into the prefix trie so that a token like " Jess" or " 2." matches the
# source words "Jessica" and "2.1".
_SOURCE_WORD_REGEX = re.compile(
    r"[^\W\d_]+"
    r"|[$£€¥]?\d+(?:[.,]\d+)*%?"
)


class PrefixTrieNode:
    """Node in the character-level source-word trie."""

    __slots__ = ("children", "is_word")

    def __init__(self) -> None:
        self.children: dict[str, "PrefixTrieNode"] = {}
        self.is_word: bool = False


class SourcePrefixTrie:
    """Character trie over the words of the source document.

    A candidate passes if its surgace form is a prefix of some source word. 
    """

    def __init__(self) -> None:
        self.root = PrefixTrieNode()
        self._n_words = 0

    def insert(self, word: str) -> None:
        word = word.strip().lower()
        if not word:
            return
        node = self.root
        for char in word:
            child = node.children.get(char)
            if child is None:
                child = PrefixTrieNode()
                node.children[char] = child
            node = child
        node.is_word = True
        self._n_words += 1

    def insert_text(self, text: str) -> None:
        """Insert every word-ish run found in ``text``.

        Monetary amounts go in twice e.g., "£2.1" and "2.1". This is so that a token
        carrying the currency symbol and one carrying only the digits both
        match a source that wrote it either way.
        """
        for match in _SOURCE_WORD_REGEX.finditer(text):
            word = match.group()
            self.insert(word)
            if word and word[0] in _CURRENCY_CHARS:
                self.insert(word[1:])

    @property
    def num_words(self) -> int:
        """Number of insert calls that stored a non-empty word (with repeats)."""
        return self._n_words

    def _walk(self, text: str) -> "PrefixTrieNode | None":
        node = self.root
        for char in text.lower():
            child = node.children.get(char)
            if child is None:
                return None
            node = child
        return node

    def has_prefix(self, text: str) -> bool:
        """True if ``text`` is a prefix of some inserted word.

        The empty string is a prefix of everything. 
        """
        return self._walk(text) is not None

    def has_word(self, text: str) -> bool:
        """True if ``text`` is itself a complete inserted word."""
        node = self._walk(text)
        return node is not None and node.is_word


@dataclass(frozen=True)
class NegativeTrie:
    """Vocabulary tokens that may not *begin* an off-source entity mention.

    Unlike :class:`TokenAllowlist` this is a *blocklist* so every token outside
    it is untouched. It is safe to apply at every decoding step because the
    blocked set contains only entity-initial tokens. 
    """

    blocked_token_ids: frozenset[int]
    n_candidates: int = 0
    n_source_words: int = 0

    def __contains__(self, token_id: int) -> bool:
        return token_id in self.blocked_token_ids

    def __len__(self) -> int:
        return len(self.blocked_token_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.blocked_token_ids)

    @property
    def is_empty(self) -> bool:
        return not self.blocked_token_ids

    @property
    def blocked_fraction_of_candidates(self) -> float:
        """Share of entity-initial candidates that the source does not support."""
        if not self.n_candidates:
            return 0.0
        return len(self.blocked_token_ids) / self.n_candidates


def _piece_surface(piece: str) -> tuple[str, bool]:
    """Split a raw vocabulary piece into (surface text, is_continuation)."""
    for marker in _CONTINUATION_MARKERS:
        if piece.startswith(marker):
            return piece[len(marker) :], True
    for marker in _WORD_START_MARKERS:
        if piece.startswith(marker):
            return piece[len(marker) :], False
    return piece, False


def _is_entity_initial(surface: str) -> bool:
    """True if ``surface`` looks like the opening of an entity or number."""
    text = surface.strip()
    if not text:
        return False
    head = text[0]
    if head in _CURRENCY_CHARS or head.isdigit():
        return True
    if not (head.isalpha() and head.isupper()):
        return False
    return text.lower() not in CLOSED_CLASS_WORDS


def _prefix_key(surface: str) -> str:
    """Trailing punctuation isn't part of the word being started.
    """
    return surface.rstrip(".,;:!?)'\"”’-").strip()


def _get_vocab(tokenizer: Any) -> dict[str, int]:
    getter = getattr(tokenizer, "get_vocab", None)
    if callable(getter):
        return getter()
    vocab = getattr(tokenizer, "vocab", None)
    if isinstance(vocab, dict):
        return vocab
    raise TypeError(
        "tokenizer must expose get_vocab() or a vocab dict to build a "
        "negative trie"
    )


# Scanning a 128k-entry vocabulary is only worth doing once per tokenizer.
# Weak keys so holding the cache never keeps a tokenizer alive; 
# the stored size invalidates the entry for a vocabulary that grows in place.
_CANDIDATE_CACHE: "weakref.WeakKeyDictionary[Any, tuple[int, tuple[tuple[int, str], ...]]]" = (
    weakref.WeakKeyDictionary()
)


def entity_initial_candidates(tokenizer: Any) -> tuple[tuple[int, str], ...]:
    """Return (token_id, surface) for every vocabulary token that could open
    an entity mention. Cached per tokenizer.
    """
    vocab = _get_vocab(tokenizer)
    try:
        cached = _CANDIDATE_CACHE.get(tokenizer)
    except TypeError:  # unhashable tokenizer — just don't cache
        cached = None
    if cached is not None and cached[0] == len(vocab):
        return cached[1]

    special_ids = set(getattr(tokenizer, "all_special_ids", None) or ())
    candidates: list[tuple[int, str]] = []
    for piece, token_id in vocab.items():
        token_id = int(token_id)
        if token_id in special_ids:
            continue
        surface, is_continuation = _piece_surface(piece)
        if is_continuation:
            # A WordPiece continuation cannot begin anything.
            continue
        if _is_entity_initial(surface):
            candidates.append((token_id, surface.strip()))

    result = tuple(candidates)
    try:
        _CANDIDATE_CACHE[tokenizer] = (len(vocab), result)
    except TypeError:  # not weak-referenceable — recompute next time
        pass
    return result


def build_negative_trie(
    source: str,
    tokenizer: Any,
    *,
    facts: SourceFacts | None = None,
    extra_allowed_token_ids: Iterable[int] | None = None,
    min_prefix_match: int = 1,
) -> NegativeTrie:
    """Build the set of tokens that could only start an off-source entity.

    A candidate token is blocked unless *any* of the following says the source
    supports it:

    1. Its ID appears in the tokenisation of the source document. This protects every 
       token the model would need to copy an entity verbatim, including mid-entity 
       pieces that happen to be capitalised (``"McDonald"`` → ``"Mc"`` + ``"Donald"``).
    2. Its ID appears in the extracted factual spans (``facts``), which are
       tokenised separately and so may segment differently.
    3. Its surface is a prefix of some word in the source. 
       This catches the reverse mismatch, where the model reaches an entity 
       through a segmentation the source never used (``" Jess"`` + ``"ica"`` 
       against a source that tokenised ``" Jessica"`` whole).

    Args:
        source: The source document.
        tokenizer: Tokenizer exposing ``get_vocab()``/``vocab`` and ``encode``.
        facts: Optional pre-computed facts, to reuse their token sequences.
        extra_allowed_token_ids: Never block these (EOS, punctuation, ...).
        min_prefix_match: Minimum length for rule 3 to grant a pass.

    Returns:
        A :class:`NegativeTrie`. Blocking every member of it is safe at every
        decoding step: the model can still emit any function word, any
        continuation, any punctuation, and can still copy every entity the
        source actually contains.
    """
    trie = SourcePrefixTrie()
    trie.insert_text(source)

    allowed_ids: set[int] = set()
    try:
        allowed_ids.update(int(t) for t in tokenizer.encode(source, add_special_tokens=False))
    except TypeError:  # tokenizers whose encode takes no add_special_tokens
        allowed_ids.update(int(t) for t in tokenizer.encode(source))
    if facts is not None:
        allowed_ids.update(int(t) for t in facts.factual_token_ids)
    if extra_allowed_token_ids is not None:
        allowed_ids.update(int(t) for t in extra_allowed_token_ids)

    def _source_supports(surface: str) -> bool:
        key = _prefix_key(surface)
        if not key:
            return True
        if trie.has_word(key):
            return True
        if len(key) < min_prefix_match:
            return False
        return trie.has_prefix(key)

    candidates = entity_initial_candidates(tokenizer)
    blocked = {
        token_id
        for token_id, surface in candidates
        if token_id not in allowed_ids and not _source_supports(surface)
    }

    return NegativeTrie(
        blocked_token_ids=frozenset(blocked),
        n_candidates=len(candidates),
        n_source_words=trie.num_words,
    )


__all__ = [
    "CLOSED_CLASS_WORDS",
    "NegativeTrie",
    "PrefixTrieNode",
    "SourcePrefixTrie",
    "build_negative_trie",
    "entity_initial_candidates",
    "TokenAllowlist",
    "TrieNode",
    "EntityTrie",
    "GroundedConstraint",
    "build_constraint",
    "build_allowlist",
    "extend_live_after_terminal",
]
