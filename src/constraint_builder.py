"""Build token allowlists and prefix tries from extracted source facts.

Consumes a :class:src.entity_extractor.SourceFacts object and produces the
data structures the constrained decoder needs at generation time:

* :class:TokenAllowlist — a flat frozenset[int] of token IDs permitted
  inside a factual span. Cheap to apply (one boolean mask over the vocab) but
  ignores entity-level sequence structure.

* :class:EntityTrie — a prefix trie over the tokenised source-entity
  sequences. Supports non-deterministic traversal: at every step we track the
  set of trie nodes consistent with the tokens generated so far. This lets
  the decoder constrain multi-token entities to actual continuations of
  entities that appeared in the source (e.g. allow "Ll" followed only by
  "oyds" if "Lloyds Banking Group" appeared in the source).

The two structures are deliberately decoupled: the always-on fully
constrained baseline only needs the flat allowlist, while the selective
decoder will use both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

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

    __slots__ = ("children", "is_terminal", "depth")

    def __init__(self, depth: int = 0) -> None:
        self.children: dict[int, "TrieNode"] = {}
        self.is_terminal: bool = False
        self.depth: int = depth

    def __repr__(self) -> str:  # pragma: no cover — debug only
        return (
            f"TrieNode(depth={self.depth}, "
            f"n_children={len(self.children)}, terminal={self.is_terminal})"
        )


class EntityTrie:
    """Prefix trie over tokenised source-entity sequences.

    Supports non-deterministic traversal: at each step we track the *set* of
    trie nodes consistent with the tokens generated so far (a token may
    simultaneously continue one partial entity and start another). The trie
    answers "what tokens are allowed next?" without committing to a single
    interpretation of the prefix.

    The trie itself is stateless. Callers carry their own ``set[TrieNode]``
    live set, obtained from :meth:`new_run` and advanced via :meth:`step`.
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
        (i.e. start tokens of a new entity) via :meth:`new_run` before calling
        this method — see :func:`extend_live_after_terminal`.
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


__all__ = [
    "TokenAllowlist",
    "TrieNode",
    "EntityTrie",
    "GroundedConstraint",
    "build_constraint",
    "build_allowlist",
    "extend_live_after_terminal",
]
