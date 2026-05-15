"""Tests for ``src.constraint_builder``.

Covers TokenAllowlist behaviour, EntityTrie insertion / step / lookup,
deduplication, GroundedConstraint, build_constraint end-to-end, and the
extend_live_after_terminal helper.
"""

from __future__ import annotations

import pytest

from src.constraint_builder import (
    EntityTrie,
    GroundedConstraint,
    TokenAllowlist,
    TrieNode,
    build_allowlist,
    build_constraint,
    extend_live_after_terminal,
)
from src.entity_extractor import Entity, Number, SourceFacts


# ---------------------------------------------------------------------------
# TokenAllowlist
# ---------------------------------------------------------------------------


class TestTokenAllowlist:
    def test_contains_and_len(self) -> None:
        a = TokenAllowlist(frozenset({1, 2, 3}))
        assert 1 in a
        assert 4 not in a
        assert len(a) == 3

    def test_iteration_order_agnostic(self) -> None:
        a = TokenAllowlist(frozenset({5, 9, 1}))
        assert sorted(a) == [1, 5, 9]

    def test_empty(self) -> None:
        a = TokenAllowlist(frozenset())
        assert a.is_empty
        assert len(a) == 0


# ---------------------------------------------------------------------------
# EntityTrie
# ---------------------------------------------------------------------------


class TestEntityTrieBasic:
    def test_empty_insert_is_noop(self) -> None:
        trie = EntityTrie()
        trie.insert([])
        assert trie.num_entities == 0
        assert trie.start_tokens == set()

    def test_single_insert_and_lookup(self) -> None:
        trie = EntityTrie()
        trie.insert([1, 2, 3])
        assert trie.num_entities == 1
        assert trie.start_tokens == {1}

        live = trie.new_run()
        assert trie.allowed_next(live) == {1}
        live = trie.step(live, 1)
        assert trie.allowed_next(live) == {2}
        live = trie.step(live, 2)
        assert trie.allowed_next(live) == {3}
        live = trie.step(live, 3)
        assert trie.any_terminal(live)
        assert trie.allowed_next(live) == set()  # no continuations

    def test_branching_sequences(self) -> None:
        trie = EntityTrie()
        trie.insert([1, 2, 3])
        trie.insert([1, 2, 4])
        assert trie.num_entities == 2

        live = trie.new_run()
        live = trie.step(live, 1)
        live = trie.step(live, 2)
        assert trie.allowed_next(live) == {3, 4}

    def test_off_trie_token_empties_live_set(self) -> None:
        trie = EntityTrie()
        trie.insert([1, 2, 3])
        live = trie.new_run()
        live = trie.step(live, 99)  # unknown
        assert live == set()
        assert trie.allowed_next(live) == set()

    def test_terminal_with_continuation(self) -> None:
        # "Lloyds" is itself a valid entity AND "Lloyds Banking Group" extends it.
        trie = EntityTrie()
        trie.insert([1])
        trie.insert([1, 2, 3])
        live = trie.new_run()
        live = trie.step(live, 1)
        assert trie.any_terminal(live)
        assert trie.allowed_next(live) == {2}

    def test_duplicate_insert_is_idempotent(self) -> None:
        trie = EntityTrie()
        trie.insert([1, 2])
        trie.insert([1, 2])
        trie.insert([1, 2])
        assert trie.num_entities == 1

    def test_insert_many(self) -> None:
        trie = EntityTrie()
        trie.insert_many([[1], [2], [3, 4]])
        assert trie.num_entities == 3
        assert trie.start_tokens == {1, 2, 3}

    def test_paths_lists_all_terminals(self) -> None:
        trie = EntityTrie()
        trie.insert_many([[1, 2], [1, 3], [4, 5, 6]])
        paths = sorted(trie.paths())
        assert paths == [[1, 2], [1, 3], [4, 5, 6]]

    def test_root_is_not_terminal_by_default(self) -> None:
        trie = EntityTrie()
        assert not trie.root.is_terminal

    def test_step_does_not_mutate_input_live_set(self) -> None:
        trie = EntityTrie()
        trie.insert([1, 2])
        live = trie.new_run()
        before = set(live)
        _ = trie.step(live, 1)
        assert live == before


class TestExtendLiveAfterTerminal:
    def test_adds_root_when_terminal_present(self) -> None:
        trie = EntityTrie()
        trie.insert([1])
        trie.insert([2])
        live = trie.new_run()
        live = trie.step(live, 1)  # terminal
        extended = extend_live_after_terminal(trie, live)
        assert trie.root in extended
        # We can now start a new entity → token 2 should be allowed.
        assert 2 in trie.allowed_next(extended)

    def test_no_change_when_no_terminal(self) -> None:
        trie = EntityTrie()
        trie.insert([1, 2, 3])
        live = trie.new_run()
        live = trie.step(live, 1)  # not terminal
        assert extend_live_after_terminal(trie, live) == live


# ---------------------------------------------------------------------------
# GroundedConstraint + build_constraint
# ---------------------------------------------------------------------------


def _make_facts(
    entity_tokens: set[int] = frozenset(),
    number_tokens: set[int] = frozenset(),
    entity_sequences: list[list[int]] | None = None,
    number_sequences: list[list[int]] | None = None,
) -> SourceFacts:
    """Build a minimal SourceFacts for unit tests, no spaCy needed."""
    return SourceFacts(
        source_text="(test)",
        entities=[Entity("Foo", "ORG", 0, 3)] if entity_sequences else [],
        numbers=[Number("42", 42.0, 4, 6)] if number_sequences else [],
        entity_tokens=set(entity_tokens),
        number_tokens=set(number_tokens),
        entity_token_sequences=entity_sequences or [],
        number_token_sequences=number_sequences or [],
    )


class TestBuildConstraint:
    def test_allowlist_is_union_of_entity_and_number_tokens(self) -> None:
        facts = _make_facts(
            entity_tokens={1, 2, 3},
            number_tokens={3, 4, 5},
            entity_sequences=[[1, 2, 3]],
            number_sequences=[[3, 4, 5]],
        )
        c = build_constraint(facts)
        assert c.allowlist.token_ids == frozenset({1, 2, 3, 4, 5})
        assert c.num_entities == 2

    def test_extra_token_ids_added_to_allowlist_only(self) -> None:
        facts = _make_facts(
            entity_tokens={1, 2},
            entity_sequences=[[1, 2]],
        )
        c = build_constraint(facts, extra_token_ids=[100, 200])
        assert {100, 200}.issubset(c.allowlist.token_ids)
        # ...but they should not appear in the trie.
        assert c.trie.start_tokens == {1}

    def test_vocab_size_lower_bound(self) -> None:
        facts = _make_facts(
            entity_tokens={1, 50, 7},
            entity_sequences=[[1, 50, 7]],
        )
        c = build_constraint(facts)
        assert c.vocab_size_lower_bound == 51

    def test_empty_facts_produce_empty_constraint(self) -> None:
        facts = SourceFacts(source_text="", entities=[], numbers=[])
        c = build_constraint(facts)
        assert c.allowlist.is_empty
        assert c.num_entities == 0
        assert c.vocab_size_lower_bound == 0

    def test_build_allowlist_convenience(self) -> None:
        facts = _make_facts(
            entity_tokens={1, 2}, entity_sequences=[[1, 2]]
        )
        a = build_allowlist(facts, extra_token_ids=[9])
        assert a.token_ids == frozenset({1, 2, 9})

    def test_trie_round_trip(self) -> None:
        facts = _make_facts(
            entity_tokens={10, 11, 12, 20, 21},
            entity_sequences=[[10, 11, 12], [20, 21]],
        )
        c = build_constraint(facts)
        paths = sorted(c.trie.paths())
        assert paths == [[10, 11, 12], [20, 21]]


# ---------------------------------------------------------------------------
# Integration: build a constraint from a real spaCy extraction
# ---------------------------------------------------------------------------


class TestBuildConstraintIntegration:
    def test_round_trip_from_extract_facts(self, fake_tokenizer, spacy_nlp) -> None:
        from src.entity_extractor import extract_facts

        source = "Lloyds Banking Group reported a profit of £2.1 billion in 2023."
        facts = extract_facts(source, fake_tokenizer)
        constraint = build_constraint(facts, extra_token_ids=[999])

        # Every per-entity tokenisation must be a valid trie path.
        for seq in facts.entity_token_sequences:
            live = constraint.trie.new_run()
            for tok in seq:
                live = constraint.trie.step(live, tok)
                assert live, f"trie went off-path on entity token {tok}"
            assert constraint.trie.any_terminal(live)

        # The flat allowlist must cover every entity/number token plus extras.
        assert facts.factual_token_ids.issubset(constraint.allowlist.token_ids)
        assert 999 in constraint.allowlist.token_ids
