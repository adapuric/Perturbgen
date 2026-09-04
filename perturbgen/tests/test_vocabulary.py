"""Regression tests for target-vocabulary allocation."""

import pytest

from perturbgen.src.vocabulary import next_available_token_id


def test_finds_global_maximum_across_nonlexicographic_sequences():
    token_sequences = [[9, 1], [8, 100]]

    assert next_available_token_id(token_sequences) == 101


def test_reserves_ids_for_unobserved_mapped_genes():
    token_sequences = [[0, 4, 7], [0, 5]]

    assert (
        next_available_token_id(
            token_sequences,
            reserved_token_ids=[0, 1, 2, 3, 4, 5, 6, 7, 12],
        )
        == 13
    )


def test_rejects_empty_token_vocabulary():
    with pytest.raises(ValueError, match='empty token IDs'):
        next_available_token_id([[], []])
