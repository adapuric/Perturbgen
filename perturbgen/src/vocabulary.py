"""Target-vocabulary allocation helpers."""

from __future__ import annotations

from collections.abc import Iterable


def next_available_token_id(
    token_sequences: Iterable[Iterable[int]],
    *,
    reserved_token_ids: Iterable[int] = (),
) -> int:
    """Return an ID above every observed or reserved target token ID."""
    largest_token_id = -1
    for sequence in token_sequences:
        largest_token_id = max(largest_token_id, max(sequence, default=-1))
    largest_token_id = max(
        largest_token_id,
        max(reserved_token_ids, default=-1),
    )
    if largest_token_id < 0:
        raise ValueError('Cannot allocate a vocabulary from empty token IDs.')
    return largest_token_id + 1
