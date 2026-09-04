"""Checkpoint compatibility helpers."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import TypeVar

_StateValue = TypeVar('_StateValue')
_COMPILED_MODULE_SEGMENT = '._orig_mod.'


def restore_compiled_module_keys(
    state_dict: MutableMapping[str, _StateValue],
    expected_keys: Iterable[str],
) -> None:
    """Restore ``torch.compile`` key segments removed during checkpoint save.

    Only keys whose normalized form exactly matches a key expected by the
    current model are changed. Unknown keys remain untouched so strict state
    dictionary loading can still report genuine incompatibilities.
    """
    portable_to_expected: dict[str, str] = {}
    for expected_key in expected_keys:
        portable_key = expected_key.replace(_COMPILED_MODULE_SEGMENT, '.')
        if portable_key == expected_key:
            continue
        previous_key = portable_to_expected.setdefault(portable_key, expected_key)
        if previous_key != expected_key:
            raise RuntimeError(
                'Multiple compiled model keys normalize to '
                f"{portable_key!r}: {previous_key!r} and {expected_key!r}."
            )

    for portable_key, expected_key in portable_to_expected.items():
        if portable_key in state_dict and expected_key not in state_dict:
            state_dict[expected_key] = state_dict.pop(portable_key)
