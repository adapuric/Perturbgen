"""Regression tests for checkpoint key compatibility."""

import pytest
import torch
from torch import nn

from perturbgen.src.checkpoint import restore_compiled_module_keys


class _EncoderWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 2)


class _CompiledModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layers = torch.compile(_EncoderWrapper())


def _portable_state_dict(model):
    return {
        key.replace('._orig_mod.', '.'): value.clone()
        for key, value in model.state_dict().items()
    }


def test_restores_normalized_keys_for_strict_compiled_model_load():
    model = _CompiledModel()
    state_dict = _portable_state_dict(model)

    with pytest.raises(RuntimeError, match='Missing key'):
        model.load_state_dict(state_dict, strict=True)

    restore_compiled_module_keys(state_dict, model.state_dict().keys())

    incompatible = model.load_state_dict(state_dict, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_preserves_already_compiled_keys():
    model = _CompiledModel()
    state_dict = model.state_dict()
    original_keys = tuple(state_dict)

    restore_compiled_module_keys(state_dict, model.state_dict().keys())

    assert tuple(state_dict) == original_keys
    incompatible = model.load_state_dict(state_dict, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_preserves_unknown_keys_for_strict_load_validation():
    model = _CompiledModel()
    state_dict = _portable_state_dict(model)
    state_dict['unrelated.weight'] = torch.ones(1)

    restore_compiled_module_keys(state_dict, model.state_dict().keys())

    with pytest.raises(RuntimeError, match='Unexpected key'):
        model.load_state_dict(state_dict, strict=True)
