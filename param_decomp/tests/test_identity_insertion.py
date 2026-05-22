"""Test identity insertion functionality."""

from typing import override

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from param_decomp.configs import ModulePatternInfoConfig
from param_decomp.identity_insertion import insert_identity_operations_
from param_decomp.models.components import Identity


class SimpleModel(nn.Module):
    """Simple test model with multiple linear layers."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(100, d_model)
        self.layer1 = nn.Linear(d_model, d_model)
        self.layer2 = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, 100)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return self.output(x)


DEVICE = "cpu"

BATCH_SIZE = 2
SEQ_LEN = 10


def random_input():
    return torch.randint(0, 100, (BATCH_SIZE, SEQ_LEN), device=DEVICE)


def test_inserts_identity_layers():
    model = SimpleModel(d_model=32).to(DEVICE)
    model.eval()

    insert_identity_operations_(
        target_model=model,
        identity_module_info=[
            ModulePatternInfoConfig(module_pattern="layer1", C=1),
            ModulePatternInfoConfig(module_pattern="layer2", C=1),
        ],
    )

    assert hasattr(model.layer1, "pre_identity")
    assert hasattr(model.layer2, "pre_identity")
    assert isinstance(model.layer1.pre_identity, Identity)
    assert isinstance(model.layer2.pre_identity, Identity)
    assert model.layer1.pre_identity.d == 32
    assert model.layer2.pre_identity.d == 32

    assert not hasattr(model.embedding, "pre_identity")
    assert not hasattr(model.output, "pre_identity")


def test_adds_hooks():
    model = SimpleModel(d_model=32).to(DEVICE)
    model.eval()

    assert len(model.layer1._forward_hooks) == 0
    assert len(model.layer2._forward_hooks) == 0

    insert_identity_operations_(
        target_model=model,
        identity_module_info=[
            ModulePatternInfoConfig(module_pattern="layer1", C=1),
            ModulePatternInfoConfig(module_pattern="layer2", C=1),
        ],
    )

    assert len(model.layer1._forward_pre_hooks) == 1
    assert len(model.layer2._forward_pre_hooks) == 1


def test_preserves_output():
    """Test that inserting identity operations doesn't change model output."""
    model = SimpleModel(d_model=32).to(DEVICE)
    model.eval()

    input_ids = random_input()

    original_output = model(input_ids)

    insert_identity_operations_(
        model, identity_module_info=[ModulePatternInfoConfig(module_pattern="layer1", C=1)]
    )

    new_output = model(input_ids)

    assert_close(original_output, new_output, atol=1e-6, rtol=1e-6)


def test_uses_correct_dims():
    """Test identity insertion with layers of different dimensions."""

    class VaryingDimModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 64)
            self.layer1 = nn.Linear(64, 128)
            self.layer2 = nn.Linear(128, 256)
            self.output = nn.Linear(256, 100)

        @override
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embedding(x)
            x = self.layer1(x)
            x = torch.relu(x)
            x = self.layer2(x)
            return self.output(x)

    model = VaryingDimModel().to(DEVICE)
    model.eval()

    # Insert identity only before layer1 (which takes 64-dim input)
    insert_identity_operations_(
        target_model=model,
        identity_module_info=[
            ModulePatternInfoConfig(module_pattern="layer1", C=1),
            ModulePatternInfoConfig(module_pattern="layer2", C=1),
        ],
    )

    # Check that identity has correct dimension
    assert isinstance(model.layer1.pre_identity, Identity)
    assert model.layer1.pre_identity.d == 64

    assert isinstance(model.layer2.pre_identity, Identity)
    assert model.layer2.pre_identity.d == 128

    assert not hasattr(model.embedding, "pre_identity")
    assert not hasattr(model.output, "pre_identity")


def test_empty_patterns():
    """Test that empty patterns don't break anything."""
    model = SimpleModel().to(DEVICE)

    # No patterns should result in no modifications
    insert_identity_operations_(target_model=model, identity_module_info=[])

    # No identity layers should be added
    assert not hasattr(model.embedding, "pre_identity")
    assert not hasattr(model.layer1, "pre_identity")
    assert not hasattr(model.layer2, "pre_identity")
    assert not hasattr(model.output, "pre_identity")


def test_embedding_raises_error():
    model = SimpleModel(d_model=32).to("cpu")

    with pytest.raises(ValueError, match="Embedding modules not supported"):
        insert_identity_operations_(
            target_model=model,
            identity_module_info=[ModulePatternInfoConfig(module_pattern="embedding", C=1)],
        )


def test_unmatched_pattern_raises_error():
    model = SimpleModel(d_model=32).to("cpu")

    with pytest.raises(ValueError, match="did not match any modules"):
        insert_identity_operations_(
            target_model=model,
            identity_module_info=[ModulePatternInfoConfig(module_pattern="does.not.exist*", C=1)],
        )
