"""Regression test for filter_ci_to_included_nodes accepting canonical node keys."""

import pytest
import torch

from spd.app.backend.compute import filter_ci_to_included_nodes
from spd.pretrain.models.gpt2_simple import GPT2Simple, GPT2SimpleConfig
from spd.topology import TransformerTopology


@pytest.fixture
def topology() -> TransformerTopology:
    model = GPT2Simple(
        GPT2SimpleConfig(
            model_type="GPT2Simple",
            block_size=8,
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_embd=16,
            flash_attention=False,
        )
    )
    model.eval()
    return TransformerTopology(model)


def test_filter_ci_accepts_canonical_keys(topology: TransformerTopology):
    """Canonical layer keys (e.g. '0.mlp.up') must be translated to the concrete paths
    used as keys in ci_lower_leaky (e.g. 'h.0.mlp.c_fc'). Before the fix, passing
    canonical keys raised an assertion ("Nodes reference invalid layers")."""
    seq_len, n_components = 3, 4
    mlp_up_path = topology.canon_to_target("0.mlp.up")
    attn_q_path = topology.canon_to_target("0.attn.q")
    assert mlp_up_path == "h.0.mlp.c_fc"
    assert attn_q_path == "h.0.attn.q_proj"

    ci_lower_leaky = {
        mlp_up_path: torch.arange(1, 1 + seq_len * n_components, dtype=torch.float32)
        .reshape(1, seq_len, n_components)
        .clone(),
        attn_q_path: torch.arange(100, 100 + seq_len * n_components, dtype=torch.float32)
        .reshape(1, seq_len, n_components)
        .clone(),
    }

    included_nodes = {"0.mlp.up:1:2", "0.attn.q:0:0"}
    filtered = filter_ci_to_included_nodes(ci_lower_leaky, included_nodes, topology)

    assert set(filtered.keys()) == set(ci_lower_leaky.keys())

    # Only the selected (seq_pos, c_idx) in each layer retains its original CI value.
    assert filtered[mlp_up_path][0, 1, 2] == ci_lower_leaky[mlp_up_path][0, 1, 2]
    assert filtered[attn_q_path][0, 0, 0] == ci_lower_leaky[attn_q_path][0, 0, 0]

    # Everything else is zeroed.
    mlp_expected = torch.zeros_like(ci_lower_leaky[mlp_up_path])
    mlp_expected[0, 1, 2] = ci_lower_leaky[mlp_up_path][0, 1, 2]
    assert torch.equal(filtered[mlp_up_path], mlp_expected)

    attn_expected = torch.zeros_like(ci_lower_leaky[attn_q_path])
    attn_expected[0, 0, 0] = ci_lower_leaky[attn_q_path][0, 0, 0]
    assert torch.equal(filtered[attn_q_path], attn_expected)


def test_filter_ci_empty_included_zeros_everything(topology: TransformerTopology):
    seq_len, n_components = 2, 3
    mlp_up_path = topology.canon_to_target("0.mlp.up")

    ci_lower_leaky = {
        mlp_up_path: torch.ones(1, seq_len, n_components),
    }
    filtered = filter_ci_to_included_nodes(ci_lower_leaky, set(), topology)
    assert torch.equal(filtered[mlp_up_path], torch.zeros(1, seq_len, n_components))
