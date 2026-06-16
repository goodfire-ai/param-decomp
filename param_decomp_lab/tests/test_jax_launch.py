"""The lab-side wrapper validator (`pd-jax-lm`, torch venv) and the runtime loader
(`jax_single_pool.torch_config`, jax venv) share one key set; the jax side can't be
imported here, so this exercises only the lab half. The shared-set contract itself is
asserted in `param_decomp_config` terms below."""

from pathlib import Path

import pytest
import yaml

from param_decomp_config.jax_wrapper import (
    RUN_ID_KEY,
    SUBMIT_MINTED_KEYS,
    WRAPPER_KEYS,
    WRAPPER_KEYS_BEFORE_SUBMIT,
    WRAPPER_OPTIONAL_KEYS,
)
from param_decomp_lab.experiments.lm.jax_launch import _validate_wrapper


def test_submit_minted_keys_reconstitute_the_required_set():
    assert WRAPPER_KEYS_BEFORE_SUBMIT | {RUN_ID_KEY} == WRAPPER_KEYS
    assert {RUN_ID_KEY} | WRAPPER_OPTIONAL_KEYS == SUBMIT_MINTED_KEYS


def test_validate_wrapper_rejects_unexpected_key(tmp_path: Path):
    wrapper = tmp_path / "wrapper.yaml"
    raw = {key: "x" for key in WRAPPER_KEYS_BEFORE_SUBMIT} | {"bogus_key": "x"}
    wrapper.write_text(yaml.safe_dump(raw))
    with pytest.raises(AssertionError, match="keys must be"):
        _validate_wrapper(wrapper)
