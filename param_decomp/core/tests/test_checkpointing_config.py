"""`Cadence.checkpointing` is a closed union — `periodic` (save_every + retention) or
`none`. The periodic arm's validity is already pinned by the seat parse gate
(`test_repo_configs_parse`); here we pin the `none` arm's spelling and that the union
fails closed on the shapes it must refuse: the incident's invalid combination (a save
cadence alongside `none`) and the retired flat spelling stored pins still carry."""

import pytest
from pydantic import ValidationError

from param_decomp.core.configs import Cadence, NoCheckpointing


def test_none_arm_validates():
    cadence = Cadence.model_validate({"train_log_every": 10, "checkpointing": {"kind": "none"}})
    assert isinstance(cadence.checkpointing, NoCheckpointing)


@pytest.mark.parametrize(
    "cadence_raw",
    [
        pytest.param(
            {"train_log_every": 10, "checkpointing": {"kind": "none", "save_every": 1000}},
            id="none-with-save-cadence",
        ),
        pytest.param(
            {
                "train_log_every": 10,
                "checkpointing": {"kind": "none", "retention": {"kind": "keep_all"}},
            },
            id="none-with-retention",
        ),
        pytest.param(
            {
                "train_log_every": 10,
                "save_every": 1000,
                "checkpoint_retention": {"kind": "keep_last", "n": 2},
            },
            id="retired-flat-spelling",
        ),
    ],
)
def test_invalid_checkpointing_shapes_refuse(cadence_raw: dict[str, object]):
    with pytest.raises(ValidationError):
        Cadence.model_validate(cadence_raw)
