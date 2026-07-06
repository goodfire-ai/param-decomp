import torch

from param_decomp.metrics.output import _clean_metric_output


def test_scalar_tensor_gets_namespace_and_metric_name() -> None:
    out = _clean_metric_output(log_namespace="loss", metric_name="MyLoss", computed_raw=torch.tensor(2.0))
    assert out == {"loss/MyLoss": 2.0}


def test_dict_keys_are_namespace_prefixed() -> None:
    out = _clean_metric_output(
        log_namespace="loss",
        metric_name="MyLoss",
        computed_raw={"MyLoss": torch.tensor(1.0), "MyLoss_extra": torch.tensor(2.0)},
    )
    assert out == {"loss/MyLoss": 1.0, "loss/MyLoss_extra": 2.0}


def test_fully_qualified_keys_pass_through_unprefixed() -> None:
    """A key already carrying a namespace (contains `/`) bypasses `log_namespace`,
    letting one metric emit a secondary key outside its headline namespace."""
    out = _clean_metric_output(
        log_namespace="loss",
        metric_name="MyLoss",
        computed_raw={"MyLoss": torch.tensor(1.0), "imp_min/MyLoss_no_beta": torch.tensor(2.0)},
    )
    assert out == {"loss/MyLoss": 1.0, "imp_min/MyLoss_no_beta": 2.0}
