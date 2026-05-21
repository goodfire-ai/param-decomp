import inspect

import param_decomp


def test_public_api_exports_resolve() -> None:
    for name in param_decomp.__all__:
        assert getattr(param_decomp, name)


def test_public_training_entrypoint_accepts_device() -> None:
    assert "device" in inspect.signature(param_decomp.optimize).parameters


def test_public_run_sink_is_protocol() -> None:
    assert getattr(param_decomp.RunSink, "_is_protocol", False)
