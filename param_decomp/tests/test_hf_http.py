"""The HF HTTP retry guard is idempotent and a clean no-op without huggingface_hub."""

import importlib

import param_decomp.hf_http as hf_http


def test_configure_is_idempotent_and_no_ops_without_hub():
    importlib.reload(hf_http)
    assert hf_http._configured is False
    hf_http.configure_hf_http_retries()
    assert hf_http._configured is True
    # second call must be a no-op (process-global guard), matching the torch guard.
    hf_http.configure_hf_http_retries()
    assert hf_http._configured is True
