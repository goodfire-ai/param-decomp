"""The pretrain store resolver (`infra.pretrain_cache.resolved_cache_dir`): a complete
entry must short-circuit without any network I/O — that is what makes resolution safe on
every rank's cold start and on requeue — and run references must spell out their entity."""

import socket
from pathlib import Path

import pytest

from param_decomp.infra import pretrain_cache


def _write_complete_entry(data_root: Path, run_path: str) -> Path:
    cache_dir = pretrain_cache.cache_dir_for_run(data_root, run_path)
    cache_dir.mkdir(parents=True)
    (cache_dir / pretrain_cache.MODEL_CONFIG_FILENAME).write_text("n_layer: 1\n")
    (cache_dir / "model_step_100.safetensors").write_bytes(b"")
    return cache_dir


def test_complete_entry_resolves_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_path = "entity/proj/runs/t-0a1b2c3d"
    cache_dir = _write_complete_entry(tmp_path, run_path)

    def _no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("resolved_cache_dir touched the network on a complete entry")

    monkeypatch.setattr(socket, "getaddrinfo", _no_network)
    monkeypatch.setattr(socket, "socket", _no_network)
    assert pretrain_cache.resolved_cache_dir(tmp_path, run_path) == cache_dir


def test_incomplete_entry_is_not_complete(tmp_path: Path) -> None:
    run_path = "entity/proj/runs/t-0a1b2c3d"
    cache_dir = _write_complete_entry(tmp_path, run_path)
    (cache_dir / "model_step_200.safetensors").write_bytes(b"")
    assert not pretrain_cache.is_complete(cache_dir)


def test_bare_run_id_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="entity"):
        pretrain_cache.cache_dir_for_run(tmp_path, "t-0a1b2c3d")
