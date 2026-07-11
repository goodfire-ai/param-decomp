import stat
from pathlib import Path

from param_decomp.compile_cache import ensure_group_writable_cache_dirs


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


def test_creates_cache_tree_group_writable(tmp_path: Path):
    cache_dir = tmp_path / "xla_compilation_cache"
    ensure_group_writable_cache_dirs(cache_dir)
    autotune_dir = cache_dir / "xla_gpu_per_fusion_autotune_cache_dir"
    for d in (cache_dir, autotune_dir, autotune_dir / "tmp"):
        assert _mode(d) & stat.S_IWGRP


def test_repairs_group_unwritable_dirs(tmp_path: Path):
    cache_dir = tmp_path / "xla_compilation_cache"
    autotune_dir = cache_dir / "xla_gpu_per_fusion_autotune_cache_dir"
    autotune_dir.mkdir(parents=True)
    autotune_dir.chmod(0o755)
    (autotune_dir / "tmp").mkdir(mode=0o755)
    ensure_group_writable_cache_dirs(cache_dir)
    assert _mode(autotune_dir) & stat.S_IWGRP
    assert _mode(autotune_dir / "tmp") & stat.S_IWGRP


def test_leaves_already_writable_dirs_alone(tmp_path: Path):
    cache_dir = tmp_path / "xla_compilation_cache"
    ensure_group_writable_cache_dirs(cache_dir)
    cache_dir.chmod(0o2770)
    ensure_group_writable_cache_dirs(cache_dir)
    assert _mode(cache_dir) == 0o2770
