"""Shared-FS layout of the persistent XLA compilation cache."""

import stat
from pathlib import Path


def ensure_group_writable_cache_dirs(cache_dir: Path) -> None:
    """Pre-create the XLA cache dirs group-writable so first-creator identity stops mattering.

    The cache dir is shared across users, but XLA itself creates the per-fusion autotune
    subdir (auto-enabled alongside the persistent cache on jax >= 0.10, see
    `jax_persistent_cache_enable_xla_caches`) and its `tmp/` with hardcoded 0755 —
    umask-independent — so whichever user's run touched the cache first left everyone
    else dying with PERMISSION_DENIED at the first train step. Creating the dirs here,
    before jax initializes the cache, removes that dependence. Dirs that already have
    group write are left alone; a group-unwritable dir owned by someone else makes the
    chmod fail fast at startup instead of one compile (~25 min) later.
    """
    autotune_dir = cache_dir / "xla_gpu_per_fusion_autotune_cache_dir"
    for d in (cache_dir, autotune_dir, autotune_dir / "tmp"):
        d.mkdir(parents=True, exist_ok=True)
        if d.stat().st_mode & stat.S_IWGRP == 0:
            d.chmod(0o2775)
