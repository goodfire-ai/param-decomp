"""The one definition of the persistent XLA compilation-cache policy."""

from pathlib import Path

import jax


def enable_persistent_compilation_cache(runs_dir: Path) -> Path:
    """Cache compiled XLA executables to a shared-FS dir reused across runs/requeues.

    Every jax entrypoint (trainer, pretrainer, harvest/clustering workers) calls this at
    startup so a compile is keyed by HLO + backend + topology + jax/xla version and a
    matching re-compile (requeue, another worker of the same fan-out, a fresh run at the
    same config+topology) loads from disk in seconds. The dir is a SIBLING of `runs_dir`
    (not per-run, not inside the immutable per-run workspace) so every run and consumer
    shares it. Multi-host safe: jax gates writes on `process_id == 0`; every rank reads.
    Independent single-process workers all write, but entries land via atomic rename, so
    same-key writers just overwrite each other with identical bytes.

    Call before the first compile that should cache; the cache initializes lazily at
    first compile (before/after `init_distributed` both work — the write gate reads the
    distributed state at write time, not here). Rank-0's HLO dump (`XLA_FLAGS`
    `--xla_dump_*`, see the LM composition root's `_enable_hlo_dump`) does not fork
    rank 0's cache key: jax strips debug/dump options when computing it.

    Do NOT add `jax_persistent_cache_enable_xla_caches='all'` — it crashes at runtime on
    jax 0.10.1/B200 (`CUDA_ERROR_NOT_FOUND`); the default already persists the
    per-fusion autotune cache on jax >= 0.10.
    """
    cache_dir = runs_dir.parent / "xla_compilation_cache"
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    # 5s (jax default: 1s): high enough that trivial slice/utility jits don't churn the
    # dir, low enough that every step-level compile caches — eval tiers and init fans
    # (1-60s) recompiled on every requeue under the old 60s threshold, and the worker
    # forward must not miss the cache just because it compiles in under a minute.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return cache_dir
