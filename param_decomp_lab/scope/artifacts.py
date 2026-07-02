"""Scope artifact store: per-(run, site, subrun) binary activation-example shards.

Layout under `PARAM_DECOMP_OUT_DIR/runs/<run>/scope/`:

    <site>/<subrun>/meta.json       SiteMeta — shapes, tokenizer, provenance
    <site>/<subrun>/examples.bin    fixed-shape arrays, mmap-addressable by component idx
    <site>/<subrun>/site.db         per-component scalars + top-k PMI (sqlite, indexed)
    labels.db                       (site, component_idx) -> label + provenance

Subruns are attempts: readers use the newest complete subrun of a site; subruns are
never combined. A subrun dir is written under a `.tmp-*` name and atomically renamed,
so `meta.json` existing under the final name == the subrun is complete and readable.

examples.bin holds, in order, for C components with at most K examples of W tokens:
    n_examples   u16 [C]        example slots filled per component
    example_len  u16 [C, K]     real token count per example (rest is zero pad)
    token_ids    u32 [C, K, W]
    firings      u8  [C, K, W]
    ci           f16 [C, K, W]
    act          f16 [C, K, W]
Examples are left-packed to `example_len[c, j]` and zero-padded to W; unused example
slots are zero. Per-component reads are O(K*W) seeks into the mmap.
"""

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR

FORMAT_VERSION = 2
_FIELD_DTYPES = {"token_ids": np.uint32, "firings": np.uint8, "ci": np.float16, "act": np.float16}


def scope_dir(run_id: str) -> Path:
    return PARAM_DECOMP_OUT_DIR / "runs" / run_id / "scope"


@dataclass(frozen=True)
class SiteMeta:
    format_version: int
    run_id: str
    site: str
    subrun_id: str
    n_components: int
    k_examples: int
    window: int
    tokenizer_name: str
    n_tokens_seen: int
    pmi_top_k: int
    provenance: str
    created_at: str

    @classmethod
    def load(cls, subrun_dir: Path) -> "SiteMeta":
        meta = cls(**json.loads((subrun_dir / "meta.json").read_text()))
        assert meta.format_version == FORMAT_VERSION, (
            f"scope format v{meta.format_version} at {subrun_dir}, reader is v{FORMAT_VERSION}"
        )
        return meta


def _array_offsets(c: int, k: int, w: int) -> dict[str, tuple[int, np.dtype, tuple[int, ...]]]:
    """Field name -> (byte offset, dtype, shape) within examples.bin."""
    offsets: dict[str, tuple[int, np.dtype, tuple[int, ...]]] = {}
    cursor = 0

    def add(name: str, dtype: np.dtype, shape: tuple[int, ...]) -> None:
        nonlocal cursor
        offsets[name] = (cursor, dtype, shape)
        cursor += int(np.prod(shape)) * dtype.itemsize

    add("n_examples", np.dtype(np.uint16), (c,))
    add("example_len", np.dtype(np.uint16), (c, k))
    for name, dt in _FIELD_DTYPES.items():
        add(name, np.dtype(dt), (c, k, w))
    return offsets


@dataclass(frozen=True)
class ComponentExamples:
    """One component's examples: arrays of shape [n_examples, W], each left-packed to
    `lengths[j]` real tokens and zero-padded to W."""

    token_ids: np.ndarray
    firings: np.ndarray
    ci: np.ndarray
    act: np.ndarray
    lengths: np.ndarray


class SiteShardWriter:
    """Writes one (run, site, subrun) shard; `publish()` makes it visible atomically."""

    def __init__(self, meta: SiteMeta) -> None:
        self.meta = meta
        final = scope_dir(meta.run_id) / meta.site / meta.subrun_id
        assert not final.exists(), f"subrun already published: {final}"
        self._final_dir = final
        self._tmp_dir = final.with_name(f".tmp-{meta.subrun_id}-{os.getpid()}")
        self._tmp_dir.mkdir(parents=True)

        c, k, w = meta.n_components, meta.k_examples, meta.window
        self._offsets = _array_offsets(c, k, w)
        total_bytes = max(
            off + int(np.prod(shape)) * dt.itemsize for off, dt, shape in self._offsets.values()
        )
        self._bin = np.memmap(
            self._tmp_dir / "examples.bin", dtype=np.uint8, mode="w+", shape=(total_bytes,)
        )

        self._db = sqlite3.connect(self._tmp_dir / "site.db")
        self._db.execute("""
            CREATE TABLE components (
                idx INTEGER PRIMARY KEY,
                firing_count INTEGER NOT NULL,
                firing_density REAL NOT NULL,
                max_act REAL NOT NULL,
                mean_ci REAL NOT NULL,
                mean_act REAL NOT NULL,
                n_examples INTEGER NOT NULL,
                input_pmi TEXT NOT NULL,
                output_pmi TEXT NOT NULL
            )""")
        self._n_written = 0

    def _field(self, name: str) -> np.ndarray:
        off, dt, shape = self._offsets[name]
        return np.frombuffer(self._bin, dtype=dt, count=int(np.prod(shape)), offset=off).reshape(
            shape
        )

    def write_component(
        self,
        idx: int,
        examples: ComponentExamples,
        firing_count: int,
        firing_density: float,
        max_act: float,
        mean_ci: float,
        mean_act: float,
        input_pmi: list[tuple[int, float]],
        output_pmi: list[tuple[int, float]],
    ) -> None:
        n, w = examples.token_ids.shape
        assert n <= self.meta.k_examples and w == self.meta.window, (n, w)
        assert examples.lengths.shape == (n,), examples.lengths.shape
        self._field("n_examples")[idx] = n
        self._field("example_len")[idx, :n] = examples.lengths
        for name in _FIELD_DTYPES:
            getattr_arr: np.ndarray = getattr(examples, name)
            self._field(name)[idx, :n] = getattr_arr
        self._db.execute(
            "INSERT INTO components VALUES (?,?,?,?,?,?,?,?,?)",
            (
                idx,
                firing_count,
                firing_density,
                max_act,
                mean_ci,
                mean_act,
                n,
                json.dumps(input_pmi),
                json.dumps(output_pmi),
            ),
        )
        self._n_written += 1

    def publish(self) -> Path:
        assert self._n_written == self.meta.n_components, (
            f"wrote {self._n_written} of {self.meta.n_components} components"
        )
        self._bin.flush()
        self._db.execute("CREATE INDEX idx_density ON components(firing_density DESC)")
        self._db.execute("CREATE INDEX idx_max_act ON components(max_act DESC)")
        self._db.commit()
        self._db.close()
        (self._tmp_dir / "meta.json").write_text(json.dumps(asdict(self.meta), indent=2))
        self._final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(self._tmp_dir, self._final_dir)
        return self._final_dir


class SiteShardReader:
    """Random access into one published (run, site, subrun) shard."""

    def __init__(self, subrun_dir: Path) -> None:
        self.meta = SiteMeta.load(subrun_dir)
        m = self.meta
        self._offsets = _array_offsets(m.n_components, m.k_examples, m.window)
        self._bin = np.memmap(subrun_dir / "examples.bin", dtype=np.uint8, mode="r")
        # Readers are cached and served from FastAPI's threadpool. check_same_thread=False
        # permits cross-thread use, but CPython's sqlite3 execute() is not atomic across
        # threads on one connection — concurrent callers must serialize access themselves
        # (the scope backend does, via store._DB_LOCK).
        self.db = sqlite3.connect(
            f"file:{subrun_dir / 'site.db'}?mode=ro", uri=True, check_same_thread=False
        )

    def examples(self, idx: int) -> ComponentExamples:
        assert 0 <= idx < self.meta.n_components, idx
        off, dt, shape = self._offsets["n_examples"]
        n = int(np.frombuffer(self._bin, dtype=dt, count=shape[0], offset=off)[idx])
        loff, ldt, lshape = self._offsets["example_len"]
        _, lk = lshape
        lengths = np.frombuffer(
            self._bin, dtype=ldt, count=lk, offset=loff + idx * lk * ldt.itemsize
        )[:n].copy()
        fields = {}
        for name in _FIELD_DTYPES:
            foff, fdt, fshape = self._offsets[name]
            _, k, w = fshape
            row_bytes = k * w * fdt.itemsize
            arr = np.frombuffer(self._bin, dtype=fdt, count=k * w, offset=foff + idx * row_bytes)
            fields[name] = arr.reshape(k, w)[:n].copy()
        return ComponentExamples(**fields, lengths=lengths)


def find_subruns(run_id: str, site: str) -> list[Path]:
    """Published subrun dirs for a site, oldest first. Newest-complete-wins: take [-1]."""
    site_dir = scope_dir(run_id) / site
    if not site_dir.exists():
        return []
    return sorted(d for d in site_dir.iterdir() if d.is_dir() and (d / "meta.json").exists())


LABELS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS labels (
        site TEXT NOT NULL,
        component_idx INTEGER NOT NULL,
        label TEXT NOT NULL,
        model TEXT NOT NULL,
        cost_usd REAL,
        created_at TEXT NOT NULL,
        provenance TEXT NOT NULL,
        PRIMARY KEY (site, component_idx)
    )"""


def open_labels_db(run_id: str, readonly: bool) -> sqlite3.Connection:
    path = scope_dir(run_id) / "labels.db"
    if readonly:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(LABELS_SCHEMA)
    return conn
