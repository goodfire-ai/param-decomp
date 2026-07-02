"""Port a legacy harvest.db (pre-unification JSON-blob format) into v2 scope shards.

    python -m param_decomp_lab.scope.migrate_legacy_harvest p-19645bf7 h-20260612_000000 \
        --tokenizer_name meta-llama/Llama-3.1-8B

Kept as a legacy bridge (like import_labels.py): new harvests write shards natively via
harvest/scope_writer.py, but pre-unification harvest.db files remain on disk and this is
the only way to view them in scope. Ports the FULL legacy reservoir pool (no top-k trim),
matching native v2 semantics.

Legacy dbs are tens of GB of JSON blobs with no useful indexes, so the tool makes exactly
two sequential table scans: a cheap one for per-site component counts, then one unordered
pass feeding every site's shard writer at once (component indices address the mmap
directly, so row order is irrelevant). Legacy dbs stored only fired components and only
the firing ratio: component indices never seen become dead slots (store-all: shard idx ==
component idx) and firing_count is recovered as round(density * n_tokens_seen), which
round-trips exactly because the legacy density was computed as count / n_tokens_seen.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import fire
import numpy as np
import orjson

from param_decomp.log import logger
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.scope.artifacts import (
    FORMAT_VERSION,
    ComponentExamples,
    SiteMeta,
    SiteShardWriter,
)

LEGACY_SEQ_LEN = 512
CI_ACT_TYPE = "causal_importance"
COMPONENT_ACT_TYPE = "component_activation"


def _parse_examples(examples_blob: bytes, window: int) -> tuple[ComponentExamples, float]:
    """(examples, max_ci) — max_ci is the peak CI over all real positions, matching the
    native writer's max_act semantics."""
    raw = orjson.loads(examples_blob)
    n = len(raw)
    token_ids = np.zeros((n, window), dtype=np.uint32)
    firings = np.zeros((n, window), dtype=np.uint8)
    ci = np.zeros((n, window), dtype=np.float16)
    act = np.zeros((n, window), dtype=np.float16)
    lengths = np.zeros(n, dtype=np.uint16)
    max_ci = 0.0
    for j, ex in enumerate(raw):
        length = len(ex["token_ids"])
        assert 0 < length <= window, (length, window)
        lengths[j] = length
        token_ids[j, :length] = ex["token_ids"]
        firings[j, :length] = ex["firings"]
        ci_row = ex["activations"][CI_ACT_TYPE]
        ci[j, :length] = ci_row
        act[j, :length] = ex["activations"][COMPONENT_ACT_TYPE]
        max_ci = max(max_ci, max(ci_row))
    examples = ComponentExamples(
        token_ids=token_ids, firings=firings, ci=ci, act=act, lengths=lengths
    )
    return examples, max_ci


def _write_dead(writer: SiteShardWriter, idx: int, window: int) -> None:
    writer.write_component(
        idx=idx,
        examples=ComponentExamples(
            token_ids=np.zeros((0, window), np.uint32),
            firings=np.zeros((0, window), np.uint8),
            ci=np.zeros((0, window), np.float16),
            act=np.zeros((0, window), np.float16),
            lengths=np.zeros((0,), np.uint16),
        ),
        firing_count=0,
        firing_density=0.0,
        max_act=0.0,
        mean_ci=0.0,
        mean_act=0.0,
        input_pmi=[],
        output_pmi=[],
    )


def _pmi_top(pmi_blob: bytes, top_k: int) -> list[tuple[int, float]]:
    return [(int(t), float(p)) for t, p in orjson.loads(pmi_blob)["top"][:top_k]]


def migrate_subrun(run_id: str, harvest_subrun_id: str, tokenizer_name: str) -> list[Path]:
    src = PARAM_DECOMP_OUT_DIR / "runs" / run_id / "harvest" / harvest_subrun_id / "harvest.db"
    assert src.exists(), src
    db = sqlite3.connect(f"file:{src}?mode=ro", uri=True)

    cfg = dict(db.execute("SELECT key, value FROM config"))
    window = 2 * int(cfg["activation_context_tokens_per_side"]) + 1
    k_examples = int(cfg["activation_examples_per_component"])
    pmi_top_k = int(cfg["pmi_token_top_k"])
    n_tokens_seen = int(cfg["n_batches"]) * int(cfg["batch_size"]) * LEGACY_SEQ_LEN
    created_at = datetime.now(UTC).isoformat()

    site_components = {
        layer: int(max_idx) + 1
        for layer, max_idx in db.execute(
            "SELECT layer, MAX(component_idx) FROM components GROUP BY layer"
        )
    }
    logger.info(f"{harvest_subrun_id}: sites {site_components}, k={k_examples}, W={window}")

    writers: dict[str, SiteShardWriter] = {}
    seen: dict[str, np.ndarray] = {}
    for site, n_components in site_components.items():
        meta = SiteMeta(
            format_version=FORMAT_VERSION,
            run_id=run_id,
            site=site,
            subrun_id=harvest_subrun_id,
            n_components=n_components,
            k_examples=k_examples,
            window=window,
            tokenizer_name=tokenizer_name,
            n_tokens_seen=n_tokens_seen,
            pmi_top_k=pmi_top_k,
            provenance=f"migrated from legacy harvest/{harvest_subrun_id}/harvest.db",
            created_at=created_at,
        )
        writers[site] = SiteShardWriter(meta)
        seen[site] = np.zeros(n_components, dtype=np.bool_)

    cursor = db.execute(
        """SELECT layer, component_idx, firing_density, mean_activations,
                  activation_examples, input_token_pmi, output_token_pmi
           FROM components"""
    )
    for n_done, (site, idx, density, mean_acts_blob, ex_blob, in_pmi, out_pmi) in enumerate(
        cursor, start=1
    ):
        examples, max_ci = _parse_examples(ex_blob, window)
        mean_acts = orjson.loads(mean_acts_blob)
        writers[site].write_component(
            idx=idx,
            examples=examples,
            firing_count=round(density * n_tokens_seen),
            firing_density=density,
            max_act=max_ci,
            mean_ci=mean_acts[CI_ACT_TYPE],
            mean_act=mean_acts[COMPONENT_ACT_TYPE],
            input_pmi=_pmi_top(in_pmi, pmi_top_k),
            output_pmi=_pmi_top(out_pmi, pmi_top_k),
        )
        seen[site][idx] = True
        if n_done % 5000 == 0:
            logger.info(f"{n_done} components migrated")

    published: list[Path] = []
    for site, writer in writers.items():
        for idx in np.flatnonzero(~seen[site]):
            _write_dead(writer, int(idx), window)
        published.append(writer.publish())
        logger.info(f"published {published[-1]}")
    return published


if __name__ == "__main__":
    fire.Fire(migrate_subrun)
