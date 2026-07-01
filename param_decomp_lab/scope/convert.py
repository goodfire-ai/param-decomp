"""Convert legacy harvest.db subruns into scope artifact shards.

    python -m param_decomp_lab.scope.convert p-19645bf7 h-20260611_210341 \
        --k_examples 30 --tokenizer_name meta-llama/Llama-3.1-8B

Reads one legacy harvest subrun (which may hold several sites' components), writes one
scope shard per site found. Examples are ranked by causal importance at the firing
(center) position; the top `k_examples` of the legacy uniform-400 pool are kept.
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import fire
import numpy as np

from param_decomp.log import logger
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.scope.artifacts import (
    FORMAT_VERSION,
    ComponentExamples,
    SiteMeta,
    SiteShardWriter,
)


def _parse_examples(examples_json: bytes, window: int) -> ComponentExamples:
    raw = json.loads(examples_json)
    n = len(raw)
    token_ids = np.zeros((n, window), dtype=np.uint32)
    firings = np.zeros((n, window), dtype=np.uint8)
    ci = np.zeros((n, window), dtype=np.float16)
    act = np.zeros((n, window), dtype=np.float16)
    for i, ex in enumerate(raw):
        w = len(ex["token_ids"])
        assert w <= window, (w, window)
        token_ids[i, :w] = ex["token_ids"]
        firings[i, :w] = ex["firings"]
        ci[i, :w] = ex["activations"]["causal_importance"]
        act[i, :w] = ex["activations"]["component_activation"]
    return ComponentExamples(token_ids=token_ids, firings=firings, ci=ci, act=act)


def _center_ci(ex: ComponentExamples) -> np.ndarray:
    center = ex.token_ids.shape[1] // 2
    return ex.ci[:, center].astype(np.float32)


def _top_k(ex: ComponentExamples, k: int) -> ComponentExamples:
    order = np.argsort(-_center_ci(ex), kind="stable")[:k]
    return ComponentExamples(
        token_ids=ex.token_ids[order],
        firings=ex.firings[order],
        ci=ex.ci[order],
        act=ex.act[order],
    )


def _pmi_top(pmi_json: bytes, top_k: int) -> list[tuple[int, float]]:
    return [(int(t), float(p)) for t, p in json.loads(pmi_json)["top"][:top_k]]


def convert_subrun(
    run_id: str,
    harvest_subrun_id: str,
    k_examples: int,
    tokenizer_name: str,
    window: int = 41,
    pmi_top_k: int = 40,
) -> list[Path]:
    src = PARAM_DECOMP_OUT_DIR / "runs" / run_id / "harvest" / harvest_subrun_id / "harvest.db"
    assert src.exists(), src
    db = sqlite3.connect(f"file:{src}?mode=ro", uri=True)

    n_tokens_seen = _infer_tokens_seen(db)
    sites = [
        (layer, n)
        for layer, n in db.execute(
            "SELECT layer, COUNT(*) FROM components GROUP BY layer ORDER BY layer"
        )
    ]
    logger.info(f"{harvest_subrun_id}: {len(sites)} site(s): {sites}")

    published: list[Path] = []
    for site, n_components in sites:
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
            provenance=f"converted from harvest/{harvest_subrun_id}/harvest.db",
            created_at=datetime.now(UTC).isoformat(),
        )
        writer = SiteShardWriter(meta)
        cursor = db.execute(
            """SELECT component_idx, firing_density, mean_activations, activation_examples,
                      input_token_pmi, output_token_pmi
               FROM components WHERE layer = ?""",
            (site,),
        )
        for i, (idx, density, mean_acts_json, ex_json, in_pmi, out_pmi) in enumerate(cursor):
            full = _parse_examples(ex_json, window)
            kept = _top_k(full, k_examples)
            center_ci = _center_ci(full)
            mean_acts = json.loads(mean_acts_json)
            writer.write_component(
                idx=idx,
                examples=kept,
                firing_count=round(density * n_tokens_seen),
                firing_density=density,
                max_act=float(center_ci.max()) if len(center_ci) else 0.0,
                mean_ci=mean_acts["causal_importance"],
                mean_act=mean_acts["component_activation"],
                input_pmi=_pmi_top(in_pmi, pmi_top_k),
                output_pmi=_pmi_top(out_pmi, pmi_top_k),
            )
            if (i + 1) % 2000 == 0:
                logger.info(f"{site}: {i + 1}/{n_components}")
        published.append(writer.publish())
        logger.info(f"published {published[-1]}")
    return published


def _infer_tokens_seen(db: sqlite3.Connection) -> int:
    cfg = dict(db.execute("SELECT key, value FROM config"))
    seq_len = 512
    return int(cfg["n_batches"]) * int(cfg["batch_size"]) * seq_len


def import_legacy_labels(run_id: str, interp_subrun_id: str) -> int:
    """Copy labels from a legacy autointerp interp.db into scope/labels.db."""
    from param_decomp_lab.scope.artifacts import open_labels_db

    src = PARAM_DECOMP_OUT_DIR / "runs" / run_id / "autointerp" / interp_subrun_id / "interp.db"
    assert src.exists(), src
    legacy = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    dst = open_labels_db(run_id, readonly=False)
    n = 0
    for key, label in legacy.execute("SELECT component_key, label FROM interpretations"):
        site, _, idx = key.rpartition(":")
        dst.execute(
            "INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?)",
            (
                site,
                int(idx),
                label,
                "google/gemini-3.1-pro-preview",
                None,
                datetime.now(UTC).isoformat(),
                f"imported from autointerp/{interp_subrun_id}",
            ),
        )
        n += 1
    dst.commit()
    logger.info(f"imported {n} labels from {interp_subrun_id}")
    return n


if __name__ == "__main__":
    fire.Fire({"subrun": convert_subrun, "labels": import_legacy_labels})
