"""Copy component labels from a legacy autointerp interp.db into scope/labels.db.

    python -m param_decomp_lab.scope.import_labels p-19645bf7 a-20260611_210341

Component examples/scalars are written natively by harvest (see
`param_decomp_lab/harvest/scope_writer.py`); this is the one remaining legacy bridge,
carrying autointerp labels into the scope store the viewer reads.
"""

import sqlite3
from datetime import UTC, datetime

import fire

from param_decomp.log import logger
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.scope.artifacts import open_labels_db


def import_legacy_labels(run_id: str, interp_subrun_id: str) -> int:
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
    fire.Fire(import_legacy_labels)
