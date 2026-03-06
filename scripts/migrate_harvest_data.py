"""Migrate legacy harvest + autointerp data to the new layout.

Copies data into new directories/DBs without modifying the originals.

Legacy layout:
  harvest/<run_id>/activation_contexts/harvest.db   (schema: mean_ci, ci_values, component_acts)
  harvest/<run_id>/correlations/*.pt
  harvest/<run_id>/eval/intruder/*.jsonl
  autointerp/<run_id>/interp.db                     (top-level, with scores already merged)

New layout:
  harvest/<run_id>/h-<timestamp>/harvest.db          (schema: firing_density, mean_activations, firings, activations)
  harvest/<run_id>/h-<timestamp>/*.pt
  autointerp/<run_id>/a-<timestamp>/interp.db
"""

import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import fire
import orjson
import torch

from spd.harvest.storage import TokenStatsStorage
from spd.settings import SPD_OUT_DIR


@dataclass
class RunDiagnostic:
    run_id: str
    harvest_db: bool = False
    token_stats: bool = False
    correlations: bool = False
    n_components: int = 0
    n_tokens: int = 0
    ci_threshold: str = ""
    intruder_scores: int = 0
    interp_db: bool = False
    n_interpretations: int = 0
    interp_score_types: list[str] = field(default_factory=list)
    already_migrated: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.problems and not self.already_migrated


def diagnose_run(run_id: str, timestamp: str = "20260218_000000") -> RunDiagnostic:
    harvest_root = SPD_OUT_DIR / "harvest" / run_id
    autointerp_root = SPD_OUT_DIR / "autointerp" / run_id
    diag = RunDiagnostic(run_id=run_id)

    # Already migrated?
    if (harvest_root / f"h-{timestamp}").exists():
        diag.already_migrated = True
        return diag

    # Harvest DB
    old_db = harvest_root / "activation_contexts" / "harvest.db"
    diag.harvest_db = old_db.exists()
    if not diag.harvest_db:
        diag.problems.append("missing harvest.db")
        return diag

    conn = sqlite3.connect(f"file:{old_db}?immutable=1", uri=True)
    diag.n_components = conn.execute("SELECT COUNT(*) FROM components").fetchone()[0]
    threshold_row = conn.execute("SELECT value FROM config WHERE key = 'ci_threshold'").fetchone()
    diag.ci_threshold = threshold_row[0] if threshold_row else "0.0 (default)"
    conn.close()

    # Token stats
    ts_path = harvest_root / "correlations" / "token_stats.pt"
    diag.token_stats = ts_path.exists()
    if diag.token_stats:
        ts_data = torch.load(ts_path, weights_only=False)
        diag.n_tokens = ts_data["n_tokens"]
    else:
        diag.problems.append("missing token_stats.pt")

    # Correlations
    diag.correlations = (harvest_root / "correlations" / "component_correlations.pt").exists()

    # Intruder scores
    intruder_dir = harvest_root / "eval" / "intruder"
    if intruder_dir.exists():
        jsonl_files = list(intruder_dir.glob("*.jsonl"))
        if jsonl_files:
            largest = max(jsonl_files, key=lambda f: f.stat().st_size)
            with open(largest, "rb") as f:
                diag.intruder_scores = sum(1 for _ in f)

    # Autointerp
    interp_db = autointerp_root / "interp.db"
    diag.interp_db = interp_db.exists()
    if diag.interp_db:
        conn = sqlite3.connect(f"file:{interp_db}?immutable=1", uri=True)
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        if "interpretations" in tables:
            diag.n_interpretations = conn.execute(
                "SELECT COUNT(*) FROM interpretations"
            ).fetchone()[0]
        if "scores" in tables:
            diag.interp_score_types = [
                r[0] for r in conn.execute("SELECT DISTINCT score_type FROM scores").fetchall()
            ]
        conn.close()

    return diag


def print_diagnostics(diagnostics: list[RunDiagnostic]) -> None:
    ready = [d for d in diagnostics if d.ready]
    skipped = [d for d in diagnostics if d.already_migrated]
    blocked = [d for d in diagnostics if d.problems]

    if skipped:
        print(f"Already migrated ({len(skipped)}): {', '.join(d.run_id for d in skipped)}\n")

    if blocked:
        print(f"BLOCKED ({len(blocked)}):")
        for d in blocked:
            print(f"  {d.run_id}: {', '.join(d.problems)}")
        print()

    if ready:
        print(f"Ready to migrate ({len(ready)}):")
        print(
            f"{'run_id':20s} {'comps':>6s} {'n_tokens':>14s} {'threshold':>10s} "
            f"{'intruder':>8s} {'interps':>7s} {'scores':>20s} {'corr':>5s}"
        )
        print("-" * 95)
        for d in ready:
            scores_str = ", ".join(d.interp_score_types) if d.interp_score_types else "-"
            print(
                f"{d.run_id:20s} {d.n_components:>6d} {d.n_tokens:>14,} {d.ci_threshold:>10s} "
                f"{d.intruder_scores:>8d} {d.n_interpretations:>7d} {scores_str:>20s} "
                f"{'yes' if d.correlations else 'no':>5s}"
            )
    print()


def migrate_harvest_db(old_db_path: Path, new_db_path: Path, token_stats_path: Path) -> int:
    """Copy harvest DB, transforming schema from legacy to new format.

    Old schema: mean_ci REAL, activation_examples with {token_ids, ci_values, component_acts}
    New schema: firing_density REAL, mean_activations TEXT,
                activation_examples with {token_ids, firings, activations}
    """
    old_conn = sqlite3.connect(f"file:{old_db_path}?immutable=1", uri=True)
    old_conn.row_factory = sqlite3.Row

    new_conn = sqlite3.connect(str(new_db_path))
    new_conn.execute("PRAGMA journal_mode=WAL")
    new_conn.executescript("""\
        CREATE TABLE IF NOT EXISTS components (
            component_key TEXT PRIMARY KEY,
            layer TEXT NOT NULL,
            component_idx INTEGER NOT NULL,
            firing_density REAL NOT NULL,
            mean_activations TEXT NOT NULL,
            activation_examples TEXT NOT NULL,
            input_token_pmi TEXT NOT NULL,
            output_token_pmi TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scores (
            component_key TEXT NOT NULL,
            score_type TEXT NOT NULL,
            score REAL NOT NULL,
            details TEXT NOT NULL,
            PRIMARY KEY (component_key, score_type)
        );
    """)

    # Load firing densities from token_stats.pt
    assert token_stats_path.exists(), f"No token_stats.pt at {token_stats_path}"
    ts = TokenStatsStorage.load(token_stats_path)
    firing_density_map: dict[str, float] = {}
    for i, key in enumerate(ts.component_keys):
        firing_density_map[key] = ts.firing_counts[i].item() / ts.n_tokens

    # Read activation threshold from old DB config
    threshold_row = old_conn.execute(
        "SELECT value FROM config WHERE key = 'ci_threshold'"
    ).fetchone()
    activation_threshold = float(threshold_row["value"]) if threshold_row else 0.0

    # Migrate config
    for row in old_conn.execute("SELECT key, value FROM config").fetchall():
        key = row["key"]
        value = row["value"]
        if key == "ci_threshold":
            key = "activation_threshold"
        new_conn.execute("INSERT OR REPLACE INTO config VALUES (?, ?)", (key, value))

    # Migrate components row-by-row
    n = 0
    rows = old_conn.execute("SELECT * FROM components").fetchall()
    for row in rows:
        old_examples = orjson.loads(row["activation_examples"])
        new_examples = []
        for ex in old_examples:
            ci_values = ex["ci_values"]
            component_acts = ex["component_acts"]
            new_examples.append(
                {
                    "token_ids": ex["token_ids"],
                    "firings": [v > activation_threshold for v in ci_values],
                    "activations": {
                        "causal_importance": ci_values,
                        "component_activation": component_acts,
                    },
                }
            )

        mean_ci = row["mean_ci"]
        component_key = row["component_key"]
        assert component_key in firing_density_map, f"{component_key} missing from token_stats.pt"
        firing_density = firing_density_map[component_key]
        mean_activations = {"causal_importance": mean_ci}

        new_conn.execute(
            "INSERT OR REPLACE INTO components VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                component_key,
                row["layer"],
                row["component_idx"],
                firing_density,
                orjson.dumps(mean_activations).decode(),
                orjson.dumps(new_examples).decode(),
                row["input_token_pmi"],
                row["output_token_pmi"],
            ),
        )
        n += 1

    new_conn.commit()

    # Migrate intruder scores from JSONL into harvest DB scores table
    old_intruder_dir = old_db_path.parent.parent / "eval" / "intruder"
    if old_intruder_dir.exists():
        # Use the largest file (most complete run)
        jsonl_files = sorted(old_intruder_dir.glob("*.jsonl"), key=lambda f: f.stat().st_size)
        if jsonl_files:
            intruder_file = jsonl_files[-1]
            n_scores = 0
            with open(intruder_file, "rb") as f:
                for line in f:
                    record = orjson.loads(line)
                    new_conn.execute(
                        "INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?)",
                        (
                            record["component_key"],
                            "intruder",
                            record["score"],
                            orjson.dumps(record.get("trials", [])).decode(),
                        ),
                    )
                    n_scores += 1
            new_conn.commit()
            print(f"  Migrated {n_scores} intruder scores from {intruder_file.name}")

    old_conn.close()
    new_conn.close()
    return n


def migrate_autointerp_db(old_db_path: Path, new_db_path: Path) -> int:
    """Copy autointerp DB (schema is compatible, just copy and strip intruder scores)."""
    shutil.copy2(old_db_path, new_db_path)

    # Remove intruder scores — those belong in harvest.db in the new layout
    conn = sqlite3.connect(str(new_db_path))
    conn.execute("DELETE FROM scores WHERE score_type = 'intruder'")
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM interpretations").fetchone()[0]
    conn.close()
    return n


def migrate_run(run_id: str, timestamp: str = "20260218_000000") -> None:
    """Migrate a single run's harvest + autointerp data to the new layout."""
    harvest_root = SPD_OUT_DIR / "harvest" / run_id
    autointerp_root = SPD_OUT_DIR / "autointerp" / run_id

    print(f"Migrating {run_id}...")

    # --- Harvest ---
    old_harvest_db = harvest_root / "activation_contexts" / "harvest.db"
    assert old_harvest_db.exists(), f"No legacy harvest DB at {old_harvest_db}"

    new_subrun = harvest_root / f"h-{timestamp}"
    assert not new_subrun.exists(), f"Target already exists: {new_subrun}"
    new_subrun.mkdir(parents=True)

    token_stats_path = harvest_root / "correlations" / "token_stats.pt"
    assert token_stats_path.exists(), f"No token_stats.pt at {token_stats_path}"
    print(f"  Harvest DB: {old_harvest_db} -> {new_subrun / 'harvest.db'}")
    n_components = migrate_harvest_db(old_harvest_db, new_subrun / "harvest.db", token_stats_path)
    print(f"  Migrated {n_components} components")

    # Copy .pt files
    old_corr_dir = harvest_root / "correlations"
    for pt_file in ["component_correlations.pt", "token_stats.pt"]:
        src = old_corr_dir / pt_file
        if src.exists():
            dst = new_subrun / pt_file
            shutil.copy2(src, dst)
            print(f"  Copied {pt_file}")

    # --- Autointerp ---
    old_interp_db = autointerp_root / "interp.db"
    if old_interp_db.exists():
        new_autointerp_subrun = autointerp_root / f"a-{timestamp}"
        assert not new_autointerp_subrun.exists(), f"Target already exists: {new_autointerp_subrun}"
        new_autointerp_subrun.mkdir(parents=True)

        print(f"  Interp DB: {old_interp_db} -> {new_autointerp_subrun / 'interp.db'}")
        n_interps = migrate_autointerp_db(old_interp_db, new_autointerp_subrun / "interp.db")
        print(f"  Migrated {n_interps} interpretations (detection + fuzzing scores preserved)")
    else:
        print("  No top-level interp.db found, skipping autointerp")

    print("Done!")


def _find_legacy_run_ids() -> list[str]:
    harvest_root = SPD_OUT_DIR / "harvest"
    return sorted(
        d.name
        for d in harvest_root.iterdir()
        if (d / "activation_contexts" / "harvest.db").exists()
    )


def diagnose(run_id: str | None = None, timestamp: str = "20260218_000000") -> None:
    """Print diagnostic report without migrating anything."""
    run_ids = [run_id] if run_id else _find_legacy_run_ids()
    diagnostics = [diagnose_run(rid, timestamp) for rid in run_ids]
    print_diagnostics(diagnostics)


def migrate_all(timestamp: str = "20260218_000000", dry_run: bool = False) -> None:
    """Discover and migrate all legacy harvest runs."""
    run_ids = _find_legacy_run_ids()
    diagnostics = [diagnose_run(rid, timestamp) for rid in run_ids]

    print_diagnostics(diagnostics)

    ready = [d for d in diagnostics if d.ready]
    if dry_run or not ready:
        return

    for d in ready:
        migrate_run(d.run_id, timestamp=timestamp)
    print(f"\nAll done — migrated {len(ready)} runs")


if __name__ == "__main__":
    fire.Fire({"run": migrate_run, "all": migrate_all, "diagnose": diagnose})
