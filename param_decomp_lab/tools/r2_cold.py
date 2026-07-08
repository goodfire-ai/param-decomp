"""pd-cold: R2 cold storage for finished decomposition runs.

Push a finished run's DECOMPOSITION checkpoint item (SPEC S22 split layout — never the
resume-only `training` item, ~90% of the bytes) plus its pinned `launch_config.yaml` and a
provenance manifest to the no-expiry R2 archive bucket, and pull it back onto any cluster
as a consumer-ready run dir. This restores the torch-era "every finished run points at a
durable artifact" contract that the JAX migration dropped, without a human SSHing between
clusters.

R2 layout (bucket: the cluster profile's `R2_ARCHIVE_BUCKET`):

    <user>/pd-cold/<run_id>/
        manifest.json                       # uploaded LAST: its presence == upload complete
        launch_config.yaml
        ckpts/<step>/_CHECKPOINT_METADATA
        ckpts/<step>/decomposition/...      # the orbax decomposition item

`<user>` is the pushing kernel user (`id -un`), matching `bin/r2`'s namespacing. Credentials
/ endpoint / tuned multipart config come from the cluster scripts (`$DATA_MOUNT/scripts/
{config.sh,r2.sh}`, creds group-readable via `goodfire`); we drive the `r2_*` bash functions
directly because `bin/r2 push --name` forbids the nested prefix this layout needs.

    python -m param_decomp_lab.tools.r2_cold push <run_dir> [--allow-incomplete]
    python -m param_decomp_lab.tools.r2_cold pull <run_id> [--user USER] [--dest DIR]

Push is strict by default: the latest checkpoint step must equal the config's `pd.steps`
(a SIGTERM/requeue save is not a finished run). `--allow-incomplete` exists for archiving
abandoned-but-kept runs (e.g. `strip_checkpoint`-migrated pre-split runs). Both directions
are `aws s3 sync` under the hood: idempotent, resume-on-rerun.

The end-of-training auto-push edge is `maybe_push_completed_run` (called by the LM
composition root on rank 0 after the engine returns): push iff the final-step checkpoint
exists, then stamp `wandb.config["r2_cold"]` so the wandb run points at its durable
artifact. Upload failure there is LOUD but non-fatal — a multi-day run must not read as
failed because R2 hiccuped; the printed retry command re-runs the same idempotent sync.
"""

import datetime
import getpass
import json
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import fire

from param_decomp_lab.infra.settings import DATA_MOUNT, PARAM_DECOMP_OUT_DIR

MANIFEST_FORMAT = "pd-cold-v1"
DECOMPOSITION_ITEM = "decomposition"
TRAINING_ITEM = "training"
CHECKPOINT_METADATA = "_CHECKPOINT_METADATA"
LAUNCH_CONFIG = "launch_config.yaml"
MANIFEST = "manifest.json"


# ---------------------------------------------------------------------------
# R2 shell plumbing
# ---------------------------------------------------------------------------


def _scripts_dir() -> Path:
    assert DATA_MOUNT is not None, "pd-cold needs a cluster ($DATA_MOUNT unset)"
    scripts = DATA_MOUNT / "scripts"
    assert (scripts / "r2.sh").is_file(), f"no r2 infra at {scripts}"
    return scripts


def _r2_bash(script: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    """Run `script` in a bash that has the cluster's `r2_*` functions loaded and
    `R2_BUCKET` pointed at the archive bucket. Raises on any failure (creds unreadable,
    R2 disabled on this cluster, aws error) with the shell's own diagnostic."""
    scripts = _scripts_dir()
    prelude = f"""\
set -euo pipefail
source "{scripts}/config.sh"
[[ "${{ENABLE_R2_TRANSFER:-false}}" == true ]] || {{ echo "R2 transfers not enabled on $CLUSTER_NAME" >&2; exit 3; }}
source "{scripts}/r2.sh"
export R2_BUCKET="${{R2_ARCHIVE_BUCKET:?archive bucket not configured on this cluster}}"
"""
    return subprocess.run(
        ["bash", "-c", prelude + script],
        check=True,
        text=True,
        capture_output=capture,
    )


def _cluster_name() -> str:
    scripts = _scripts_dir()
    out = subprocess.run(
        ["bash", "-c", f'source "{scripts}/config.sh" && echo "${{CLUSTER_NAME:-}}"'],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return out.strip("[]") or socket.gethostname()


# ---------------------------------------------------------------------------
# Run-dir inspection
# ---------------------------------------------------------------------------


def _config_steps(run_dir: Path) -> int:
    import yaml

    launch_config = run_dir / LAUNCH_CONFIG
    assert launch_config.is_file(), f"no {LAUNCH_CONFIG} under {run_dir} (not a JAX run dir?)"
    steps = yaml.safe_load(launch_config.read_text())["pd"]["steps"]
    assert isinstance(steps, int) and steps > 0, steps
    return steps


def latest_decomposition_step(run_dir: Path) -> int | None:
    """Latest checkpoint step saved in the split (S22) layout, i.e. with a
    `decomposition` item dir. Pre-split (`default`-item) steps don't count — they need
    `tools/strip_checkpoint.py` first."""
    ckpts = run_dir / "ckpts"
    if not ckpts.is_dir():
        return None
    steps = [
        int(d.name)
        for d in ckpts.iterdir()
        if d.name.isdigit() and (d / DECOMPOSITION_ITEM).is_dir()
    ]
    return max(steps, default=None)


def _tree_stats(paths: list[Path]) -> tuple[int, int]:
    """(file count, total bytes) over files and directory trees."""
    files: list[Path] = []
    for p in paths:
        assert p.exists(), p
        files.extend(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else files.append(p)
    return len(files), sum(f.stat().st_size for f in files)


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------


def push(run_dir: str, allow_incomplete: bool = False) -> str:
    """Archive `run_dir`'s decomposition artifact to R2. Returns the R2 prefix.

    Uploads `ckpts/<latest step>/decomposition` + `_CHECKPOINT_METADATA` +
    `launch_config.yaml`, then `manifest.json` last as the completion marker. Idempotent:
    re-running syncs only what's missing/changed.
    """
    run = Path(run_dir).resolve()
    run_id = run.name
    steps = _config_steps(run)
    step = latest_decomposition_step(run)
    assert step is not None, (
        f"{run}: no split-format checkpoint (pre-split run? strip it first: "
        f"python -m param_decomp.tools.strip_checkpoint {run} --apply)"
    )
    complete = step == steps
    assert complete or allow_incomplete, (
        f"{run_id}: latest checkpoint {step} != pd.steps {steps} — not a finished run "
        "(pass --allow-incomplete to archive it anyway)"
    )

    step_dir = run / "ckpts" / str(step)
    decomposition = step_dir / DECOMPOSITION_ITEM
    ckpt_metadata = step_dir / CHECKPOINT_METADATA
    assert ckpt_metadata.is_file(), ckpt_metadata
    launch_config = run / LAUNCH_CONFIG

    user = getpass.getuser()
    prefix = f"{user}/pd-cold/{run_id}"
    n_files, n_bytes = _tree_stats([decomposition, ckpt_metadata, launch_config])
    print(
        f"pd-cold push: {run_id} step {step}/{steps} -> {prefix}/ "
        f"({n_files} files, {n_bytes / 2**30:.2f} GiB)",
        flush=True,
    )

    manifest = {
        "format": MANIFEST_FORMAT,
        "run_id": run_id,
        "step": step,
        "pd_steps": steps,
        "complete": complete,
        "files": n_files,
        "bytes": n_bytes,
        "prefix": prefix,
        "source_cluster": _cluster_name(),
        "source_run_dir": str(run),
        "uploaded_by": user,
        "uploaded_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    _r2_bash(
        f"""\
r2_sync_up "{decomposition}" "{prefix}/ckpts/{step}/{DECOMPOSITION_ITEM}/" --exact-timestamps
r2_cp "{ckpt_metadata}" "{prefix}/ckpts/{step}/{CHECKPOINT_METADATA}"
r2_cp "{launch_config}" "{prefix}/{LAUNCH_CONFIG}"
"""
    )

    remote_files, remote_bytes = _remote_stats(prefix)
    assert remote_files == n_files and remote_bytes == n_bytes, (
        f"post-upload verification failed: remote {remote_files} files / {remote_bytes} B "
        f"!= local {n_files} files / {n_bytes} B (re-run to resume the sync)"
    )

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(manifest, f, indent=2)
        manifest_tmp = f.name
    _r2_bash(f'r2_cp "{manifest_tmp}" "{prefix}/{MANIFEST}"')
    Path(manifest_tmp).unlink()

    print(
        f"pd-cold push complete: {prefix}/\n"
        f"  pull anywhere with: python -m param_decomp_lab.tools.r2_cold pull {run_id}",
        flush=True,
    )
    return prefix


def _remote_stats(prefix: str) -> tuple[int, int]:
    """(object count, total bytes) under `prefix`/ (excluding manifest.json)."""
    out = _r2_bash(f'r2_ls "{prefix}/" --recursive --summarize', capture=True).stdout
    objects = re.search(r"Total Objects: (\d+)", out)
    size = re.search(r"Total Size: (\d+)", out)
    assert objects and size, f"unparseable s3 ls output:\n{out}"
    n, b = int(objects.group(1)), int(size.group(1))
    manifest_line = re.search(rf"(\d+) \S*{re.escape(MANIFEST)}$", out, re.MULTILINE)
    if manifest_line:  # a previous push's marker: not part of the payload comparison
        n -= 1
        b -= int(manifest_line.group(1))
    return n, b


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------


def pull(run_id: str, user: str | None = None, dest: str | None = None) -> Path:
    """Fetch an archived run onto this cluster as a consumer-ready run dir
    (`$PARAM_DECOMP_OUT_DIR/runs/<run_id>/` by default): `launch_config.yaml` +
    `ckpts/<step>/decomposition` (+ the provenance `manifest.json`), the shape
    `restore_decomposition*` consumers expect. `--user` skips the owner search."""
    owner = user if user is not None else _find_owner(run_id)
    prefix = f"{owner}/pd-cold/{run_id}"
    manifest_ls = _r2_bash(f'r2_ls "{prefix}/{MANIFEST}"', capture=True).stdout
    assert MANIFEST in manifest_ls, (
        f"{prefix}/{MANIFEST} not found — incomplete or missing upload "
        f"(check: r2 ls {prefix}/ --bucket archive)"
    )

    dest_dir = Path(dest).resolve() if dest is not None else PARAM_DECOMP_OUT_DIR / "runs" / run_id
    training_items = list(dest_dir.glob(f"ckpts/*/{TRAINING_ITEM}"))
    assert not training_items, (
        f"{dest_dir} holds a resumable training run ({training_items[0]}) — refusing to "
        "overlay the archive onto it"
    )

    print(f"pd-cold pull: {prefix}/ -> {dest_dir}", flush=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    _r2_bash(f'r2_sync_down "{prefix}/" "{dest_dir}" --exact-timestamps')
    manifest = json.loads((dest_dir / MANIFEST).read_text())
    assert manifest["run_id"] == run_id, manifest
    print(
        f"pd-cold pull complete: {dest_dir} "
        f"(step {manifest['step']}, {manifest['bytes'] / 2**30:.2f} GiB, "
        f"from {manifest['source_cluster']})",
        flush=True,
    )
    return dest_dir


def _find_owner(run_id: str) -> str:
    """Find which `<user>/` namespace holds `pd-cold/<run_id>` by probing each top-level
    user prefix in the archive bucket."""
    top = _r2_bash("r2_ls", capture=True).stdout
    users = re.findall(r"PRE (\S+)/", top)
    owners = [
        u
        for u in users
        if MANIFEST in _r2_bash(f'r2_ls "{u}/pd-cold/{run_id}/{MANIFEST}"', capture=True).stdout
    ]
    assert owners, f"pd-cold/{run_id} not found under any user prefix (searched: {users})"
    assert len(owners) == 1, f"pd-cold/{run_id} found under multiple users: {owners} — pass --user"
    return owners[0]


# ---------------------------------------------------------------------------
# end-of-training edge hook
# ---------------------------------------------------------------------------


def maybe_push_completed_run(run_dir: Path, pd_steps: int) -> None:
    """Rank-0, after the engine returns: archive the run iff it actually finished (the
    final-step checkpoint exists — a SIGTERM/requeue save never satisfies this), then
    stamp `wandb.config["r2_cold"]` on the still-open wandb run. Failure is loud but
    non-fatal; the retry command re-runs the same idempotent sync."""
    step = latest_decomposition_step(run_dir)
    if step != pd_steps:
        print(
            f"pd-cold: latest checkpoint {step} != pd.steps {pd_steps}, not archiving", flush=True
        )
        return
    try:
        prefix = push(str(run_dir))
    except Exception as e:  # noqa: BLE001 — the run itself succeeded; archive later
        print(
            "="
            * 72
            + f"\nWARNING: pd-cold archive upload FAILED (training itself succeeded):\n  {e}\n"
            f"  retry manually with:\n"
            f"    python -m param_decomp_lab.tools.r2_cold push {run_dir}\n" + "=" * 72,
            file=sys.stderr,
            flush=True,
        )
        return
    _stamp_wandb(prefix, pd_steps)


def _stamp_wandb(prefix: str, step: int) -> None:
    """Record the durable artifact's location on the wandb run, if one is open in this
    process. Uses the already-imported module — never imports wandb itself, so the tool
    stays usable standalone/offline."""
    wandb = sys.modules.get("wandb")
    if wandb is None or wandb.run is None:
        return
    wandb.run.config.update(
        {"r2_cold": {"bucket": "archive", "prefix": prefix, "step": step}},
        allow_val_change=True,
    )


if __name__ == "__main__":
    fire.Fire({"push": push, "pull": pull})
