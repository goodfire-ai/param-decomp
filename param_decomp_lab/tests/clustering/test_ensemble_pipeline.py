"""End-to-end test of the restored ensemble pipeline post-harvest: a seeded ensemble of
merges over a shared synthetic membership snapshot, then the consensus driver
(normalization -> per-iteration distances -> stability plot)."""

from pathlib import Path

import numpy as np
import pytest

from param_decomp_lab.clustering.harvest_config import HarvestConfig
from param_decomp_lab.clustering.memberships import MembershipBuilder
from param_decomp_lab.clustering.merge_config import MergeConfig
from param_decomp_lab.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from param_decomp_lab.clustering.scripts import calc_distances, run_merge
from param_decomp_lab.clustering.types import ComponentLabels


def _write_synthetic_snapshot(path: Path, *, n_samples: int, n_components: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    acts = rng.random((n_samples, n_components)).astype(np.float32)
    builder = MembershipBuilder(
        activation_threshold=0.5,
        filter_dead_threshold=0.0,
        filter_dead_stat="max",
        filter_modules=lambda _: True,
    )
    builder.add_batch({"site": acts})
    builder.finalize().save(path)


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    snap = tmp_path / "harvest"
    snap.mkdir()
    _write_synthetic_snapshot(snap, n_samples=200, n_components=12, seed=0)
    return snap


def _patch_clustering_paths(monkeypatch: pytest.MonkeyPatch, base: Path) -> None:
    """Redirect clustering run/ensemble dirs into a temp base across the modules that
    resolve them, so the test never touches PARAM_DECOMP_OUT_DIR."""

    def run_dir(run_id: str) -> Path:
        d = base / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ensemble_dir(ensemble_id: str) -> Path:
        d = base / "ensembles" / ensemble_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr(run_merge, "clustering_run_dir", run_dir)
    monkeypatch.setattr(calc_distances, "clustering_run_dir", run_dir)
    monkeypatch.setattr(calc_distances, "clustering_ensemble_dir", ensemble_dir)


def test_ensemble_merge_then_consensus(
    snapshot_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "clustering"
    _patch_clustering_paths(monkeypatch, base)

    merge_config = MergeConfig(
        alpha=1.0,
        iters=8,
        merge_pair_sampling_method="range",
        merge_pair_sampling_kwargs={"threshold": 0.1},
    )

    run_ids = [f"c-test{i}" for i in range(3)]
    for i, run_id in enumerate(run_ids):
        history_path = run_merge.merge(
            snapshot_path=snapshot_dir,
            merge_config=merge_config,
            run_id=run_id,
            seed=i,
            plot_dir=base / "runs" / run_id / "plots",
        )
        assert history_path.exists()
        assert (base / "runs" / run_id / "plots" / "cluster_sizes.png").exists()

    ensemble_id = "e-test"
    calc_distances.calc_distances(
        ensemble_id=ensemble_id,
        clustering_run_ids=run_ids,
        distances_method="perm_invariant_hamming",
    )

    ens_dir = base / "ensembles" / ensemble_id
    assert (ens_dir / "ensemble_meta.json").exists()
    assert (ens_dir / "ensemble_merge_array.npz").exists()
    assert (ens_dir / "distances_perm_invariant_hamming.npz").exists()
    assert (ens_dir / "plots" / "distances_perm_invariant_hamming.png").exists()

    distances = np.load(ens_dir / "distances_perm_invariant_hamming.npz")["distances"]
    n_ens = len(run_ids)
    assert distances.shape == (8, n_ens, n_ens)
    # perm_invariant_hamming fills only the strict lower triangle; diag/upper are NaN.
    lower = distances[:, np.tril_indices(n_ens, k=-1)[0], np.tril_indices(n_ens, k=-1)[1]]
    assert np.all(np.isfinite(lower)) and np.all(lower >= 0.0)


def test_seed_determinism(
    snapshot_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same seed -> identical merge trajectory; different seed -> may differ."""
    base = tmp_path / "clustering"
    _patch_clustering_paths(monkeypatch, base)
    merge_config = MergeConfig(
        alpha=1.0,
        iters=8,
        merge_pair_sampling_method="range",
        merge_pair_sampling_kwargs={"threshold": 0.3},
    )

    def merges_for(run_id: str, seed: int) -> np.ndarray:
        path = run_merge.merge(
            snapshot_path=snapshot_dir,
            merge_config=merge_config,
            run_id=run_id,
            seed=seed,
            plot_dir=None,
        )
        return MergeHistory.read(path).merges.group_idxs.copy()

    a = merges_for("c-a", seed=0)
    a_repeat = merges_for("c-a2", seed=0)
    np.testing.assert_array_equal(a, a_repeat)


def test_pipeline_local_fans_out_three_tiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`submit(local=True)` builds the seeded harvest -> merge -> consensus command tiers
    with one harvest + one merge per member and one consensus job per distance method."""
    from param_decomp_lab.clustering.scripts import run_pipeline

    base = tmp_path / "clustering"
    monkeypatch.setattr(
        run_pipeline,
        "clustering_ensemble_dir",
        lambda eid: (base / "ensembles" / eid),
    )
    monkeypatch.setattr(
        run_pipeline,
        "clustering_harvest_dir",
        lambda hid: (base / "harvests" / hid),
    )

    tiers: list[list[str]] = []
    monkeypatch.setattr(run_pipeline, "run_locally", lambda commands: tiers.append(commands))

    config = run_pipeline.ClusteringEnsembleConfig(
        harvest=HarvestConfig(
            model_path=str(tmp_path / "fake_run"),
            batch_size=4,
            n_tokens=100,
            n_tokens_per_seq=4,
            activation_threshold=0.1,
        ),
        merge=MergeConfig(iters=5),
        n_runs=3,
        distances_methods=["perm_invariant_hamming", "matching_dist"],
    )
    run_pipeline.submit(config, local=True)

    harvest_cmds, merge_cmds, consensus_cmds = tiers
    assert len(harvest_cmds) == 3
    assert len(merge_cmds) == 3
    assert len(consensus_cmds) == 2
    assert all("run_worker" in c and "--dataset_seed" in c for c in harvest_cmds)
    assert all("run_merge" in c and "--seed" in c for c in merge_cmds)
    assert {"--distances-method" in c for c in consensus_cmds} == {True}


def test_normalized_handles_differing_dead_components() -> None:
    """Ensemble normalization unions component labels across members with disjoint
    alive sets (each member sees a different subset)."""
    config = MergeConfig(iters=3, alpha=1.0)
    h0 = MergeHistory.from_config(config, ComponentLabels([f"site:{j}" for j in range(4)]))
    h1 = MergeHistory.from_config(config, ComponentLabels([f"site:{j}" for j in range(2, 6)]))
    merges, meta = MergeHistoryEnsemble(data=[h0, h1]).normalized()
    assert meta["c_components"] == 6
    assert merges.shape[0] == 2
