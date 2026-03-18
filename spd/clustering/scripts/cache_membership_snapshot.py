"""Collect compressed memberships once and save them for repeated merge benchmarking."""

import argparse
from pathlib import Path

from spd.clustering.activations import collect_memberships_lm, collect_memberships_resid_mlp
from spd.clustering.clustering_run_config import ClusteringRunConfig
from spd.clustering.dataset import create_clustering_dataloader
from spd.clustering.membership_snapshot import save_membership_snapshot
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.spd_types import TaskName
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import replace_pydantic_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-tokens", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    run_config = ClusteringRunConfig.from_file(args.config)
    overrides: dict[str, int] = {}
    if args.n_tokens is not None:
        overrides["n_tokens"] = args.n_tokens
    if args.n_samples is not None:
        overrides["n_samples"] = args.n_samples
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if overrides:
        run_config = replace_pydantic_model(run_config, overrides)

    assert run_config.merge_config.activation_threshold is not None, (
        "Snapshotting only supports thresholded compressed memberships"
    )

    spd_run = SPDRunInfo.from_path(run_config.model_path)
    task_name: TaskName = spd_run.config.task_config.task_name
    device = get_device()
    model = ComponentModel.from_run_info(spd_run).to(device)
    dataloader = create_clustering_dataloader(
        model_path=run_config.model_path,
        task_name=task_name,
        batch_size=run_config.batch_size,
        seed=run_config.dataset_seed,
    )

    if task_name == "lm":
        assert run_config.n_tokens is not None
        assert run_config.n_tokens_per_seq is not None
        processed = collect_memberships_lm(
            model=model,
            dataloader=dataloader,
            n_tokens=run_config.n_tokens,
            n_tokens_per_seq=run_config.n_tokens_per_seq,
            device=device,
            seed=run_config.dataset_seed,
            activation_threshold=run_config.merge_config.activation_threshold,
            filter_dead_threshold=run_config.merge_config.filter_dead_threshold,
            filter_modules=run_config.merge_config.filter_modules,
        )
    else:
        n_samples = run_config.n_samples or run_config.batch_size
        processed = collect_memberships_resid_mlp(
            model=model,
            dataloader=dataloader,
            n_samples=n_samples,
            device=device,
            activation_threshold=run_config.merge_config.activation_threshold,
            filter_dead_threshold=run_config.merge_config.filter_dead_threshold,
            filter_modules=run_config.merge_config.filter_modules,
        )

    save_membership_snapshot(
        args.output_dir,
        memberships=processed.memberships,
        labels=processed.labels,
        n_samples=processed.n_samples,
    )
    print(
        {
            "output_dir": str(args.output_dir),
            "n_samples": processed.n_samples,
            "n_components": len(processed.labels),
        }
    )


if __name__ == "__main__":
    main()
