import fire

from param_decomp.settings import DEFAULT_PARTITION_NAME, DEFAULT_PROJECT_NAME


def main(
    experiments: str | tuple[str, ...] | None = None,
    sweep: str | bool = False,
    n_agents: int | None = None,
    create_report: bool = False,
    report_title: str | None = None,
    job_suffix: str | None = None,
    cpu: bool = False,
    partition: str = DEFAULT_PARTITION_NAME,
    dp: int | None = None,
    project: str = DEFAULT_PROJECT_NAME,
) -> None:
    """Launch PD experiments on a SLURM cluster, with optional parameter grid expansion.

    Run ``pd-launch`` to see the discovered list of built-in experiments. Examples:

        pd-launch --experiments tms_5-2                            # one job
        pd-launch --experiments tms_5-2,resid_mlp1                 # two jobs
        pd-launch                                                  # all discovered experiments
        pd-launch --experiments tms_5-2 --project my-proj          # custom W&B project
        pd-launch --experiments tms_5-2 --cpu                      # CPU job
        pd-launch --experiments ss_llama_simple_mlp-2L --dp 4      # 4-GPU single-node DDP
        pd-launch --experiments ss_llama_simple_mlp-2L --dp 16     # 16-GPU multi-node DDP

    Grid expansion (``--sweep``) reads a YAML of parameter grids (defaults to
    ``param_decomp/scripts/sweep_params.yaml``) and submits one SLURM array task per
    combination. This is a local Cartesian product, not a W&B sweep agent; W&B sees
    independent runs tagged with a shared launch_id.

        pd-launch --experiments tms_5-2 --sweep --n_agents 4
        pd-launch --experiments tms_5-2 --sweep my_sweep.yaml --n_agents 4
    """
    from param_decomp.scripts.run_slurm import launch_slurm_run

    launch_slurm_run(
        experiments=experiments,
        sweep=sweep,
        n_agents=n_agents,
        create_report=create_report,
        report_title=report_title,
        job_suffix=job_suffix,
        cpu=cpu,
        partition=partition,
        dp=dp,
        project=project,
    )


def cli():
    fire.Fire(main)
