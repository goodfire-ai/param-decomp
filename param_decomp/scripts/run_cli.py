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
    """Run PD experiments on SLURM cluster with optional sweeps.

    Available experiments:
    - tms_5-2
    - tms_5-2-id
    - tms_40-10
    - tms_40-10-id
    - resid_mlp1
    - resid_mlp2
    - resid_mlp3
    - ss_llama_simple
    - ss_gpt2
    - ss_gpt2_simple
    - ss_gpt2_simple_noln
    - gpt2
    - ts

    Examples:

    # Run subset of experiments (no sweep)
    pd-run --experiments tms_5-2,resid_mlp1

    # Run parameter sweep on a subset of experiments with default sweep_params.yaml
    pd-run --experiments tms_5-2,resid_mlp2 --sweep

    # Run parameter sweep on an experiment with custom sweep params at param_decomp/scripts/my_sweep.yaml
    pd-run --experiments tms_5-2 --sweep my_sweep.yaml

    # Run all experiments (no sweep)
    pd-run

    # Use custom W&B project
    pd-run --experiments tms_5-2 --project my-pd-project

    # Run all experiments on CPU
    pd-run --experiments tms_5-2 --cpu

    # Run with data parallelism over 4 GPUs (single node)
    pd-run --experiments ss_llama_simple --dp 4

    # Run with multi-node training over 16 GPUs (2 nodes x 8 GPUs each)
    pd-run --experiments ss_llama_simple --dp 16

    """
    from param_decomp.scripts.run import launch_slurm_run

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
