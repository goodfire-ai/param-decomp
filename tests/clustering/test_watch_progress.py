from pathlib import Path

from spd.clustering.scripts.watch_progress import (
    LogSummary,
    SchedulerInfo,
    JobStatus,
    _render_section,
    expand_job_tokens,
    summarize_log,
)


def test_expand_job_tokens_supports_ranges_commas_and_array_ids() -> None:
    assert expand_job_tokens(["376819-376821", "376830,376831", "376788_7"]) == [
        "376819",
        "376820",
        "376821",
        "376830",
        "376831",
        "376788_7",
    ]


def test_summarize_log_parses_merge_progress_and_eta(tmp_path: Path) -> None:
    log_path = tmp_path / "slurm-123.out"
    log_path.write_text(
        "\n".join(
            [
                "Built component activity CSR in 393.66s (shape=(30000000, 10007), nnz=5744682298)",
                "Building coactivation matrix from compressed memberships (n_groups=10007, n_samples=30000000)",
                "Compressed merge progress: iter=200/20000, elapsed=1093.44s, sec_per_iter=5.4672, k_groups=9807",
            ]
        )
    )

    summary = summarize_log(log_path)

    assert summary.stage == "merge"
    assert "200/20000 iters" in summary.detail
    assert "5.467s/it" in summary.detail
    assert "groups=9807" in summary.detail
    assert "eta=" in summary.detail


def test_summarize_log_parses_harvest_save_stage(tmp_path: Path) -> None:
    log_path = tmp_path / "slurm-456.out"
    log_path.write_text(
        "\n".join(
            [
                "Collected 30000000 token activations (requested 30000000)",
                "Saving snapshot: 10007 alive components, 30000000 samples",
            ]
        )
    )

    summary = summarize_log(log_path)

    assert summary.stage == "save"
    assert summary.detail == "30000000 samples, 10007 comps"


def test_render_section_includes_progress_bar_and_percentage() -> None:
    section = _render_section(
        "Running Merges",
        [
            JobStatus(
                scheduler=SchedulerInfo(
                    job_id="376819",
                    state="RUNNING",
                    reason="-",
                    elapsed="1:52:14",
                    start_time="-",
                    name="jose_merge_10m",
                ),
                log=LogSummary(
                    stage="merge",
                    detail="8990/20000 iters, 0.570s/it, groups=1017, eta=1h44m",
                    progress=0.53,
                ),
            )
        ],
        width=140,
    )

    assert "Running Merges" in section
    assert "376819" in section
    assert "[" in section and "]" in section
    assert "%" in section
