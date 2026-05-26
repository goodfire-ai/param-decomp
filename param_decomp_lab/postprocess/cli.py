"""CLI entry point for unified postprocessing pipeline.

Thin wrapper for fast --help. Heavy imports deferred to postprocess.py.

Uses argparse instead of Fire because SLURM job IDs like "311644_1" get
parsed by Fire as integers (underscore is a numeric separator in Python).

Usage:
    pd-postprocess config.yaml
    pd-postprocess config.yaml --dependency 311644_1
"""

import argparse


def main() -> None:
    """Parse CLI args and submit (or `--dry_run`) the postprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Submit all postprocessing jobs for a PD run.")
    parser.add_argument("config", help="Path to PostprocessConfig YAML.")
    parser.add_argument(
        "--dependency",
        help="SLURM job ID to wait for before starting (e.g. a training job).",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    import yaml

    from param_decomp.log import logger
    from param_decomp_lab.postprocess import postprocess
    from param_decomp_lab.postprocess.config import PostprocessConfig

    cfg = PostprocessConfig.from_file(args.config)

    if args.dry_run:
        logger.info("Dry run: skipping submission\n\nConfig:\n")
        logger.info(yaml.dump(cfg.model_dump(), indent=2, sort_keys=False))
        return

    metadata_path = postprocess(config=cfg, dependency_job_id=args.dependency)
    logger.info(f"Metadata: {metadata_path}")


def cli() -> None:
    main()
