"""CLI entry point for unified postprocessing pipeline.

Thin wrapper for fast --help. Heavy imports deferred to postprocess.py.

Usage:
    spd-postprocess config.yaml
    spd-postprocess config.yaml --dependency 311644_1
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit all postprocessing jobs for an SPD run.")
    parser.add_argument("config", help="Path to PostprocessConfig YAML.")
    parser.add_argument(
        "--dependency",
        help="SLURM job ID to wait for before starting (e.g. a training job).",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    import yaml

    from spd.log import logger
    from spd.postprocess import postprocess
    from spd.postprocess.config import PostprocessConfig

    cfg = PostprocessConfig.from_file(args.config)

    if args.dry_run:
        logger.info("Dry run: skipping submission\n\nConfig:\n")
        logger.info(yaml.dump(cfg.model_dump(), indent=2, sort_keys=False))
        return

    manifest_path = postprocess(config=cfg, dependency_job_id=args.dependency)
    logger.info(f"Manifest: {manifest_path}")


def cli() -> None:
    main()
