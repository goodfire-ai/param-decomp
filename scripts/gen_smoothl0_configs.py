"""Generate the 12 smooth-L0-vs-Lp investigation configs from the baseline config.yaml.

3 methods x 4 coeffs, identical to baseline p-0ff8e5d3 except the single importance-
minimality loss metric (and run_name). The CI_L0 eval metric gets explicit
`l0_thresholds: [0.0, 0.01]` so the persisted config records both cutoffs.
"""

import copy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "scratchpad" / "config.yaml"  # downloaded from wandb run p-0ff8e5d3
OUT_DIR = REPO / "param_decomp" / "configs" / "smoothl0_investigation"

COEFFS = {"5e-5": 5e-5, "1e-4": 1e-4, "2e-4": 2e-4, "5e-4": 5e-4}


def lp_metric(coeff: float, final_p: float) -> dict:
    return {
        "type": "ImportanceMinimalityLoss",
        "coeff": coeff,
        "pnorm": 2.0,
        "beta": 0.5,
        "p_anneal_start_frac": 0.0,
        "p_anneal_final_p": final_p,
        "p_anneal_end_frac": 1.0,
        "eps": 1e-6,
    }


def smoothl0_metric(coeff: float) -> dict:
    return {
        "type": "SmoothL0ImportanceMinimalityLoss",
        "coeff": coeff,
        "gamma": 1.0,
        "beta": 0.5,
        "gamma_anneal_start_frac": 0.0,
        "gamma_anneal_final_gamma": 0.1,
        "gamma_anneal_end_frac": 1.0,
    }


METHODS = {
    "lp2to04": lambda c: lp_metric(c, 0.4),
    "lp2to1": lambda c: lp_metric(c, 1.0),
    "smoothl0": smoothl0_metric,
}


def main() -> None:
    base = yaml.safe_load(BASELINE.read_text())
    assert base["pd"]["loss_metrics"][0]["type"] == "ImportanceMinimalityLoss", (
        "expected imp-min as first loss metric"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for method, make_metric in METHODS.items():
        for coeff_tag, coeff in COEFFS.items():
            cfg = copy.deepcopy(base)
            cfg["pd"]["loss_metrics"][0] = make_metric(coeff)
            run_name = f"sl0inv-{method}-c{coeff_tag}"
            cfg["run_name"] = run_name
            for metric in cfg["eval"]["metrics"]:
                if metric.get("type") == "CI_L0":
                    metric["l0_thresholds"] = [0.0, 0.01]
            # Disable the slow/plot eval tier: it crashes on multi-host GPU (cuDNN-fp32
            # attention — see PR #885 — and an np.asarray-on-sharded-array bug at
            # slow_eval.py:170). The comparison only needs the fast-eval scalars (L0@0/0.01,
            # CE/KL) + the host-side L0 bar charts, all on the fast path.
            cfg["eval"]["slow_on_first_step"] = False
            cfg["eval"]["slow_every"] = cfg["pd"]["steps"] + 10000
            cfg.pop("run_id", None)  # minted fresh by pd-lm
            path = OUT_DIR / f"{run_name}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
            print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
