import yaml, copy
from pathlib import Path

base_p = Path("param_decomp/configs/smoothl0_investigation/sl0inv-smoothl0-c2e-4.yaml")
base = yaml.safe_load(base_p.read_text())
outdir = base_p.parent

PEN = {"frac": "FractionImportanceMinimalityLoss",
       "mcp": "MCPImportanceMinimalityLoss",
       "arctan": "ArctanImportanceMinimalityLoss"}
COEFF = {"c1e-4": 1.0e-4, "c2e-4": 2.0e-4}
STEPS = {"100k": 100000, "400k": 400000}

lm = base["pd"]["loss_metrics"]
idx = next(i for i, m in enumerate(lm) if "ImportanceMinimality" in m["type"])
print("base imp-min block:", {k: lm[idx][k] for k in lm[idx]})
print("base dp:", base["runtime"]["dp"], "base steps:", base["pd"]["steps"])

n = 0
for sh, typ in PEN.items():
    for cs, cv in COEFF.items():
        for st, sv in STEPS.items():
            cfg = copy.deepcopy(base)
            m = cfg["pd"]["loss_metrics"][idx]
            m["type"] = typ
            m["coeff"] = cv
            cfg["pd"]["steps"] = sv
            cfg["runtime"]["dp"] = 8
            name = f"rdsc-{sh}-{cs}-{st}"
            cfg["run_name"] = name
            cfg.setdefault("wandb", {})
            cfg["wandb"]["group"] = "redescending-impmin"
            cfg["wandb"]["tags"] = ["rdsc", sh, st, "btdr"]
            (outdir / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            n += 1
            print("wrote", name + ".yaml")
print("generated", n)
