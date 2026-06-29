import yaml, copy
from pathlib import Path

D = Path("param_decomp/configs/smoothl0_investigation")
base = yaml.safe_load((D / "rdsc-frac-c2e-4-400k.yaml").read_text())  # frac 2e-4 g0.1 400k dp8

TYPE = {"frac": "FractionImportanceMinimalityLoss",
        "sl0": "SmoothL0ImportanceMinimalityLoss",
        "mcp": "MCPImportanceMinimalityLoss"}
CV = {"1e-4": 1.0e-4, "2e-4": 2.0e-4, "5e-4": 5.0e-4}
GV = {"0.1": 0.1, "0.2": 0.2, "0.05": 0.05, "0.01": 0.01}

specs = [("frac", c, g) for c in ["1e-4", "2e-4", "5e-4"] for g in ["0.2", "0.05", "0.01"]]
specs += [("frac", "5e-4", "0.1")]
specs += [("sl0", "2e-4", g) for g in ["0.1", "0.2", "0.05", "0.01"]]
specs += [("mcp", "2e-4", g) for g in ["0.2", "0.05", "0.01"]]

idx = next(i for i, m in enumerate(base["pd"]["loss_metrics"]) if "ImportanceMinimality" in m["type"])
n = 0
for pen, cs, gs in specs:
    cfg = copy.deepcopy(base)
    m = cfg["pd"]["loss_metrics"][idx]
    m["type"] = TYPE[pen]; m["coeff"] = CV[cs]
    m["gamma"] = 1.0
    m["gamma_anneal_start_frac"] = 0.0
    m["gamma_anneal_final_gamma"] = GV[gs]
    m["gamma_anneal_end_frac"] = 1.0
    cfg["pd"]["steps"] = 400000
    cfg["runtime"]["dp"] = 8
    name = f"rdsc-{pen}-c{cs}-g{gs}-400k"
    cfg["run_name"] = name
    cfg.setdefault("wandb", {})["group"] = "redescending-gamma-sweep"
    cfg["wandb"]["tags"] = ["rdsc", "gamma-sweep", pen, f"g{gs}"]
    (D / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    n += 1
    print(f"{name}  type={m['type']}  coeff={m['coeff']}  final_gamma={m['gamma_anneal_final_gamma']}")
print("generated", n)
