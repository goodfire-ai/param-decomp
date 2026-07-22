"""Migrate experiment yamls from the pre-carve decomposition schema to `decomposition:`.

Old shape (removed from `PDConfig` by the decomposition-config carve):

    pd:
      ci_config: {type: chunkwise_transformer, ...}     # or layerwise_mlp / global_mlp
      decomposition_targets:
      - {module_pattern: model.layers.18.mlp.gate_proj, C: 49152}

New shape (per-domain `decomposition` section, sibling of `pd`):

    decomposition:
      sites: {kind: glu_transformer, layers: {kind: list, indices: [18]}, cs: {gate: 49152}}
      ci: {type: chunkwise_transformer, ...}

`migrate_raw` is the pure dict->dict transform (reusable on stored `launch_config.yaml`s);
the CLI rewrites files IN PLACE with line surgery so comments and untouched sections
survive, then proves the surgery equals `migrate_raw` and that the migrated file parses
under the new schema with an identical resolved site list.

Usage: python -m param_decomp_lab.migrations.decomposition_apparatus <yaml files...>
"""

import copy
import re
import sys
from pathlib import Path
from typing import Any, Literal

import yaml

# The two LM site grammars (mirrors llama8b / llama_simple_mlp SITE_NAME_PATTERN, plus the
# optional raw-HF `model.` prefix and the `*` layer wildcard the old specs allowed).
_GLU_PATTERN = re.compile(
    r"^(?:model\.)?layers\.(\d+|\*)\.(?:self_attn\.(q|k|v|o)|mlp\.(gate|up|down))_proj$"
)
_SMLP_PATTERN = re.compile(
    r"^h\.(\d+|\*)\.(?:attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(c_fc|down_proj))$"
)
_GLU_KIND_ORDER = ("q", "k", "v", "o", "gate", "up", "down")
_SMLP_KIND_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")
_LLAMA8B_N_LAYER = 32  # the GLU family's one target (Llama-3.1-8B)

Domain = Literal["lm", "toy"]


def _parse_lm_targets(
    targets: list[dict[str, Any]],
) -> tuple[str, tuple[str, ...], dict[str, dict[str, int]]]:
    """-> (family kind, family KIND_ORDER, {layer_key: {matrix: C}}); layer_key '*' = wildcard."""
    family: str | None = None
    per_layer: dict[str, dict[str, int]] = {}
    for t in targets:
        pattern, c = t["module_pattern"], t["C"]
        if m := _GLU_PATTERN.match(pattern):
            fam = "glu_transformer"
        elif m := _SMLP_PATTERN.match(pattern):
            fam = "simple_mlp"
        else:
            raise ValueError(f"unsupported decomposition target {pattern!r}")
        assert family is None or family == fam, f"mixed site families: {family} vs {fam}"
        family = fam
        layer = m.group(1)
        matrix = next(g for g in m.groups()[1:] if g is not None)
        assert matrix not in per_layer.get(layer, {}), f"duplicate target {pattern!r}"
        per_layer.setdefault(layer, {})[matrix] = c
    assert family is not None, "empty decomposition_targets"
    order = _GLU_KIND_ORDER if family == "glu_transformer" else _SMLP_KIND_ORDER
    return family, order, per_layer


def _layer_selection(layers: list[int], family: str) -> dict[str, Any]:
    if family == "glu_transformer" and layers == list(range(_LLAMA8B_N_LAYER)):
        return {"kind": "all"}
    if len(layers) > 1 and layers == list(range(layers[0], layers[-1] + 1)):
        return {"kind": "range", "start": layers[0], "end": layers[-1] + 1}
    return {"kind": "list", "indices": layers}


def _lm_sites(targets: list[dict[str, Any]]) -> dict[str, Any]:
    """Old per-module-pattern list -> tiled `{kind, layers, cs}`. Refuses specs the tiled
    schema cannot represent (per-layer heterogeneous cs) rather than silently changing them."""
    family, order, per_layer = _parse_lm_targets(targets)
    per_layer_cs = list(per_layer.values())
    assert all(cs == per_layer_cs[0] for cs in per_layer_cs), (
        f"per-layer heterogeneous cs not representable in the tiled schema: {per_layer}"
    )
    cs = {matrix: per_layer_cs[0][matrix] for matrix in order if matrix in per_layer_cs[0]}
    if "*" in per_layer:
        assert set(per_layer) == {"*"}, f"mixed wildcard/explicit layers: {sorted(per_layer)}"
        selection: dict[str, Any] = {"kind": "all"}
    else:
        selection = _layer_selection(sorted(int(k) for k in per_layer), family)
    return {"kind": family, "layers": selection, "cs": cs}


def _toy_sites(targets: list[dict[str, Any]]) -> dict[str, Any]:
    sites = [{"name": t["module_pattern"], "C": t["C"]} for t in targets]
    return {"kind": "explicit", "sites": sites}


def migrate_raw(raw: dict[str, Any], domain: Domain) -> dict[str, Any]:
    """Pure transform: old-shape experiment-config dict -> new-shape. Asserts the input IS
    old-shape (both `pd.ci_config` and `pd.decomposition_targets` present)."""
    new = copy.deepcopy(raw)
    pd = new["pd"]
    ci = pd.pop("ci_config")
    targets = pd.pop("decomposition_targets")
    assert "type" in ci, f"pre-chunkwise ci_config schema (keys {sorted(ci)}); migrate by hand"
    sites = _lm_sites(targets) if domain == "lm" else _toy_sites(targets)
    new["decomposition"] = {"sites": sites, "ci": ci}
    return new


# ---------------------------------------------------------------------------
# File surgery (comment-preserving)
# ---------------------------------------------------------------------------


def _block_span(lines: list[str], start: int) -> int:
    """End index (exclusive) of the yaml block whose key line is `lines[start]`: subsequent
    lines that are blank, deeper-indented, or list items at the key's own indent."""
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines):
        line = lines[end]
        stripped = line.strip()
        deeper = (len(line) - len(line.lstrip())) > indent
        list_item = line[indent:].startswith("- ") if len(line) > indent else False
        if stripped and not deeper and not list_item:
            break
        end += 1
    return end


def _remove_pd_key(lines: list[str], key: str) -> list[str]:
    matches = [i for i, line in enumerate(lines) if line.rstrip("\n") == f"  {key}:"]
    assert len(matches) == 1, f"expected exactly one `  {key}:` line, found {len(matches)}"
    start = matches[0]
    return lines[:start] + lines[_block_span(lines, start) :]


def migrate_file(path: Path, domain: Domain) -> None:
    old_text = path.read_text()
    old_raw = yaml.safe_load(old_text)
    expected = migrate_raw(old_raw, domain)

    lines = old_text.splitlines(keepends=True)
    lines = _remove_pd_key(lines, "ci_config")
    lines = _remove_pd_key(lines, "decomposition_targets")
    section = "decomposition:\n" + "".join(
        "  " + line
        for line in yaml.safe_dump(
            expected["decomposition"], sort_keys=False, default_flow_style=False
        ).splitlines(keepends=True)
    )
    (pd_line,) = [i for i, line in enumerate(lines) if line.rstrip("\n") == "pd:"]
    lines = lines[:pd_line] + [section] + lines[pd_line:]

    new_text = "".join(lines)
    assert yaml.safe_load(new_text) == expected, f"surgery != migrate_raw for {path}"
    path.write_text(new_text)


def _domain_of(path: Path) -> Domain:
    return "toy" if ("/tms/" in str(path) or "/resid_mlp/" in str(path)) else "lm"


def _validate_lm(path: Path) -> str:
    """Parse under the new schema; for HF/vendored llama8b specs also prove the resolved
    site list is byte-identical to the old canonical expansion of the pre-migration spec
    (pretrained specs need the pretrain cache to resolve, so they get parse-only)."""
    from param_decomp.components import SiteC
    from param_decomp.targets import glu_transformer, llama8b
    from param_decomp.targets.glu_transformer import canonical_site_cs
    from param_decomp_lab.experiments.lm.config import LMExperimentConfig, resolve_site_tree

    cfg = LMExperimentConfig.model_validate(yaml.safe_load(path.read_text()))
    if cfg.decomposition.sites.kind != "glu_transformer":
        return "parse-only (pretrained target; resolution needs the pretrain cache)"
    tree = resolve_site_tree(
        cfg.decomposition.sites, glu_transformer.FAMILY, llama8b.llama31_8b_config().n_layer
    )
    new_sites = tree.site_cs(glu_transformer.FAMILY.name_of)
    old_targets = _OLD_RAW[path]["pd"]["decomposition_targets"]
    old_sites = canonical_site_cs(
        tuple(SiteC(t["module_pattern"].removeprefix("model."), t["C"]) for t in old_targets)
    )
    assert new_sites == old_sites, f"resolved sites drifted for {path}"
    return f"sites verified identical ({len(new_sites)} sites)"


def _validate_toy(path: Path) -> str:
    from param_decomp_lab.experiments.resid_mlp.config import ResidMLPExperimentConfig
    from param_decomp_lab.experiments.tms.config import TMSExperimentConfig

    cls = TMSExperimentConfig if "/tms/" in str(path) else ResidMLPExperimentConfig
    cfg = cls.model_validate(yaml.safe_load(path.read_text()))
    old = [(t["module_pattern"], t["C"]) for t in _OLD_RAW[path]["pd"]["decomposition_targets"]]
    new = [(s.name, s.C) for s in cfg.decomposition.sites.sites]
    assert new == old, f"explicit sites drifted for {path}"
    return f"sites verified identical ({len(new)} sites)"


_OLD_RAW: dict[Path, dict[str, Any]] = {}


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv]
    assert paths, __doc__
    failures = 0
    for path in paths:
        domain = _domain_of(path)
        _OLD_RAW[path] = yaml.safe_load(path.read_text())
        try:
            migrate_file(path, domain)
            verdict = _validate_lm(path) if domain == "lm" else _validate_toy(path)
            print(f"OK   {path}: {verdict}")
        except Exception as e:  # noqa: BLE001 - report per-file, keep going
            failures += 1
            print(f"FAIL {path}: {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
