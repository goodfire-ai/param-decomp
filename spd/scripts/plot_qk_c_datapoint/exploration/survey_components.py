"""Survey component CI distributions for SPD run s-55ea3f9b."""

import sys
from collections import defaultdict
from pathlib import Path

from spd.harvest.repo import HarvestRepo

OUTPUT_PATH = Path(__file__).parent / "component_survey.txt"
DECOMPOSITION_ID = "s-55ea3f9b"


def main() -> None:
    repo = HarvestRepo.open_most_recent(DECOMPOSITION_ID)
    assert repo is not None, f"No harvest data found for {DECOMPOSITION_ID}"

    summary = repo.get_summary()
    print(f"Total components: {len(summary)}")

    # Filter to q_proj and k_proj layers
    qk_components: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for key, comp in summary.items():
        if "q_proj" in comp.layer or "k_proj" in comp.layer:
            qk_components[comp.layer].append((key, comp))

    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log(f"Decomposition: {DECOMPOSITION_ID}")
    log(f"Total components in harvest: {len(summary)}")
    log(f"Q/K layers found: {sorted(qk_components.keys())}")
    log()

    # Collect moderate component keys for bulk fetch
    moderate_keys: list[str] = []

    for layer_name in sorted(qk_components.keys()):
        comps = qk_components[layer_name]
        comps.sort(key=lambda x: x[1].component_idx)

        always_on = []
        moderate = []
        low = []
        dead = []

        for key, comp in comps:
            mean_ci = comp.mean_activations.get("causal_importance", 0.0)
            if mean_ci > 0.8:
                always_on.append((key, comp, mean_ci))
            elif mean_ci > 0.05:
                moderate.append((key, comp, mean_ci))
            elif mean_ci > 0.001:
                low.append((key, comp, mean_ci))
            else:
                dead.append((key, comp, mean_ci))

        log(f"=== {layer_name} ({len(comps)} components) ===")
        log(f"  always on (CI > 0.8):    {len(always_on)}")
        log(f"  moderate  (0.05-0.8):    {len(moderate)}")
        log(f"  low       (0.001-0.05):  {len(low)}")
        log(f"  dead      (CI < 0.001):  {len(dead)}")
        log()

        if always_on:
            log(f"  Always-on components:")
            for key, comp, ci in sorted(always_on, key=lambda x: -x[2]):
                log(f"    {key}: mean_CI={ci:.4f}, firing_density={comp.firing_density:.4f}")
            log()

        if moderate:
            log(f"  Moderate components (interesting):")
            for key, comp, ci in sorted(moderate, key=lambda x: -x[2]):
                mean_act = comp.mean_activations.get("component_activation", None)
                act_str = f", mean_act={mean_act:.4f}" if mean_act is not None else ""
                log(
                    f"    {key}: mean_CI={ci:.4f}, firing_density={comp.firing_density:.4f}{act_str}"
                )
                moderate_keys.append(key)
            log()

        if low:
            log(f"  Low components:")
            for key, comp, ci in sorted(low, key=lambda x: -x[2])[:10]:
                log(f"    {key}: mean_CI={ci:.4f}, firing_density={comp.firing_density:.4f}")
            if len(low) > 10:
                log(f"    ... and {len(low) - 10} more")
            log()

    # Fetch full data for moderate components to show activation context info
    if moderate_keys:
        log("=" * 60)
        log("DETAILED INFO FOR MODERATE COMPONENTS")
        log("=" * 60)
        bulk_data = repo.get_components_bulk(moderate_keys)
        for key in moderate_keys:
            comp_data = bulk_data.get(key)
            if comp_data is None:
                continue
            log(f"\n--- {key} (layer={comp_data.layer}, idx={comp_data.component_idx}) ---")
            log(f"  firing_density: {comp_data.firing_density:.4f}")
            log(f"  mean_activations: {comp_data.mean_activations}")
            log(f"  n_activation_examples: {len(comp_data.activation_examples)}")
            if comp_data.activation_examples:
                ex = comp_data.activation_examples[0]
                n_firing = sum(ex.firings)
                log(f"  first example: {len(ex.token_ids)} tokens, {n_firing} firing")
            log(f"  input_pmi top-5: {comp_data.input_token_pmi.top[:5]}")
            log(f"  output_pmi top-5: {comp_data.output_token_pmi.top[:5]}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines))
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
