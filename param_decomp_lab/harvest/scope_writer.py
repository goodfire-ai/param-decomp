"""Write a finished `Harvester` directly to scope artifact shards (one per site).

This is the native path that replaces the old harvest.db JSON-blob store plus the
`scope.convert` transcode: harvest builds its NumPy accumulators + reservoir once and
writes the fast mmap/indexed scope format straight out, storing the FULL reservoir pool
(no top-k trim) so the viewer can page examples ranked by CI.

One shard per site holds ALL `layer_c` components — dead components (no firings) get an
empty example set and zero scalars, keeping shard idx == component idx so the viewer's
stable per-site indexing holds. The harvest-consumer facade (`HarvestRepo`) filters to
fired components to preserve the old only-fired semantics.
"""

from datetime import UTC, datetime

import numpy as np

from param_decomp_lab.harvest.accumulator import Harvester
from param_decomp_lab.harvest.reservoir import WINDOW_PAD_SENTINEL, ActivationExamplesReservoir
from param_decomp_lab.harvest.sampling import top_k_pmi
from param_decomp_lab.scope.artifacts import (
    FORMAT_VERSION,
    ComponentExamples,
    SiteMeta,
    SiteShardWriter,
)

CI_ACT_TYPE = "causal_importance"
COMPONENT_ACT_TYPE = "component_activation"


def _component_examples(
    reservoir: ActivationExamplesReservoir, flat_idx: int
) -> tuple[ComponentExamples, float]:
    """Left-pack one component's reservoir slots into fixed-width scope arrays.

    Reservoir windows carry `WINDOW_PAD_SENTINEL` at the edges when a firing sits near a
    sequence boundary; filtering those out yields the contiguous real tokens the scope
    format stores left-packed to `lengths[j]`. Returns `(examples, max_ci)` where `max_ci`
    is the peak causal-importance over all real positions (0.0 for a dead component)."""
    n = int(reservoir.n_items[flat_idx])
    w = reservoir.window
    toks = reservoir.tokens[flat_idx, :n]
    firs = reservoir.firings[flat_idx, :n]
    ci = reservoir.acts[CI_ACT_TYPE][flat_idx, :n]
    act = reservoir.acts[COMPONENT_ACT_TYPE][flat_idx, :n]

    out_tok = np.zeros((n, w), dtype=np.uint32)
    out_fir = np.zeros((n, w), dtype=np.uint8)
    out_ci = np.zeros((n, w), dtype=np.float16)
    out_act = np.zeros((n, w), dtype=np.float16)
    lengths = np.zeros(n, dtype=np.uint16)
    max_ci = 0.0
    for j in range(n):
        mask = toks[j] != WINDOW_PAD_SENTINEL
        length = int(mask.sum())
        lengths[j] = length
        out_tok[j, :length] = toks[j, mask]
        out_fir[j, :length] = firs[j, mask]
        out_ci[j, :length] = ci[j, mask]
        out_act[j, :length] = act[j, mask]
        if length:
            max_ci = max(max_ci, float(ci[j, mask].max()))

    examples = ComponentExamples(
        token_ids=out_tok, firings=out_fir, ci=out_ci, act=out_act, lengths=lengths
    )
    return examples, max_ci


def write_scope_shards(
    harvester: Harvester,
    run_id: str,
    subrun_id: str,
    tokenizer_name: str,
    pmi_top_k: int,
) -> None:
    """Publish one scope shard per site from a finished (merged) `Harvester`."""
    total = harvester.total_tokens_processed
    assert total > 0, "harvester saw no tokens"
    assert CI_ACT_TYPE in harvester.activation_sums, harvester.activation_sums.keys()
    assert COMPONENT_ACT_TYPE in harvester.activation_sums, harvester.activation_sums.keys()

    mean_ci_all = harvester.activation_sums[CI_ACT_TYPE] / total
    mean_act_all = harvester.activation_sums[COMPONENT_ACT_TYPE] / total
    input_marginals = harvester.input_marginals.astype(np.float64)
    output_marginals = harvester.output_marginals

    for site, layer_c in harvester.layers:
        offset = harvester.layer_offsets[site]
        meta = SiteMeta(
            format_version=FORMAT_VERSION,
            run_id=run_id,
            site=site,
            subrun_id=subrun_id,
            n_components=layer_c,
            k_examples=harvester.reservoir.k,
            window=harvester.reservoir.window,
            tokenizer_name=tokenizer_name,
            n_tokens_seen=total,
            pmi_top_k=pmi_top_k,
            provenance=f"harvested natively from {subrun_id}",
            created_at=datetime.now(UTC).isoformat(),
        )
        writer = SiteShardWriter(meta)
        for i in range(layer_c):
            flat = offset + i
            firing_count = int(harvester.firing_counts[flat])
            examples, max_ci = _component_examples(harvester.reservoir, flat)
            input_top, _ = top_k_pmi(
                harvester.input_cooccurrence[flat].astype(np.float64),
                input_marginals,
                float(firing_count),
                total,
                pmi_top_k,
            )
            output_top, _ = top_k_pmi(
                harvester.output_cooccurrence[flat],
                output_marginals,
                float(firing_count),
                total,
                pmi_top_k,
            )
            writer.write_component(
                idx=i,
                examples=examples,
                firing_count=firing_count,
                firing_density=firing_count / total,
                max_act=max_ci,
                mean_ci=float(mean_ci_all[flat]),
                mean_act=float(mean_act_all[flat]),
                input_pmi=input_top,
                output_pmi=output_top,
            )
        writer.publish()
