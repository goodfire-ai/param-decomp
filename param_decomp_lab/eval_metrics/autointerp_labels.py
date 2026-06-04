"""Slow eval metric: autointerp labels for a fixed random subset of components.

Picks `k` components uniformly at random over the concatenated component space (the
same `k` every checkpoint, seeded by `cfg.seed`), runs a subset-restricted in-memory
harvest over the eval pass (reusing `harvest.accumulator.Harvester`), and labels each
via the autointerp LLM stack. Designed to run in the async slow-eval sidecar
(`experiments/lm/async_eval.py`), off the training critical path — the LLM calls block
only rank 0, after the cross-rank reduction has completed.

`set_run_context` must be called (by the composition root that has the run config) to
supply the `ModelMetadata` + tokenizer name the autointerp prompt needs; these aren't
derivable from the bare `ComponentModel`.
"""

import asyncio
import json
import random
from typing import Annotated, Literal, override

import torch
import wandb
from pydantic import Field
from torch import Tensor

from param_decomp.base_config import BaseConfig
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import all_reduce, is_main_process
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.app.backend.app_tokenizer import AppTokenizer
from param_decomp_lab.autointerp.config import StrategyConfig
from param_decomp_lab.autointerp.providers import LLMConfig, create_provider
from param_decomp_lab.autointerp.schemas import ModelMetadata
from param_decomp_lab.autointerp.strategies.dispatch import INTERPRETATION_SCHEMA, format_prompt
from param_decomp_lab.component_model_io import get_all_component_acts
from param_decomp_lab.harvest.accumulator import Harvester
from param_decomp_lab.harvest.analysis import get_input_token_stats, get_output_token_stats
from param_decomp_lab.harvest.schemas import ComponentData
from param_decomp_lab.harvest.storage import TokenStatsStorage
from param_decomp_lab.topology import TransformerTopology

_MAX_EXAMPLES_PER_BATCH_PER_COMPONENT = 8
_INPUT_TOKEN_TOP_K = 20
_OUTPUT_TOKEN_TOP_K = 50


class AutointerpLabelsConfig(BaseConfig):
    type: Literal["AutointerpLabels"] = "AutointerpLabels"
    k: int
    """Number of components to sample uniformly over the concatenated component space."""
    seed: int
    activation_threshold: float
    max_examples: int
    """Reservoir capacity (activation examples kept per sampled component)."""
    context_tokens_per_side: int
    llm: LLMConfig
    template_strategy: Annotated[StrategyConfig, Field(discriminator="type")]
    # Run/data facts the prompt needs that a bare ComponentModel doesn't carry. They
    # mirror `data.*` / are the eval data the metric renders — kept here so the metric
    # is self-contained (plain config dispatch). `n_blocks` / `layer_descriptions` are
    # derived from the model at `bind`.
    dataset_name: str
    seq_len: int
    tokenizer_name: str


class AutointerpLabels(Metric[AutointerpLabelsConfig]):
    """Autointerp labels for a fixed random component subset, logged as a wandb table."""

    log_namespace = "autointerp"
    slow = True
    short_name = "Autointerp"

    def __init__(self, cfg: AutointerpLabelsConfig) -> None:
        super().__init__(cfg)
        self._model_metadata: ModelMetadata | None = None
        self.reset()

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        super().bind(model=model, device=device)
        topology = TransformerTopology(model.target_model)
        self._model_metadata = ModelMetadata(
            n_blocks=topology.n_blocks,
            dataset_name=self.cfg.dataset_name,
            layer_descriptions={
                path: topology.target_to_canon(path) for path in model.target_module_paths
            },
            seq_len=self.cfg.seq_len,
            decomposition_method="pd",
        )

    @override
    def reset(self) -> None:
        self._harvester: Harvester | None = None
        self._selection: dict[str, list[int]] | None = None
        self._u_norms: dict[str, Tensor] | None = None

    def _selected_components(self, ci_lower_leaky: dict[str, Tensor]) -> dict[str, list[int]]:
        """Fixed random component subset, drawn uniformly over the concatenated component
        space. Sampled once (seeded by `cfg.seed`) on first call and cached, so the same
        components are tracked across every eval pass / checkpoint of a run.

        Returns per-site local indices.
        """
        if self._selection is not None:
            return self._selection

        sites = sorted(ci_lower_leaky)
        c_per_site = {s: ci_lower_leaky[s].shape[-1] for s in sites}
        total = sum(c_per_site.values())
        assert self.cfg.k <= total, f"k={self.cfg.k} exceeds total components {total}"

        flat = sorted(random.Random(self.cfg.seed).sample(range(total), self.cfg.k))

        bounds: list[tuple[str, int, int]] = []
        offset = 0
        for s in sites:
            bounds.append((s, offset, offset + c_per_site[s]))
            offset += c_per_site[s]

        selection: dict[str, list[int]] = {}
        for idx in flat:
            site, lo = next((s, lo) for s, lo, hi in bounds if lo <= idx < hi)
            selection.setdefault(site, []).append(idx - lo)

        self._selection = {s: sorted(local) for s, local in selection.items()}
        return self._selection

    @override
    def update(self, ctx: MetricContext) -> None:
        ci = ctx.ci.lower_leaky
        assert isinstance(ctx.batch, Tensor), "AutointerpLabels expects tokenized Tensor batches"
        selection = self._selected_components(ci)

        if self._harvester is None:
            self._u_norms = {site: self.model.components[site].U.norm(dim=1) for site in selection}
            layers = [(site, len(selection[site])) for site in sorted(selection)]
            self._harvester = Harvester(
                layers=layers,
                vocab_size=ctx.target_out.shape[-1],
                max_examples_per_component=self.cfg.max_examples,
                context_tokens_per_side=self.cfg.context_tokens_per_side,
                max_examples_per_batch_per_component=_MAX_EXAMPLES_PER_BATCH_PER_COMPONENT,
                device=torch.device(self.device),
            )
        assert self._u_norms is not None

        comp_acts = get_all_component_acts(self.model, ctx.pre_weight_acts)
        output_probs = torch.softmax(ctx.target_out, dim=-1)

        firings: dict[str, Tensor] = {}
        activations: dict[str, dict[str, Tensor]] = {}
        for site in sorted(selection):
            idx = torch.tensor(selection[site], device=self.device)
            ci_sel = ci[site][..., idx]
            firings[site] = ci_sel > self.cfg.activation_threshold
            activations[site] = {
                "causal_importance": ci_sel,
                "component_activation": comp_acts[site][..., idx] * self._u_norms[site][idx],
            }

        self._harvester.process_batch(ctx.batch, firings, activations, output_probs)
        return None

    @override
    def compute(self) -> MetricResult:
        assert self._harvester is not None and self._selection is not None
        h = self._harvester
        _all_reduce_harvester_counts(h, device=self.device)

        if not is_main_process():
            return {}

        assert self._model_metadata is not None, "compute() called before bind()"

        storage = TokenStatsStorage(
            component_keys=h.component_keys,
            vocab_size=h.vocab_size,
            n_tokens=h.total_tokens_processed,
            input_counts=h.input_cooccurrence.float().cpu(),
            input_totals=h.input_marginals.float().cpu(),
            output_counts=h.output_cooccurrence.cpu(),
            output_totals=h.output_marginals.cpu(),
            firing_counts=h.firing_counts.cpu(),
        )
        components = list(h.build_results(pmi_top_k_tokens=_INPUT_TOKEN_TOP_K))

        rows = asyncio.run(self._interpret_all(components, storage))

        table = wandb.Table(columns=["component", "label", "reasoning"])
        for display_key, label, reasoning in rows:
            table.add_data(display_key, label, reasoning)
        return {"labels": table}

    async def _interpret_all(
        self, components: list[ComponentData], storage: TokenStatsStorage
    ) -> list[tuple[str, str, str]]:
        assert self._model_metadata is not None
        assert self._selection is not None
        model_metadata = self._model_metadata
        selection = self._selection
        app_tok = AppTokenizer.from_pretrained(self.cfg.tokenizer_name)
        provider = create_provider(self.cfg.llm)

        async def one(component: ComponentData) -> tuple[str, str, str]:
            input_stats = get_input_token_stats(
                storage, component.component_key, app_tok, top_k=_INPUT_TOKEN_TOP_K
            )
            output_stats = get_output_token_stats(
                storage, component.component_key, app_tok, top_k=_OUTPUT_TOKEN_TOP_K
            )
            prompt = format_prompt(
                strategy=self.cfg.template_strategy,
                component=component,
                model_metadata=model_metadata,
                app_tok=app_tok,
                input_token_stats=input_stats,
                output_token_stats=output_stats,
                context_tokens_per_side=self.cfg.context_tokens_per_side,
                activation_threshold=self.cfg.activation_threshold,
            )
            response = await provider.chat(
                prompt=prompt,
                max_tokens=8000,
                response_schema=INTERPRETATION_SCHEMA,
                timeout_ms=120_000,
            )
            parsed = json.loads(response.content)
            true_idx = selection[component.layer][component.component_idx]
            return f"{component.layer}:{true_idx}", parsed["label"], parsed["reasoning"]

        try:
            return await asyncio.gather(*(one(c) for c in components))
        finally:
            await provider.close()


def _all_reduce_harvester_counts(harvester: Harvester, *, device: str) -> None:
    """Sum the harvester's count accumulators across ranks, in place.

    Only the fields that feed token stats and `build_results` are reduced;
    `cooccurrence_counts` is unused here and left as-is.
    """
    h = harvester
    h.firing_counts = all_reduce(h.firing_counts)
    for act_type in list(h.activation_sums):
        h.activation_sums[act_type] = all_reduce(h.activation_sums[act_type])
    h.input_cooccurrence = all_reduce(h.input_cooccurrence)
    h.input_marginals = all_reduce(h.input_marginals)
    h.output_cooccurrence = all_reduce(h.output_cooccurrence)
    h.output_marginals = all_reduce(h.output_marginals)
    h.total_tokens_processed = int(
        all_reduce(torch.tensor(h.total_tokens_processed, device=device)).item()
    )
