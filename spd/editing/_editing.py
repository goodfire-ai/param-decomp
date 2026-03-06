"""Component-level model editing for VPD decompositions.

Core class: EditableModel wraps ComponentModel + TransformerTopology and provides
methods for component analysis, editing, and measurement. It's callable
(tokens → logits) so it works as a ForwardFn anywhere.

Usage:
    from spd.editing import EditableModel, search_interpretations, generate

    em = EditableModel.from_wandb("wandb:goodfire/spd/s-892f140b")
    matches = search_interpretations(harvest, interp, r"male pronoun")

    edit_fn = em.make_edit_fn({m.key: 0.0 for m in matches[:3]})
    text = generate(edit_fn, tokens, tokenizer)
    effect = em.measure_kl(edit_fn, token_seqs)
"""

import copy
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

import orjson
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.compute import OptimizedPromptAttributionResult
from spd.autointerp.repo import InterpRepo
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ComponentData
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import make_mask_infos
from spd.topology.topology import TransformerTopology

ForwardFn = Callable[[Int[Tensor, " seq"]], Float[Tensor, "seq vocab"]]


# -- Component key utilities ---------------------------------------------------


def parse_component_key(key: str) -> tuple[str, int]:
    """'h.1.mlp.c_fc:802' -> ('h.1.mlp.c_fc', 802)."""
    layer, idx_str = key.rsplit(":", 1)
    return layer, int(idx_str)


# -- Search (free functions, don't need the model) -----------------------------


@dataclass
class ComponentMatch:
    key: str
    label: str
    confidence: str
    firing_density: float
    mean_activations: dict[str, float]


def search_interpretations(
    harvest: HarvestRepo,
    interp: InterpRepo,
    pattern: str,
    min_firing_density: float = 0.0,
) -> list[ComponentMatch]:
    """Search component interpretations by regex on label. Sorted by firing density desc."""
    all_interps = interp.get_all_interpretations()
    summary = harvest.get_summary()

    matches = []
    for key, result in all_interps.items():
        if key not in summary:
            continue
        if not re.search(pattern, result.label, re.IGNORECASE):
            continue
        s = summary[key]
        if s.firing_density < min_firing_density:
            continue
        matches.append(
            ComponentMatch(
                key=key,
                label=result.label,
                confidence=result.confidence,
                firing_density=s.firing_density,
                mean_activations=s.mean_activations,
            )
        )

    matches.sort(key=lambda m: -m.firing_density)
    return matches


@dataclass
class TokenPMIMatch:
    key: str
    pmi: float
    firing_density: float


def search_by_token_pmi(
    harvest: HarvestRepo,
    token_ids: list[int],
    side: str,
    min_pmi: float = 0.5,
    min_firing_density: float = 0.01,
    top_k: int = 20,
) -> list[TokenPMIMatch]:
    """Find components by input or output token PMI.

    side="output" finds components that PREDICT the given tokens.
    side="input" finds components that RESPOND TO (fire on) the given tokens.

    For ablation, you almost always want side="output" — ablating output-side
    components suppresses token production with far less collateral damage than
    ablating input-side components.
    """
    assert side in ("input", "output")
    column = "output_token_pmi" if side == "output" else "input_token_pmi"
    target_set = set(token_ids)
    summary = harvest.get_summary()

    db_path = harvest._dir / "harvest.db"
    conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)

    results = []
    for row in conn.execute(f"SELECT component_key, {column} FROM components"):
        key: str = row[0]
        if key not in summary or summary[key].firing_density < min_firing_density:
            continue
        pmi_data: dict[str, list[list[float]]] = orjson.loads(row[1])
        max_pmi = 0.0
        for tok_id, pmi in pmi_data.get("top", []):
            if int(tok_id) in target_set and pmi > max_pmi:
                max_pmi = pmi
        if max_pmi >= min_pmi:
            results.append(
                TokenPMIMatch(
                    key=key,
                    pmi=max_pmi,
                    firing_density=summary[key].firing_density,
                )
            )

    conn.close()
    results.sort(key=lambda r: -r.pmi)
    return results[:top_k]


def inspect_component(
    harvest: HarvestRepo,
    interp: InterpRepo,
    key: str,
    tokenizer: AppTokenizer,
    n_examples: int = 5,
    n_pmi_tokens: int = 10,
) -> ComponentData:
    """Print a detailed inspection of a component and return its data."""
    comp = harvest.get_component(key)
    assert comp is not None, f"No harvest data for {key}"
    interp_result = interp.get_interpretation(key)

    ci = comp.mean_activations.get("causal_importance", None)
    ci_str = f", ci={ci:.4f}" if ci is not None else ""
    print(f"{'=' * 70}")
    print(f"{key}  (density={comp.firing_density:.4f}{ci_str})")
    if interp_result:
        print(f"Label: [{interp_result.confidence}] {interp_result.label}")
    print()

    decode = tokenizer.decode

    print("INPUT tokens (what makes it fire):")
    for tok_id, pmi in comp.input_token_pmi.top[:n_pmi_tokens]:
        print(f"  {decode([tok_id]):15s} PMI={pmi:.2f}")

    print("\nOUTPUT tokens (what it predicts):")
    for tok_id, pmi in comp.output_token_pmi.top[:n_pmi_tokens]:
        print(f"  {decode([tok_id]):15s} PMI={pmi:.2f}")

    print(f"\nActivation examples ({n_examples}):")
    for ex in comp.activation_examples[:n_examples]:
        parts = []
        for tid, firing in zip(ex.token_ids, ex.firings, strict=True):
            tok_str = decode([tid])
            parts.append(f">>>{tok_str}<<<" if firing else tok_str)
        act_vals = ex.activations.get("causal_importance", ex.activations.get("activation", []))
        max_act = max(act_vals) if act_vals else 0
        print(f"  [max_act={max_act:.3f}] {''.join(parts)}")
    print()

    return comp


# -- Result types --------------------------------------------------------------


@dataclass
class ComponentVectors:
    """Read (V) and write (U) vectors for a single rank-1 component.

    The component forward is: act = x @ read, out = act * write.
    So `read` is the input direction (d_in) and `write` is the output direction (d_out).
    """

    key: str
    read: Tensor
    write: Tensor
    d_in: int
    d_out: int


@dataclass
class AlignmentResult:
    cosine: float
    dot: float
    norm_a: float
    norm_b: float
    percentile: float
    space_dim: int
    space_name: str


@dataclass
class UnembedMatch:
    token_id: int
    token_str: str
    cosine: float
    dot: float


@dataclass
class AblationEffect:
    mean_kl: float
    baseline_ppl: float
    edited_ppl: float
    n_tokens: int

    @property
    def ppl_increase_pct(self) -> float:
        return (self.edited_ppl / self.baseline_ppl - 1) * 100


@dataclass
class TokenGroupShift:
    group_name: str
    baseline_mean_prob: float
    edited_mean_prob: float
    n_positions: int

    @property
    def change_pct(self) -> float:
        if self.baseline_mean_prob == 0:
            return float("inf") if self.edited_mean_prob > 0 else 0.0
        return (self.edited_mean_prob / self.baseline_mean_prob - 1) * 100


# -- EditableModel -------------------------------------------------------------


class EditableModel:
    """ComponentModel + TransformerTopology with methods for editing and analysis.

    Callable: em(tokens) returns logits, so it works as a ForwardFn.
    """

    def __init__(self, model: ComponentModel) -> None:
        self.model = model
        self.topology = TransformerTopology(model.target_model)

    @classmethod
    def from_wandb(
        cls, wandb_path: str, device: str = "cuda"
    ) -> tuple["EditableModel", AppTokenizer]:
        """Load from wandb path. Returns (editable_model, tokenizer)."""
        run_info = SPDRunInfo.from_path(wandb_path)
        model = ComponentModel.from_run_info(run_info).to(device).eval()
        assert run_info.config.tokenizer_name is not None
        tokenizer = AppTokenizer.from_pretrained(run_info.config.tokenizer_name)
        return cls(model), tokenizer

    def __call__(self, tokens: Int[Tensor, " seq"]) -> Float[Tensor, "seq vocab"]:
        return self.model(tokens.unsqueeze(0)).squeeze(0)

    # -- Component geometry ----------------------------------------------------

    def get_component_vectors(self, key: str) -> ComponentVectors:
        """Get the read (V[:, c]) and write (U[c, :]) vectors for a component."""
        layer, idx = parse_component_key(key)
        comp = self.model.components[layer]
        return ComponentVectors(
            key=key,
            read=comp.V[:, idx],
            write=comp.U[idx, :],
            d_in=int(comp.d_in),  # pyright: ignore[reportArgumentType]
            d_out=int(comp.d_out),  # pyright: ignore[reportArgumentType]
        )

    def component_alignment(self, key_a: str, key_b: str) -> AlignmentResult:
        """Cosine/dot between key_a's write direction and key_b's read direction.

        Asserts they share a space (key_a's d_out == key_b's d_in).
        Percentile is empirical over all pairs in the same two layers.
        """
        a = self.get_component_vectors(key_a)
        b = self.get_component_vectors(key_b)
        assert a.d_out == b.d_in, (
            f"{key_a} writes d={a.d_out}, {key_b} reads d={b.d_in} — no shared space"
        )

        cos = F.cosine_similarity(a.write.unsqueeze(0), b.read.unsqueeze(0)).item()
        dot = (a.write * b.read).sum().item()

        layer_a, _ = parse_component_key(key_a)
        layer_b, _ = parse_component_key(key_b)
        all_writes = self.model.components[layer_a].U
        all_reads = self.model.components[layer_b].V
        all_cos = F.normalize(all_writes, dim=1) @ F.normalize(all_reads, dim=0)
        percentile = (all_cos.abs() < abs(cos)).float().mean().item() * 100

        resid_dim = self.topology.unembed_module.in_features
        space_name = "residual" if a.d_out == resid_dim else "neuron"

        return AlignmentResult(
            cosine=cos,
            dot=dot,
            norm_a=a.write.norm().item(),
            norm_b=b.read.norm().item(),
            percentile=percentile,
            space_dim=a.d_out,
            space_name=space_name,
        )

    def unembed_alignment(
        self,
        key: str,
        tokenizer: AppTokenizer,
        top_k: int = 10,
    ) -> tuple[list[UnembedMatch], list[UnembedMatch]]:
        """Top boosted and suppressed tokens by alignment with write direction.

        Only works for components that write to the residual stream.
        Returns (top_boosted, top_suppressed).
        """
        vecs = self.get_component_vectors(key)
        unembed = self.topology.unembed_module.weight  # [vocab, d_model]
        assert vecs.d_out == unembed.shape[1], (
            f"{key} writes d={vecs.d_out}, unembed expects d={unembed.shape[1]}"
        )

        all_cos = F.cosine_similarity(vecs.write.unsqueeze(0), unembed, dim=1)
        all_dot = (vecs.write.unsqueeze(0) * unembed).sum(dim=1)

        decode = tokenizer.decode

        top_vals, top_ids = all_cos.topk(top_k)
        boosted = [
            UnembedMatch(int(t), decode([int(t)]), v.item(), all_dot[t].item())
            for v, t in zip(top_vals, top_ids, strict=True)
        ]

        bot_vals, bot_ids = all_cos.topk(top_k, largest=False)
        suppressed = [
            UnembedMatch(int(t), decode([int(t)]), v.item(), all_dot[t].item())
            for v, t in zip(bot_vals, bot_ids, strict=True)
        ]

        return boosted, suppressed

    def get_component_activations(
        self,
        tokens: Int[Tensor, " seq"],
        key: str,
    ) -> Float[Tensor, " seq"]:
        """Component activation (v_c^T @ x) at each sequence position."""
        layer, idx = parse_component_key(key)
        with torch.no_grad():
            out = self.model(tokens.unsqueeze(0), cache_type="input")
        pre_weight_acts = out.cache[layer]  # [1, seq, d_in]
        comp = self.model.components[layer]
        return (pre_weight_acts @ comp.V[:, idx]).squeeze(0)  # [seq]

    def get_ci(
        self,
        tokens: Int[Tensor, " seq"],
    ) -> dict[str, Float[Tensor, " seq C"]]:
        """Get CI values for all components at all positions. Returns {layer: [seq, C]}."""
        with torch.no_grad():
            out = self.model(tokens.unsqueeze(0), cache_type="input")
            ci = self.model.calc_causal_importances(
                pre_weight_acts=out.cache,
                sampling="continuous",
                detach_inputs=False,
            )
        return {layer: vals.squeeze(0) for layer, vals in ci.lower_leaky.items()}

    def find_components_by_examples(
        self,
        examples: list[tuple[Int[Tensor, " seq"], int]],
        optim_steps: int = 100,
        context_window: int = 10,
        ci_alive_threshold: float = 0.0,
        min_frequency: float = 0.7,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        """Find components needed for a behavior by optimizing sparse CI on examples.

        For each (token_sequence, target_position) pair, runs CI optimization
        to find the minimal set of components needed to predict the token at
        target_position. Components that appear in the sparse set across
        >= min_frequency of examples are returned.

        Args:
            examples: List of (token_sequence, target_position) pairs.
                target_position is the sequence index of the token whose
                prediction we want to explain.
            optim_steps: Number of optimization steps per example.
            ci_alive_threshold: CI threshold for considering a component "active"
                in the optimized mask.
            min_frequency: Fraction of examples where a component must be active.
            top_k: Number of components to return.

        Returns:
            List of (component_key, frequency) sorted by frequency descending.
        """
        from spd.app.backend.optim_cis import (
            CELossConfig,
            OptimCIConfig,
            optimize_ci_values,
        )
        from spd.configs import ImportanceMinimalityLossConfig

        counts: dict[str, int] = {}
        n_examples = len(examples)

        for i, (tokens, target_pos) in enumerate(examples):
            assert target_pos > 0, "target_position must be > 0 (need a previous position)"

            # Truncate to context window ending at target_pos (inclusive)
            start = max(0, target_pos - context_window + 1)
            window = tokens[start : target_pos + 1]
            window_target_pos = target_pos - start
            target_token = window[window_target_pos].item()

            config = OptimCIConfig(
                seed=42,
                lr=0.1,
                steps=optim_steps,
                weight_decay=0.0,
                lr_schedule="cosine",
                lr_exponential_halflife=None,
                lr_warmup_pct=0.1,
                log_freq=optim_steps + 1,  # suppress logging
                imp_min_config=ImportanceMinimalityLossConfig(coeff=0.1, pnorm=0.5, beta=1.0),
                loss_config=CELossConfig(
                    coeff=20.0,
                    position=window_target_pos - 1,
                    label_token=int(target_token),
                ),
                sampling="continuous",
                ce_kl_rounding_threshold=0.5,
                mask_type="ci",
                adv_pgd=None,
            )

            result = optimize_ci_values(
                model=self.model,
                tokens=window.unsqueeze(0),
                config=config,
                device=str(tokens.device),
            )

            # Extract active components from optimized CI
            ci_outputs = result.params.create_ci_outputs(self.model, str(tokens.device))
            for layer_name, ci_vals in ci_outputs.lower_leaky.items():
                # ci_vals: [1, window_len, C]
                pred_pos = window_target_pos - 1
                active = ci_vals[0, pred_pos, :] > ci_alive_threshold
                for c in active.nonzero(as_tuple=True)[0]:
                    key = f"{layer_name}:{c.item()}"
                    counts[key] = counts.get(key, 0) + 1

            print(f"  Example {i + 1}/{n_examples}: L0={result.metrics.l0_total:.0f}")

        min_count = int(min_frequency * n_examples)
        freq_results = [
            (key, count / n_examples) for key, count in counts.items() if count >= min_count
        ]
        freq_results.sort(key=lambda x: -x[1])
        return freq_results[:top_k]

    def optimize_circuit(
        self,
        tokens: Int[Tensor, " seq"],
        target_position: int,
        target_token: int,
        optim_steps: int = 200,
        imp_min_coeff: float = 0.1,
        ce_coeff: float = 20.0,
    ) -> OptimizedPromptAttributionResult:
        """Optimize a sparse circuit for predicting target_token at target_position.

        Returns the full attribution graph (edges between components) from the
        app's compute pipeline. The result includes node CI values, component
        activations, and edge strengths.

        target_position is the sequence index of the token being predicted
        (the logits at position target_position predict this token, so internally
        we optimize for loss at position target_position).
        """
        from spd.app.backend.compute import compute_prompt_attributions_optimized
        from spd.app.backend.optim_cis import CELossConfig, OptimCIConfig
        from spd.configs import ImportanceMinimalityLossConfig
        from spd.topology.gradient_connectivity import get_sources_by_target

        device = str(tokens.device)
        batched = tokens.unsqueeze(0)

        sources_by_target = get_sources_by_target(self.model, self.topology, device, "continuous")

        config = OptimCIConfig(
            seed=42,
            lr=0.1,
            steps=optim_steps,
            weight_decay=0.0,
            lr_schedule="cosine",
            lr_exponential_halflife=None,
            lr_warmup_pct=0.1,
            log_freq=optim_steps + 1,
            imp_min_config=ImportanceMinimalityLossConfig(coeff=imp_min_coeff, pnorm=0.5, beta=1.0),
            loss_config=CELossConfig(
                coeff=ce_coeff,
                position=target_position,
                label_token=target_token,
            ),
            sampling="continuous",
            ce_kl_rounding_threshold=0.5,
            mask_type="ci",
            adv_pgd=None,
        )

        return compute_prompt_attributions_optimized(
            model=self.model,
            topology=self.topology,
            tokens=batched,
            sources_by_target=sources_by_target,
            optim_config=config,
            output_prob_threshold=0.01,
            device=device,
        )

    def print_circuit(
        self,
        circuit: OptimizedPromptAttributionResult,
        tokens: Int[Tensor, " seq"],
        tok: AppTokenizer,
        interp: "InterpRepo | None" = None,
        top_edges: int = 5,
        min_ci: float = 0.0,
    ) -> None:
        """Print a human-readable summary of an optimized circuit."""
        from collections import defaultdict

        spans = tok.get_spans(tokens.tolist())

        def parse_node(key: str) -> tuple[str, int, int]:
            parts = key.split(":")
            return ":".join(parts[:-2]), int(parts[-2]), int(parts[-1])

        def node_label(key: str) -> str:
            layer, seq, cidx = parse_node(key)
            label = ""
            if interp is not None:
                ir = interp.get_interpretation(f"{layer}:{cidx}")
                if ir:
                    label = f" [{ir.label[:35]}]"
            return f"{layer}:{cidx}@{spans[seq].strip()}(p{seq}){label}"

        edges_by_target: dict[str, list[tuple[str, float, bool]]] = defaultdict(list)
        for e in circuit.edges:
            edges_by_target[str(e.target)].append((str(e.source), e.strength, e.is_cross_seq))

        print(f"Circuit: {len(circuit.edges)} edges, L0={circuit.metrics.l0_total:.0f}")
        print(f"Tokens: {list(enumerate(spans))}\n")

        for tgt_key in sorted(edges_by_target.keys()):
            ci = circuit.node_ci_vals.get(tgt_key, 0)
            if ci <= min_ci:
                continue

            sources = edges_by_target[tgt_key]
            sources.sort(key=lambda x: -abs(x[1]))

            print(f"{node_label(tgt_key)}  ci={ci:.3f}")
            for src_key, strength, cross_seq in sources[:top_edges]:
                cross = " [x-seq]" if cross_seq else ""
                print(f"  <- {node_label(src_key)}  attr={strength:+.4f}{cross}")
            print()

    # -- Editing (mask-based, runtime) -----------------------------------------

    def _edited_forward_batched(
        self,
        tokens: Int[Tensor, "1 seq"],
        edits: dict[str, float],
    ) -> Float[Tensor, "1 seq vocab"]:
        """Forward with component mask edits applied uniformly (batched internal)."""
        seq_len = tokens.shape[1]
        device = tokens.device

        component_masks = {
            layer: torch.ones(1, seq_len, C, device=device)
            for layer, C in self.model.module_to_c.items()
        }
        for key, value in edits.items():
            layer, idx = parse_component_key(key)
            assert layer in component_masks, f"Unknown layer: {layer}"
            component_masks[layer][0, :, idx] = value

        mask_infos = make_mask_infos(component_masks, routing_masks="all")
        return self.model(tokens, mask_infos=mask_infos)

    def _ci_guided_forward_batched(
        self,
        tokens: Int[Tensor, "1 seq"],
        edits: dict[str, float],
        ci_threshold: float,
    ) -> Float[Tensor, "1 seq vocab"]:
        """Forward with edits applied only where component CI exceeds threshold (batched)."""
        seq_len = tokens.shape[1]
        device = tokens.device

        output_with_cache = self.model(tokens, cache_type="input")
        ci_outputs = self.model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling="continuous",
            detach_inputs=False,
        )
        ci_vals = ci_outputs.lower_leaky

        component_masks = {
            layer: torch.ones(1, seq_len, C, device=device)
            for layer, C in self.model.module_to_c.items()
        }
        for key, value in edits.items():
            layer, idx = parse_component_key(key)
            assert layer in component_masks, f"Unknown layer: {layer}"
            high_ci = ci_vals[layer][0, :, idx] > ci_threshold
            component_masks[layer][0, high_ci, idx] = value

        mask_infos = make_mask_infos(component_masks, routing_masks="all")
        return self.model(tokens, mask_infos=mask_infos)

    def make_edit_fn(
        self,
        edits: dict[str, float],
        ci_threshold: float | None = None,
    ) -> ForwardFn:
        """Create a reusable unbatched tokens [seq] → logits [seq, vocab] function."""
        if ci_threshold is not None:
            return lambda tokens: self._ci_guided_forward_batched(
                tokens.unsqueeze(0), edits, ci_threshold
            ).squeeze(0)
        return lambda tokens: self._edited_forward_batched(tokens.unsqueeze(0), edits).squeeze(0)

    # -- Permanent weight editing ----------------------------------------------

    def without_components(self, ablate_keys: list[str]) -> "EditableModel":
        """Deep copy with components permanently subtracted from target model weights.

        The returned model's target_model is a standard transformer — no CI
        function or mask_infos needed at inference.
        """
        edited_model = copy.deepcopy(self.model)

        by_layer: dict[str, list[int]] = {}
        for key in ablate_keys:
            layer, idx = parse_component_key(key)
            by_layer.setdefault(layer, []).append(idx)

        for layer_name, indices in by_layer.items():
            components = edited_model.components[layer_name]
            target_module = edited_model.target_model.get_submodule(layer_name)

            for idx in indices:
                contribution = (components.V[:, idx : idx + 1] @ components.U[idx : idx + 1, :]).T
                target_module.weight.data -= contribution  # pyright: ignore[reportOperatorIssue]

        return EditableModel(edited_model)


# -- Free functions (work with any ForwardFn) ----------------------------------


def generate(
    forward_fn: ForwardFn,
    tokens: Int[Tensor, " seq"],
    tokenizer: AppTokenizer,
    max_new_tokens: int = 30,
    temperature: float = 0.0,
) -> str:
    """Greedy (temperature=0) or sampled generation from an arbitrary forward function.

    Takes unbatched tokens [seq]. Strips trailing EOS to avoid the model
    treating the prompt as complete.
    """
    eos_id = tokenizer.eos_token_id
    if tokens[-1].item() == eos_id:
        tokens = tokens[:-1]
    generated = tokens.clone()
    for _ in range(max_new_tokens):
        logits = forward_fn(generated)
        next_logits = logits[-1]
        if temperature == 0:
            next_id = next_logits.argmax()
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1).squeeze()
        generated = torch.cat([generated, next_id.unsqueeze(0)])
        if next_id.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated.tolist())


def measure_kl(
    baseline_fn: ForwardFn,
    edited_fn: ForwardFn,
    token_seqs: list[Int[Tensor, " seq"]],
) -> AblationEffect:
    """KL divergence and perplexity shift between two forward functions.

    Takes unbatched token sequences [seq].
    """
    total_kl = 0.0
    total_baseline_nll = 0.0
    total_edited_nll = 0.0
    total_tokens = 0

    for tokens in token_seqs:
        if tokens.shape[0] < 3:
            continue

        with torch.no_grad():
            baseline_logits = baseline_fn(tokens)
            edited_logits = edited_fn(tokens)

        baseline_lp = F.log_softmax(baseline_logits[:-1], dim=-1)
        edited_lp = F.log_softmax(edited_logits[:-1], dim=-1)

        kl = F.kl_div(edited_lp, baseline_lp.exp(), reduction="sum", log_target=False)

        targets = tokens[1:]
        baseline_nll = -baseline_lp[range(len(targets)), targets].sum()
        edited_nll = -edited_lp[range(len(targets)), targets].sum()

        total_kl += kl.item()
        total_baseline_nll += baseline_nll.item()
        total_edited_nll += edited_nll.item()
        total_tokens += len(targets)

    assert total_tokens > 0, "No tokens to evaluate"
    return AblationEffect(
        mean_kl=total_kl / total_tokens,
        baseline_ppl=torch.exp(torch.tensor(total_baseline_nll / total_tokens)).item(),
        edited_ppl=torch.exp(torch.tensor(total_edited_nll / total_tokens)).item(),
        n_tokens=total_tokens,
    )


def measure_token_probs(
    baseline_fn: ForwardFn,
    edited_fn: ForwardFn,
    token_seqs: list[Int[Tensor, " seq"]],
    token_groups: dict[str, list[int]],
) -> dict[str, TokenGroupShift]:
    """Probability shift for named groups of token IDs between two forward functions.

    Takes unbatched token sequences [seq].
    """
    baseline_sums: dict[str, float] = {name: 0.0 for name in token_groups}
    edited_sums: dict[str, float] = {name: 0.0 for name in token_groups}
    total_positions = 0

    for tokens in token_seqs:
        with torch.no_grad():
            baseline_logits = baseline_fn(tokens)
            edited_logits = edited_fn(tokens)

        bp = F.softmax(baseline_logits, dim=-1)
        ep = F.softmax(edited_logits, dim=-1)

        for name, ids in token_groups.items():
            baseline_sums[name] += bp[:, ids].sum().item()
            edited_sums[name] += ep[:, ids].sum().item()
        total_positions += bp.shape[0]

    assert total_positions > 0
    return {
        name: TokenGroupShift(
            group_name=name,
            baseline_mean_prob=baseline_sums[name] / total_positions,
            edited_mean_prob=edited_sums[name] / total_positions,
            n_positions=total_positions,
        )
        for name in token_groups
    }
