"""Circuit builder: hand-crafted rank-1 LoRA edits assembled from PD subcomponents.

Core concepts (see the slack spec from the circuit-builder request):

- **Read vector**: the normalized input direction `v_hat = V[:, c] / ||V[:, c]||` of an
  existing subcomponent `c` at some decomposed site. The U column absorbs the norm, so
  the subcomponent activation used everywhere here is `a_c = ||U[c]|| * (x @ V[:, c])`
  (identical to harvest's `component_activation` scale).

- **j-vector**: for a *downstream* subcomponent `d` (site later in the forward order),
  the derivative of its activation with respect to the target model's activations `y`
  just after the read site's weight matrix, summed over the positions the activation
  depends on and averaged over prompts and sequence positions:

      j_d = E_{prompts, t} [ sum_{t'} d a_d[t'] / d y[t] ]

  Computed on the fly with autograd: the read site's output is replaced by a detached
  leaf, downstream site inputs are captured in-graph, and one backward per downstream
  subcomponent yields the (B, T, d_out) gradient which is then averaged over B and T.

- **LoRA**: a rank-1 update on the read site's weight matrix,

      dW = scale * (sum_i lambda_i * j_hat_i) (x) v_hat_read      # (d_out, d_in)

  so `(W + dW) x = W x + scale * (v_hat_read . x) * sum_i lambda_i j_hat_i`: whenever
  the read subcomponent's input direction fires, push the output toward directions that
  activate the chosen downstream subcomponents.

Everything here is pure torch on a ComponentModel — no CI fn, no harvest. Data access
(token batches for j-vector averaging, labels, activation examples) goes through the
small provider protocols at the bottom so the app can run against the real run or a
mock with the same code path.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from pydantic import BaseModel
from torch import Tensor, nn

from param_decomp.component_model import ComponentModel

# =============================================================================
# Site ordering
# =============================================================================

# Forward-order rank of each within-block site for the separate-attn schema.
# q/k/v share a rank (parallel reads of the residual stream — none is downstream
# of another); o is downstream of q/k/v; the MLP follows attention.
_WITHIN_BLOCK_RANK = {"attn.q_proj": 0, "attn.k_proj": 0, "attn.v_proj": 0,
                      "attn.o_proj": 1, "mlp.c_fc": 2, "mlp.down_proj": 3}
_RANKS_PER_BLOCK = 4


def site_rank(site: str) -> int:
    """Forward-order rank of a decomposed site like `h.2.attn.v_proj`."""
    parts = site.split(".")
    assert parts[0] == "h" and len(parts) == 4, f"unsupported site path: {site}"
    within = ".".join(parts[2:])
    assert within in _WITHIN_BLOCK_RANK, f"unsupported site: {site}"
    return int(parts[1]) * _RANKS_PER_BLOCK + _WITHIN_BLOCK_RANK[within]


def downstream_sites(model: ComponentModel, read_site: str) -> list[str]:
    """Decomposed sites strictly downstream of `read_site`'s weight output."""
    r = site_rank(read_site)
    return sorted(
        (s for s in model.target_module_paths if site_rank(s) > r),
        key=site_rank,
    )


# =============================================================================
# Subcomponent vectors
# =============================================================================


@dataclass(frozen=True)
class SubcomponentRef:
    site: str  # decomposed module path, e.g. "h.1.attn.v_proj"
    idx: int  # component index within the site


def read_vector(model: ComponentModel, ref: SubcomponentRef) -> Float[Tensor, " d_in"]:
    """Normalized read direction `v_hat` of a subcomponent (V column, unit norm)."""
    v = model.components[ref.site].V[:, ref.idx].detach().float()
    norm = v.norm()
    assert norm > 0, f"zero V column for {ref}"
    return v / norm


def u_norm_absorbed(model: ComponentModel, ref: SubcomponentRef) -> float:
    """||U[c]|| * ||V[:, c]|| — the output magnitude after V-normalization."""
    comps = model.components[ref.site]
    return float(comps.U[ref.idx, :].detach().float().norm() * comps.V[:, ref.idx].detach().float().norm())


# =============================================================================
# j-vectors
# =============================================================================


@dataclass
class JVectorResult:
    ref: SubcomponentRef
    j: Float[Tensor, " d_out"]  # raw averaged derivative (fp32, cpu)
    raw_norm: float

    @property
    def j_hat(self) -> Float[Tensor, " d_out"]:
        assert self.raw_norm > 0, f"zero j-vector for {self.ref}"
        return self.j / self.raw_norm


def compute_j_vectors(
    model: ComponentModel,
    read_site: str,
    targets: list[SubcomponentRef],
    token_batches: Iterator[Int[Tensor, "B T"]],
    n_prompts: int,
) -> list[JVectorResult]:
    """j-vectors of `targets` w.r.t. the activations just after `read_site`'s weight.

    Runs forwards over `token_batches` until `n_prompts` prompts (batch rows) are
    consumed. For each forward, the read site's output is swapped for a detached
    autograd leaf; downstream site inputs are captured in-graph; each target's
    activation `||U[c]|| * (x @ V[:, c])` is summed over batch and positions and
    differentiated w.r.t. the leaf. Per-token gradients are averaged over all
    prompts and positions.
    """
    assert targets, "no j-vector targets given"
    assert n_prompts > 0
    allowed = set(downstream_sites(model, read_site))
    for t in targets:
        assert t.site in allowed, f"{t.site} is not downstream of {read_site}"

    device = next(model.parameters()).device
    read_module = model.target_model.get_submodule(read_site)
    assert isinstance(read_module, nn.Linear), f"read site must be nn.Linear, got {read_module}"
    d_out = read_module.out_features

    target_sites = sorted({t.site for t in targets}, key=site_rank)
    sums = {t: torch.zeros(d_out, dtype=torch.float64) for t in targets}
    n_tokens_total = 0
    n_prompts_seen = 0

    for token_ids in token_batches:
        if n_prompts_seen >= n_prompts:
            break
        token_ids = token_ids[: n_prompts - n_prompts_seen].to(device)
        n_prompts_seen += token_ids.shape[0]

        captured: dict[str, Tensor] = {}

        def make_read_hook() -> object:
            def hook(_m: nn.Module, _args: tuple, output: Tensor) -> Tensor:
                leaf = output.detach().clone().requires_grad_(True)
                captured["__read_leaf__"] = leaf
                return leaf
            return hook

        def make_input_hook(site: str) -> object:
            def hook(_m: nn.Module, args: tuple, _output: Tensor) -> None:
                captured[site] = args[0]
            return hook

        handles = [read_module.register_forward_hook(make_read_hook())]
        handles += [
            model.target_model.get_submodule(s).register_forward_hook(make_input_hook(s))
            for s in target_sites
        ]
        try:
            with torch.enable_grad():
                model(token_ids)  # plain target forward; hooks do the work
                leaf = captured["__read_leaf__"]
                B, T = token_ids.shape
                for i, t in enumerate(targets):
                    x_d = captured[t.site]
                    comps = model.components[t.site]
                    act = x_d.float() @ comps.V[:, t.idx].detach().float()
                    act = act * comps.U[t.idx, :].detach().float().norm()
                    (grad,) = torch.autograd.grad(
                        act.sum(), leaf, retain_graph=(i < len(targets) - 1)
                    )
                    sums[t] += grad.double().sum(dim=(0, 1)).cpu()
                n_tokens_total += B * T
        finally:
            for h in handles:
                h.remove()

    assert n_prompts_seen == n_prompts, (
        f"token_batches exhausted after {n_prompts_seen}/{n_prompts} prompts"
    )
    results = []
    for t in targets:
        j = (sums[t] / n_tokens_total).float()
        results.append(JVectorResult(ref=t, j=j, raw_norm=float(j.norm())))
    return results


# =============================================================================
# LoRA assembly + application
# =============================================================================


class WriteTerm(BaseModel):
    """One term of a LoRA's write vector: a downstream subcomponent + prefactor."""

    site: str
    idx: int
    weight: float  # lambda_i


class LoraSpec(BaseModel):
    """A rank-1 LoRA: read subcomponent + weighted sum of downstream j-vectors."""

    name: str
    read_site: str
    read_idx: int
    writes: list[WriteTerm]
    scale: float = 1.0  # global multiplier on dW
    normalize_j: bool = True  # use j_hat (unit norm) so lambda is the magnitude
    n_prompts: int = 16  # prompts to average j-vectors over
    enabled: bool = True


@dataclass
class BuiltLora:
    """A LoRA spec resolved against a model: the actual dW and its ingredients."""

    spec: LoraSpec
    delta_w: Float[Tensor, "d_out d_in"]
    j_norms: dict[str, float] = field(default_factory=dict)  # "site:idx" -> raw |j|


def build_lora(
    model: ComponentModel,
    spec: LoraSpec,
    j_results: list[JVectorResult],
) -> BuiltLora:
    """Assemble `dW = scale * (sum_i lambda_i j_i) (x) v_hat_read` from computed j-vectors."""
    assert spec.writes, f"LoRA {spec.name!r} has no write terms"
    by_ref = {r.ref: r for r in j_results}
    v_hat = read_vector(model, SubcomponentRef(spec.read_site, spec.read_idx))

    d_out = model.target_model.get_submodule(spec.read_site).out_features  # type: ignore[union-attr]
    w_total = torch.zeros(d_out)
    j_norms: dict[str, float] = {}
    for term in spec.writes:
        r = by_ref[SubcomponentRef(term.site, term.idx)]
        j_norms[f"{term.site}:{term.idx}"] = r.raw_norm
        w_total += term.weight * (r.j_hat if spec.normalize_j else r.j)

    delta_w = spec.scale * torch.outer(w_total, v_hat)
    return BuiltLora(spec=spec, delta_w=delta_w, j_norms=j_norms)


@contextmanager
def apply_loras(model: ComponentModel, loras: list[BuiltLora]) -> Iterator[None]:
    """Temporarily add each LoRA's dW to its read site's weight (restored on exit)."""
    touched: list[tuple[nn.Linear, Tensor]] = []
    try:
        for lora in loras:
            if not lora.spec.enabled:
                continue
            module = model.target_model.get_submodule(lora.spec.read_site)
            assert isinstance(module, nn.Linear)
            touched.append((module, module.weight.data.clone()))
            module.weight.data += lora.delta_w.to(module.weight.device, module.weight.dtype)
        yield
    finally:
        for module, original in reversed(touched):
            module.weight.data = original


# =============================================================================
# Base-vs-edited comparison
# =============================================================================


class TokenLogit(BaseModel):
    token: str
    token_id: int
    logit: float
    prob: float


class PositionComparison(BaseModel):
    position: int
    token: str  # prompt token at this position
    kl_base_to_edited: float  # KL(base || edited) of next-token distributions
    top_base: list[TokenLogit]
    top_edited: list[TokenLogit]


class GenerationResult(BaseModel):
    greedy: str
    sampled: str


class CompareResult(BaseModel):
    prompt_tokens: list[str]
    positions: list[PositionComparison]
    base: GenerationResult
    edited: GenerationResult
    mean_kl: float


class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode_tokens(self, token_ids: list[int]) -> list[str]: ...


@torch.no_grad()
def _generate(
    model: ComponentModel,
    token_ids: Int[Tensor, "1 T"],
    max_new_tokens: int,
    temperature: float,
    seed: int,
    greedy: bool,
) -> list[int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = token_ids.clone()
    out: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :].float()
        if greedy:
            nxt = int(logits.argmax(dim=-1).item())
        else:
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1).cpu()
            nxt = int(torch.multinomial(probs[0], 1, generator=generator).item())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return out


@torch.no_grad()
def _topk(logits: Float[Tensor, " vocab"], tokenizer: TokenizerProtocol, k: int) -> list[TokenLogit]:
    probs = F.softmax(logits, dim=-1)
    vals, idxs = logits.topk(k)
    tokens = tokenizer.decode_tokens([int(i) for i in idxs])
    return [
        TokenLogit(token=tok, token_id=int(i), logit=float(v), prob=float(probs[i]))
        for tok, i, v in zip(tokens, idxs, vals, strict=True)
    ]


def compare_models(
    model: ComponentModel,
    loras: list[BuiltLora],
    tokenizer: TokenizerProtocol,
    prompt: str,
    *,
    top_k: int = 10,
    max_new_tokens: int = 32,
    temperature: float = 0.8,
    seed: int = 0,
) -> CompareResult:
    """Full base-vs-edited comparison on one prompt: per-position logits + continuations."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    assert ids, "empty prompt"
    token_ids = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_tokens = tokenizer.decode_tokens(ids)

    with torch.no_grad():
        base_logits = model(token_ids)[0].float()
    base_greedy = _generate(model, token_ids, max_new_tokens, temperature, seed, greedy=True)
    base_sampled = _generate(model, token_ids, max_new_tokens, temperature, seed, greedy=False)

    with apply_loras(model, loras):
        with torch.no_grad():
            edited_logits = model(token_ids)[0].float()
        edited_greedy = _generate(model, token_ids, max_new_tokens, temperature, seed, greedy=True)
        edited_sampled = _generate(model, token_ids, max_new_tokens, temperature, seed, greedy=False)

    base_logprobs = F.log_softmax(base_logits, dim=-1)
    edited_logprobs = F.log_softmax(edited_logits, dim=-1)
    kls = F.kl_div(edited_logprobs, base_logprobs, log_target=True, reduction="none").sum(-1)

    positions = [
        PositionComparison(
            position=t,
            token=prompt_tokens[t],
            kl_base_to_edited=float(kls[t]),
            top_base=_topk(base_logits[t], tokenizer, top_k),
            top_edited=_topk(edited_logits[t], tokenizer, top_k),
        )
        for t in range(len(ids))
    ]
    return CompareResult(
        prompt_tokens=prompt_tokens,
        positions=positions,
        base=GenerationResult(
            greedy="".join(tokenizer.decode_tokens(base_greedy)),
            sampled="".join(tokenizer.decode_tokens(base_sampled)),
        ),
        edited=GenerationResult(
            greedy="".join(tokenizer.decode_tokens(edited_greedy)),
            sampled="".join(tokenizer.decode_tokens(edited_sampled)),
        ),
        mean_kl=float(kls.mean()),
    )


# =============================================================================
# Data / label providers (real run vs mock)
# =============================================================================


class TokenBatchProvider(Protocol):
    """Yields (B, T) token batches for j-vector averaging."""

    def batches(self, batch_size: int, seq_len: int) -> Iterator[Int[Tensor, "B T"]]: ...


class SubcomponentInfoProvider(Protocol):
    """Labels + activating examples for subcomponents (autointerp-backed or mock)."""

    def label(self, site: str, idx: int) -> str | None: ...
    def activating_examples(self, site: str, idx: int, limit: int) -> list[dict]: ...
