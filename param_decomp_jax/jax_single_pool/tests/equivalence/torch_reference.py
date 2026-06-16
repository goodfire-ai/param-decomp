"""Torch REFERENCE values for the four PD loss terms on the shared fixtures.

Run in the torch (`param-decomp`) env. Computes each term using the REAL torch reference
functions (the load-bearing math), on the byte-identical fixtures, and writes
`torch_reference.json` for `jax_equivalence.py` to compare against:

  * **faith** — `param_decomp.metrics.faithfulness.faithfulness_loss` directly on the
    weight-deltas. No model needed.
  * **imp**  — `param_decomp.metrics.importance_minimality.importance_minimality_loss`
    on the UPPER-leaky CI (the term's real input), with the production p/beta/eps.
  * **stoch** — the per-chunk subset recon, mirroring `chunkwise_subset_recon` +
    `recon_one_forward`: for each chunk (one decomposed layer's 3 sites), build the
    component mask `ci+(1-ci)*u`, the weight-delta mask, and the uniform-k-subset routing
    from the fixtures, run the masked suffix forward (real `LinearComponents.forward` for
    the chunk's sites; frozen `F.linear` for every other layer + the tail), and
    `recon_loss_kl(pred, clean)/n_positions`. Mean over the chunks.
  * **stoch_route_all** — the same per-chunk subset recon as `stoch`, but every chunk
    `ComponentsMaskInfo` carries `routing_mask="all"` so the live sites have no
    per-position `torch.where` fallback. Isolates the static live-set frozen-site path
    (every non-chunk layer runs the frozen MLP) from per-position routing (parity R-2).
  * **ppgd** — `get_ppgd_mask_infos` (real) builds the component masks (`ci+(1-ci)*src`)
    + the weight-delta channel from the fixed sources; masked suffix forward over ALL
    sites; `recon_loss_kl/n_positions`.

The frozen suffix (attn zeroed, rms_norm, plain MLP, lm_head) is identical math to the
JAX side; the per-site masked projection goes through torch's own `LinearComponents`,
so the comparison pins the JAX masking/loss math to the torch reference.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from param_decomp.components import LinearComponents
from param_decomp.masks import ComponentsMaskInfo
from param_decomp.metrics.faithfulness import faithfulness_loss
from param_decomp.metrics.importance_minimality import importance_minimality_loss
from param_decomp.metrics.persistent_pgd_state import get_ppgd_mask_infos
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl

HERE = Path(__file__).resolve().parent
KINDS = ("gate", "up", "down")


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


def _frozen_mlp(x: torch.Tensor, Wg, Wu, Wd) -> torch.Tensor:
    return (F.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T


class _Suffix:
    """The fixed frozen suffix + per-layer `LinearComponents`, loaded from fixtures."""

    def __init__(self, f: dict[str, np.ndarray]) -> None:
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
        self.eps = float(f["_scalar_EPS"])
        self.n_layers = int(f["_scalar_N_DECOMP_LAYERS"])
        self.n_tail = int(f["_scalar_N_TAIL"])
        self.C = int(f["_scalar_C"])
        self.resid = t(f["resid"])
        self.norm = t(f["norm"])
        self.lm_head = t(f["lm_head"])

        self.ln1 = [t(f[f"ln1_{i}"]) for i in range(self.n_layers)]
        self.ln2 = [t(f[f"ln2_{i}"]) for i in range(self.n_layers)]
        self.W = [
            {"gate": t(f[f"Wg_{i}"]), "up": t(f[f"Wu_{i}"]), "down": t(f[f"Wd_{i}"])}
            for i in range(self.n_layers)
        ]
        # One LinearComponents per (layer, kind), target_weight = the frozen W.
        self.comp: list[dict[str, LinearComponents]] = []
        for i in range(self.n_layers):
            layer: dict[str, LinearComponents] = {}
            for k, (Vname, Uname) in zip(
                KINDS, [("Vg", "Ug"), ("Vu", "Uu"), ("Vd", "Ud")], strict=True
            ):
                V = t(f[f"{Vname}_{i}"])  # (d_in, C)
                U = t(f[f"{Uname}_{i}"])  # (C, d_out)
                d_in, Cc = V.shape
                d_out = U.shape[1]
                lc = LinearComponents(C=Cc, d_in=d_in, d_out=d_out, bias=None)
                with torch.no_grad():
                    lc.V.copy_(V)
                    lc.U.copy_(U)
                lc.register_buffer("target_weight", self.W[i][k])
                layer[k] = lc
            self.comp.append(layer)

        self.tail = [
            {"ln1": t(f[f"tail_ln1_{j}"]), "ln2": t(f[f"tail_ln2_{j}"]),
             "Wg": t(f[f"tail_Wg_{j}"]), "Wu": t(f[f"tail_Wu_{j}"]), "Wd": t(f[f"tail_Wd_{j}"])}
            for j in range(self.n_tail)
        ]  # fmt: skip

    def weight_deltas(self) -> dict[str, torch.Tensor]:
        """`{site: W - VU}` per (layer,kind) — the faith input (matches `calc_weight_deltas`)."""
        out: dict[str, torch.Tensor] = {}
        for i in range(self.n_layers):
            for k in KINDS:
                lc = self.comp[i][k]
                out[f"l{i}.{k}"] = lc.target_weight - lc.weight
        return out

    def _mlp_masked(
        self, i: int, mlp_in: torch.Tensor, mask_infos: dict[str, ComponentsMaskInfo] | None
    ) -> torch.Tensor:
        """Layer i's MLP: each of gate/up/down via `LinearComponents.forward` when a
        mask_info is present, else the frozen `target_forward` (== `F.linear`).
        Routing applied inside `LinearComponents`-style `torch.where` to match the model."""

        def proj(k: str, x: torch.Tensor) -> torch.Tensor:
            lc = self.comp[i][k]
            mi = None if mask_infos is None else mask_infos.get(f"l{i}.{k}")
            if mi is None:
                return F.linear(x, lc.target_weight)
            components_out = lc(
                x, mask=mi.component_mask, weight_delta_and_mask=mi.weight_delta_and_mask
            )
            if mi.routing_mask == "all":
                return components_out
            return torch.where(
                mi.routing_mask[..., None], components_out, F.linear(x, lc.target_weight)
            )

        gate = proj("gate", mlp_in)
        up = proj("up", mlp_in)
        return proj("down", F.silu(gate) * up)

    def forward(
        self,
        mask_infos: dict[str, ComponentsMaskInfo] | None,
        decompose_layers: set[int] | None,
    ) -> torch.Tensor:
        """Masked suffix logits. `decompose_layers` (None = all) selects which decomposed
        layers run the masked MLP; the rest run their frozen MLP. Attn is zeroed, so
        `post_attn == resid`."""
        x = self.resid
        for i in range(self.n_layers):
            post_attn = x  # attn contributes 0
            mlp_in = _rms_norm(post_attn, self.ln2[i], self.eps)
            if decompose_layers is None or i in decompose_layers:
                mlp_out = self._mlp_masked(i, mlp_in, mask_infos)
            else:
                mlp_out = _frozen_mlp(mlp_in, self.W[i]["gate"], self.W[i]["up"], self.W[i]["down"])
            x = post_attn + mlp_out
        for blk in self.tail:
            post_attn = x
            mlp_in = _rms_norm(post_attn, blk["ln2"], self.eps)
            x = post_attn + _frozen_mlp(mlp_in, blk["Wg"], blk["Wu"], blk["Wd"])
        x = _rms_norm(x, self.norm, self.eps)
        return x @ self.lm_head.T


def main() -> None:
    f = dict(np.load(HERE / "fixtures.npz"))
    s = _Suffix(f)
    B, T = int(f["_scalar_B"]), int(f["_scalar_T"])
    n_positions = B * T

    # clean target logits (no masks, all layers clean).
    clean = s.forward(mask_infos=None, decompose_layers=None).detach()

    # ---- faith ----
    faith = faithfulness_loss(s.weight_deltas()).item()

    # ---- imp (upper-leaky CI) ----
    ci_upper = {
        f"l{i}.{k}": torch.tensor(f[f"ci_upper_{k}"][:, :, i, :], dtype=torch.float32)
        for i in range(s.n_layers)
        for k in KINDS
    }
    imp = importance_minimality_loss(
        ci_upper_leaky=ci_upper,
        current_frac_of_training=1.0,
        eps=float(f["_scalar_IMP_EPS"]),
        pnorm=float(f["_scalar_IMP_P"]),
        beta=float(f["_scalar_IMP_BETA"]),
        p_anneal_start_frac=1.0,
        p_anneal_final_p=None,
        p_anneal_end_frac=1.0,
    ).item()

    # ---- stoch (per-chunk subset recon) ----
    ci_lower_layer = {
        k: torch.tensor(f[f"ci_lower_{k}"], dtype=torch.float32) for k in KINDS
    }  # (B,T,L,C)
    stoch_sum = 0.0
    for i in range(s.n_layers):  # one chunk per decomposed layer
        mask_infos: dict[str, ComponentsMaskInfo] = {}
        for k in KINDS:
            ci_s = ci_lower_layer[k][:, :, i, :]  # (B,T,C)
            u = torch.tensor(f[f"stoch_u_{k}"][:, :, i, :], dtype=torch.float32)
            comp_mask = ci_s + (1 - ci_s) * u
            delta = s.comp[i][k].target_weight - s.comp[i][k].weight
            delta_mask = torch.tensor(f[f"stoch_delta_{k}"][:, :, i], dtype=torch.float32)
            routing = torch.tensor(f[f"route_chunk{i}_{k}"], dtype=torch.bool)
            mask_infos[f"l{i}.{k}"] = ComponentsMaskInfo(
                component_mask=comp_mask,
                routing_mask=routing,
                weight_delta_and_mask=(delta, delta_mask),
            )
        pred = s.forward(mask_infos=mask_infos, decompose_layers={i})
        kl_sum, n = recon_loss_kl(pred=pred, target=clean)
        assert n == n_positions
        stoch_sum += (kl_sum / n).item()
    stoch = stoch_sum / s.n_layers

    # ---- stoch_route_all (per-chunk subset recon, routing=all → static live-set only) ----
    stoch_route_all_sum = 0.0
    for i in range(s.n_layers):
        mask_infos = {}
        for k in KINDS:
            ci_s = ci_lower_layer[k][:, :, i, :]  # (B,T,C)
            u = torch.tensor(f[f"stoch_u_{k}"][:, :, i, :], dtype=torch.float32)
            comp_mask = ci_s + (1 - ci_s) * u
            delta = s.comp[i][k].target_weight - s.comp[i][k].weight
            delta_mask = torch.tensor(f[f"stoch_delta_{k}"][:, :, i], dtype=torch.float32)
            mask_infos[f"l{i}.{k}"] = ComponentsMaskInfo(
                component_mask=comp_mask,
                routing_mask="all",
                weight_delta_and_mask=(delta, delta_mask),
            )
        pred = s.forward(mask_infos=mask_infos, decompose_layers={i})
        kl_sum, n = recon_loss_kl(pred=pred, target=clean)
        assert n == n_positions
        stoch_route_all_sum += (kl_sum / n).item()
    stoch_route_all = stoch_route_all_sum / s.n_layers

    # ---- ppgd (all sites, persistent sources via get_ppgd_mask_infos) ----
    # Build per-site ci + weight_deltas + sources keyed by site, then the real mask infos.
    ci_site = {f"l{i}.{k}": ci_lower_layer[k][:, :, i, :] for i in range(s.n_layers) for k in KINDS}
    deltas_site = s.weight_deltas()
    sources_site = {
        f"l{i}.{k}": torch.tensor(f[f"ppgd_source_{k}"][:, :, i, :], dtype=torch.float32)
        for i in range(s.n_layers)
        for k in KINDS
    }  # (1, T, C+1) broadcast over batch
    batch_dims = (B, T)
    ppgd_mask_infos = get_ppgd_mask_infos(
        ci=ci_site,
        weight_deltas=deltas_site,
        ppgd_sources=sources_site,
        routing_masks="all",
        batch_dims=batch_dims,
    )
    pred = s.forward(mask_infos=ppgd_mask_infos, decompose_layers=None)
    kl_sum, n = recon_loss_kl(pred=pred, target=clean)
    ppgd = (kl_sum / n).item()

    out = {
        "faith": faith,
        "imp": imp,
        "stoch": stoch,
        "stoch_route_all": stoch_route_all,
        "ppgd": ppgd,
        "n_chunks": s.n_layers,
    }
    (HERE / "torch_reference.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
