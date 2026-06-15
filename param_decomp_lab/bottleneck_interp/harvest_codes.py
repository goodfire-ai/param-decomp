"""Phase A of bottleneck-code interpretation: stream a corpus through a trained LM
decomposition and cache the sparse signed codes `z` per token position.

Saves, under `<out_dir>/`:
  - `codes_<chunk>.pt`   : fp16 tensor [n_positions, D] of bottleneck codes
  - `tokens_<chunk>.pt`  : int32 tensor [n_positions] of token ids (aligned to codes)
  - `theta.pt`           : fp32 [D] learned per-dim gate thresholds
  - `stats.pt`           : per-dim running statistics (firing/sign counts, sums)
  - `meta.json`          : run id, D, n_positions, n_batches, tokenizer name

The raw codes feed manifold analysis and autointerp example tables; the running stats
give the Phase-A summary (firing rate, mean active magnitude, sign distribution) without
a second pass.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor

from param_decomp.ci_fns import get_bottleneck
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.experiments.lm.run import SavedLMRun, build_lm_loader


@dataclass
class CodeStats:
    """Per-dim running statistics over all harvested positions."""

    dim: int
    device: str
    n_positions: int = 0
    fired_count: Tensor = field(init=False)
    pos_count: Tensor = field(init=False)
    neg_count: Tensor = field(init=False)
    abs_sum: Tensor = field(init=False)
    sq_sum: Tensor = field(init=False)

    def __post_init__(self) -> None:
        z = lambda: torch.zeros(self.dim, device=self.device, dtype=torch.float64)  # noqa: E731
        self.fired_count = z()
        self.pos_count = z()
        self.neg_count = z()
        self.abs_sum = z()
        self.sq_sum = z()

    def update(self, codes: Float[Tensor, "n D"]) -> None:
        c = codes.to(torch.float64)
        fired = c != 0
        self.fired_count += fired.sum(dim=0)
        self.pos_count += (c > 0).sum(dim=0)
        self.neg_count += (c < 0).sum(dim=0)
        self.abs_sum += c.abs().sum(dim=0)
        self.sq_sum += (c * c).sum(dim=0)
        self.n_positions += c.shape[0]

    def to_dict(self) -> dict[str, Tensor | int]:
        return {
            "n_positions": self.n_positions,
            "fired_count": self.fired_count.cpu(),
            "pos_count": self.pos_count.cpu(),
            "neg_count": self.neg_count.cpu(),
            "abs_sum": self.abs_sum.cpu(),
            "sq_sum": self.sq_sum.cpu(),
        }


def harvest_codes(
    run_path: str,
    out_dir: Path,
    n_batches: int,
    batch_size: int,
    positions_per_chunk: int,
    device: str,
) -> None:
    pd_run = SavedLMRun.from_path(run_path)
    model = pd_run.load_model().to(device).eval()

    bottleneck = get_bottleneck(model.ci_fn)
    assert bottleneck is not None, f"run {run_path} has no bottleneck CI fn"
    dim = bottleneck.bottleneck_dim

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bottleneck.gate.theta.detach().cpu(), out_dir / "theta.pt")

    loader = build_lm_loader(
        pd_run.cfg.target,
        pd_run.cfg.data,
        split="eval",
        device=device,
        batch_size=batch_size,
        dist_state=None,
        seed=pd_run.cfg.pd.seed,
    )

    stats = CodeStats(dim=dim, device=device)
    code_buf: list[Float[Tensor, "n D"]] = []
    tok_buf: list[Int[Tensor, " n"]] = []
    seq_buf: list[Int[Tensor, "b s"]] = []
    seq_len = -1
    buffered = 0
    chunk_idx = 0

    def flush() -> None:
        nonlocal code_buf, tok_buf, buffered, chunk_idx
        if buffered == 0:
            return
        torch.save(torch.cat(code_buf, dim=0), out_dir / f"codes_{chunk_idx:04d}.pt")
        torch.save(torch.cat(tok_buf, dim=0), out_dir / f"tokens_{chunk_idx:04d}.pt")
        code_buf, tok_buf, buffered = [], [], 0
        chunk_idx += 1

    loader_iter = iter(loader)
    for batch_idx in range(n_batches):
        batch: Int[Tensor, "b s"] = next(loader_iter).to(device)
        with torch.no_grad(), bf16_autocast(enabled=pd_run.cfg.runtime.autocast_bf16):
            cache = model(batch, cache_type="input").cache
            ci = model.calc_causal_importances(
                pre_weight_acts=cache, sampling="continuous", detach_inputs=True
            )
        codes = ci.bottleneck_codes
        assert codes is not None
        seq_len = batch.shape[1]
        flat_codes: Float[Tensor, "n D"] = codes.reshape(-1, dim).float()
        flat_toks: Int[Tensor, " n"] = batch.reshape(-1)

        stats.update(flat_codes)
        code_buf.append(flat_codes.half().cpu())
        tok_buf.append(flat_toks.to(torch.int32).cpu())
        seq_buf.append(batch.to(torch.int32).cpu())
        buffered += flat_codes.shape[0]
        if buffered >= positions_per_chunk:
            flush()
        if batch_idx % 10 == 0:
            print(f"batch {batch_idx}/{n_batches}  positions={stats.n_positions}", flush=True)

    flush()
    torch.save(stats.to_dict(), out_dir / "stats.pt")
    # Token sequences in flat-code order: global position i -> sequences[i // seq_len, i % seq_len].
    # Used to recover context windows for activating examples.
    torch.save(torch.cat(seq_buf, dim=0), out_dir / "sequences.pt")
    meta = {
        "run_path": run_path,
        "dim": dim,
        "n_positions": stats.n_positions,
        "n_batches": n_batches,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokenizer_name": pd_run.cfg.data.tokenizer_name,
        "n_chunks": chunk_idx,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {stats.n_positions} positions, {chunk_idx} chunks -> {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="wandb path or local run dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n_batches", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--positions_per_chunk", type=int, default=500_000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    harvest_codes(
        run_path=args.run,
        out_dir=args.out,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        positions_per_chunk=args.positions_per_chunk,
        device=args.device,
    )


if __name__ == "__main__":
    main()
