"""2-pool benchmark on Qwen3-0.6B-Base, rebalanced topology.

vs the toy `two_pool_maxA`:
  - target is real (Qwen3-0.6B-Base, 28 layers, hidden=1024, 196 sites)
  - topology rebalanced: pool A was previously massively under-utilized (42ms
    work, 155ms idle while pool B was the floor). Going from 42A+6B → 28A+20B:
        * pool A drops from 42 ranks to 28; each rank now owns 7 sites
          (one full transformer block worth) instead of 1 — pool A does more
          work per rank.
        * pool B grows from 6 to 20 ranks — more parallelism on PPGD, which
          was the bottleneck.

batch_global=40 (divisible by N_POOL_B=20). pool A rank sees full batch,
pool B rank sees batch_local_b=2.

Layout: 6 nodes × 8 GPUs = 48 GPUs total.
Per node: ranks 0-7. We allocate pool B at one per node (last rank on first 6
nodes... no wait that's only 6 pool B). Need 20 pool B ranks. So pool B is
~3-4 per node, pool A fills the rest.

Run:
    sbatch param_decomp/scripts/two_pool_benchmark/qwen_2pool.sbatch
"""

# pyright: reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false

import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor
from transformers import AutoModelForCausalLM

from param_decomp.configs import (
    AttnConfig,
    GlobalCiConfig,
    GlobalSharedTransformerCiConfig,
    PerBatchPerPositionScope,
    PersistentPGDReconLossConfig,
    ScheduleConfig,
    SignPGDConfig,
)
from param_decomp.models.batch_and_loss_fns import make_run_batch, recon_loss_kl
from param_decomp.two_pool import BlockGroup, PhaseProfiler, TwoPoolConfig, optimize_two_pool

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
N_LAYERS = 28
ATTN_SUBS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
MLP_SUBS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
SITES_PER_LAYER = ATTN_SUBS + MLP_SUBS  # 7
N_SITES = N_LAYERS * len(SITES_PER_LAYER)  # 196
assert N_SITES == 196

BATCH = 32
SEQ_LEN = 1024
VOCAB = 151936
C = 32

# CI fn — global shared transformer over the per-rank's owned sites.
CI_D_MODEL = 128
CI_N_BLOCKS = 2
CI_N_HEADS = 4

# Topology: 112A + 8B = 120 GPUs across 15 nodes.
# Pool A is split into 56 block groups of 2 ranks each (intra-block DDP-2):
#   - 28 attention groups (one per layer, 4 sites each: q/k/v/o)
#   - 28 mlp groups       (one per layer, 3 sites each: gate/up/down)
# The mlp/attn split keeps within-group V/U sizes uniform (attn C=d_model,
# mlp C=d_mlp differ by 3x) so per-iter compute and memory are balanced.
# Pool B is shrunk from 20 → 8 ranks since pool B was massively over-staffed
# at the toy → Qwen transition (it finished its work in 638ms and waited 2.5s
# for pool A to finish layerwise).
N_NODES = 15
GPUS_PER_NODE = 8
WORLD_SIZE = N_NODES * GPUS_PER_NODE  # 120
N_BLOCK_GROUPS = 2 * N_LAYERS  # 56 (28 attn + 28 mlp)
N_PER_BLOCK_GROUP = 2
N_POOL_B = 8

# Layout: nodes 0-13 (112 GPUs) are pool A. Node 14 (8 GPUs) is pool B.
# Within pool A, each pair of consecutive ranks forms one block group (so
# the intra-block all-reduce is intra-node where possible).
POOL_A_RANKS: tuple[int, ...] = tuple(range(14 * GPUS_PER_NODE))  # 0..111
POOL_B_RANKS: tuple[int, ...] = tuple(range(14 * GPUS_PER_NODE, WORLD_SIZE))  # 112..119
assert len(POOL_A_RANKS) == N_BLOCK_GROUPS * N_PER_BLOCK_GROUP
assert len(POOL_B_RANKS) == N_POOL_B
BLOCK_GROUP_RANKS: tuple[tuple[int, ...], ...] = tuple(
    tuple(POOL_A_RANKS[i * N_PER_BLOCK_GROUP : (i + 1) * N_PER_BLOCK_GROUP])
    for i in range(N_BLOCK_GROUPS)
)

WARMUP_STEPS = 2
PROFILE_STEPS = 4


def all_sites_qwen() -> list[str]:
    """The 196 decomposable site paths in Qwen3."""
    return [f"model.layers.{i}.{sub}" for i in range(N_LAYERS) for sub in SITES_PER_LAYER]


def build_pile_buffer(n_tokens: int, *, rank: int, device: torch.device) -> Tensor:
    """Stream pile-uncopyrighted on rank 0, broadcast to all ranks.

    120 ranks all hitting HuggingFace at once causes transient FileNotFoundErrors
    on the pile shard downloads. Rank-0-then-broadcast also avoids 120× the
    tokenizer load + 120× the redundant tokenization work.
    """
    buf = torch.empty(n_tokens, dtype=torch.long, device=device)
    if rank == 0:
        from datasets import load_dataset  # local import — heavy
        from transformers import AutoTokenizer

        print(f"[qwen] rank0 tokenizing {n_tokens:,} pile tokens with Qwen tokenizer…", flush=True)
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        eos = tok.eos_token
        assert isinstance(eos, str)
        ds = load_dataset("monology/pile-uncopyrighted", streaming=True, split="train")
        ds = ds.shuffle(seed=42, buffer_size=2000)
        it = iter(ds)
        tokens: list[int] = []
        while len(tokens) < n_tokens:
            text = next(it)["text"]
            tokens.extend(tok.encode(eos + text, add_special_tokens=False))
        print(
            f"[qwen] rank0 tokenization complete: {len(tokens):,} tokens; broadcasting…", flush=True
        )
        buf.copy_(torch.tensor(tokens[:n_tokens], dtype=torch.long, device=device))
    dist.broadcast(buf, src=0)
    return buf


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == WORLD_SIZE, f"want {WORLD_SIZE}, got {world_size}"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Load Qwen in bf16 (target is frozen, no need for fp32). attn_implementation=sdpa
    # routes through PyTorch SDPA which dispatches to flash/cudnn under bf16.
    if rank == 0:
        print(f"[qwen] loading {MODEL_ID} in bf16…", flush=True)
    torch.manual_seed(0)
    target = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    target.requires_grad_(False)
    if rank == 0:
        n_params = sum(p.numel() for p in target.parameters())
        print(
            f"[qwen] loaded {n_params / 1e6:.0f}M params, dtype={next(target.parameters()).dtype}",
            flush=True,
        )

    all_sites_list = all_sites_qwen()
    assert len(all_sites_list) == N_SITES

    # 56 block groups: 28 attention groups (one per layer, 4 sites each: q/k/v/o)
    # followed by 28 mlp groups (one per layer, 3 sites each: gate/up/down).
    # Each group gets N_PER_BLOCK_GROUP ranks for intra-block DDP.
    attn_groups = tuple(
        BlockGroup(
            ranks=BLOCK_GROUP_RANKS[layer],
            owned_sites=tuple(f"model.layers.{layer}.{sub}" for sub in ATTN_SUBS),
        )
        for layer in range(N_LAYERS)
    )
    mlp_groups = tuple(
        BlockGroup(
            ranks=BLOCK_GROUP_RANKS[N_LAYERS + layer],
            owned_sites=tuple(f"model.layers.{layer}.{sub}" for sub in MLP_SUBS),
        )
        for layer in range(N_LAYERS)
    )
    block_groups = attn_groups + mlp_groups
    assert len(block_groups) == N_BLOCK_GROUPS
    c_per_site = {s: C for s in all_sites_list}

    ppgd_cfg = PersistentPGDReconLossConfig(
        coeff=1.0,
        scope=PerBatchPerPositionScope(),
        optimizer=SignPGDConfig(lr_schedule=ScheduleConfig(start_val=0.01)),
        n_warmup_steps=2,
        n_samples=1,
        use_sigmoid_parameterization=False,
    )

    # Qwen forward returns CausalLMOutput with .logits — use make_run_batch("logits").
    pool_config = TwoPoolConfig(
        block_groups=block_groups,
        pool_b_ranks=POOL_B_RANKS,
        batch_global=BATCH,
        c_per_site=c_per_site,
        ci_config=GlobalCiConfig(
            fn_type="global_shared_transformer",
            simple_transformer_ci_cfg=GlobalSharedTransformerCiConfig(
                d_model=CI_D_MODEL,
                n_blocks=CI_N_BLOCKS,
                attn_config=AttnConfig(n_heads=CI_N_HEADS),
            ),
        ),
        sigmoid_type="leaky_hard",
        run_batch=make_run_batch("logits"),
        reconstruction_loss=recon_loss_kl,
        ppgd_cfg=ppgd_cfg,
        bf16_autocast=True,
    )

    if rank == 0:
        print(
            f"[qwen] 2-POOL  ({len(POOL_A_RANKS)}A + {N_POOL_B}B = {world_size} GPUs across "
            f"{N_NODES} nodes; {N_BLOCK_GROUPS} block groups (28 attn + 28 mlp), "
            f"{N_PER_BLOCK_GROUP}-way intra-block DDP)",
            flush=True,
        )
        print(
            f"[qwen] batch={BATCH} (A_local={BATCH // N_PER_BLOCK_GROUP} B_local={BATCH // N_POOL_B}) "
            f"seq={SEQ_LEN}  C={C}  "
            f"ci_d_model={CI_D_MODEL} ci_n_blocks={CI_N_BLOCKS} ci_n_heads={CI_N_HEADS}",
            flush=True,
        )
        print(
            f"[qwen] POOL_A_RANKS={POOL_A_RANKS[:8]}…{POOL_A_RANKS[-4:]} ({len(POOL_A_RANKS)} ranks)",
            flush=True,
        )
        print(f"[qwen] POOL_B_RANKS={POOL_B_RANKS}", flush=True)

    # Pre-tokenize enough pile to feed every step we'll run. Same on every rank.
    n_steps = WARMUP_STEPS + PROFILE_STEPS
    tokens_per_step = BATCH * SEQ_LEN
    pile_buffer = build_pile_buffer(n_steps * tokens_per_step, rank=rank, device=device)

    def batch_iter(step: int) -> Tensor:
        start = step * tokens_per_step
        end = start + tokens_per_step
        return pile_buffer[start:end].view(BATCH, SEQ_LEN)

    step_times: list[float] = []

    def on_step(step: int, metrics: dict[str, float]) -> None:
        torch.cuda.synchronize()
        step_times.append(time.perf_counter())
        if rank in (0, POOL_B_RANKS[0]):
            mem = torch.cuda.memory_allocated(device) / 1e9
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            metrics_str = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
            print(
                f"[qwen rank{rank}] step={step} mem={mem:.2f}GB peak={peak:.2f}GB {metrics_str}",
                flush=True,
            )

    torch.cuda.synchronize()
    step_times.append(time.perf_counter())

    profile_mode = os.environ.get("PROFILE_MODE", "off")
    assert profile_mode in ("sync", "async", "off"), f"PROFILE_MODE={profile_mode}"
    profiler = PhaseProfiler(enabled=(profile_mode != "off"), sync=(profile_mode == "sync"))
    if rank == 0:
        print(f"[qwen] PROFILE_MODE={profile_mode}", flush=True)

    optimize_two_pool(
        target_model=target,
        pool_config=pool_config,
        device=device,
        n_steps=WARMUP_STEPS + PROFILE_STEPS,
        batch_iter=batch_iter,
        on_step=on_step,
        profiler=profiler,
    )

    intervals = [step_times[i + 1] - step_times[i] for i in range(len(step_times) - 1)]
    profile = intervals[WARMUP_STEPS:]
    if profile and rank == 0:
        avg_ms = 1000 * sum(profile) / len(profile)
        per_sample = avg_ms / BATCH
        print(
            f"\n[qwen rank0] STEP_TOTAL avg={avg_ms:.2f}ms  "
            f"min={1000 * min(profile):.2f}ms  max={1000 * max(profile):.2f}ms  (n={len(profile)})",
            flush=True,
        )
        print(
            f"[qwen rank0] per-sample throughput: {1000 / per_sample:.1f} samples/sec/world  "
            f"({BATCH * SEQ_LEN * 1000 / avg_ms:.0f} tokens/sec/world)",
            flush=True,
        )

    if rank in (0, POOL_B_RANKS[0]):
        print(f"\n[qwen rank{rank}] phase breakdown (skipping first {WARMUP_STEPS}):", flush=True)
        print(profiler.report(warmup=WARMUP_STEPS), flush=True)

        if profile_mode != "off":
            out_dir = Path(os.environ.get("PROFILE_OUT_DIR", "/tmp/two_pool_profile"))
            out_dir.mkdir(parents=True, exist_ok=True)
            pool = "a" if rank == 0 else "b"
            out_path = out_dir / f"qwen_{profile_mode}_pool{pool}_rank{rank}.json"
            with open(out_path, "w") as f:
                json.dump(
                    profiler.to_json_dict(
                        warmup=WARMUP_STEPS, rank=rank, pool=pool, mode=profile_mode
                    ),
                    f,
                )
            print(f"[qwen rank{rank}] wrote spans → {out_path}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
