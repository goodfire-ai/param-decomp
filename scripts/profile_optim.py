"""Profile the optimize_ci_values loop to find where time is spent.

Breaks down the backward pass into model-backward vs CI-backward components.
"""

import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from spd.app.backend.optim_cis import (
    CELossConfig,
    OptimCIConfig,
    _compute_recon_loss,
    compute_alive_info,
    compute_l0_stats,
    compute_specific_pos_ce_kl,
    create_optimizable_ci_params,
)
from spd.configs import ImportanceMinimalityLossConfig
from spd.metrics import importance_minimality_loss
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.utils.general_utils import bf16_autocast

MODEL_PATH = "/mnt/polished-lake/artifacts/mechanisms/spd/runs/spd-s-55ea3f9b/model_400000.pth"
DEVICE = "cuda"
STEPS = 40
SEQ_LEN = 50


def timed(label: str, timings: dict[str, float]):
    """Context manager to time a block and accumulate into timings dict."""

    class Timer:
        def __enter__(self):
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, *args):
            torch.cuda.synchronize()
            timings[label] = timings.get(label, 0.0) + (time.perf_counter() - self.t0)

    return Timer()


def profile_main_loop():
    print("Loading model...")
    model = ComponentModel.from_pretrained(MODEL_PATH)
    model = model.to(DEVICE)
    model.requires_grad_(False)

    tokens = torch.randint(0, 1000, (1, SEQ_LEN), device=DEVICE)

    config = OptimCIConfig(
        seed=0,
        lr=1e-2,
        steps=STEPS,
        weight_decay=0.0,
        lr_schedule="cosine",
        lr_exponential_halflife=None,
        lr_warmup_pct=0.01,
        log_freq=max(1, STEPS // 4),
        imp_min_config=ImportanceMinimalityLossConfig(coeff=0.01, pnorm=0.5, beta=0.0),
        loss_config=CELossConfig(coeff=1.0, position=SEQ_LEN - 1, label_token=42),
        sampling="binary",
        ce_kl_rounding_threshold=0.5,
        mask_type="ci",
        adv_pgd=None,
    )

    print("Computing initial CI...")
    with torch.no_grad(), bf16_autocast():
        output_with_cache: OutputWithCache = model(tokens, cache_type="input")
        initial_ci_outputs = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=config.sampling,
            detach_inputs=False,
        )
        target_out = output_with_cache.output.detach()

    alive_info = compute_alive_info(initial_ci_outputs.lower_leaky)
    ci_params = create_optimizable_ci_params(
        alive_info=alive_info,
        initial_pre_sigmoid=initial_ci_outputs.pre_sigmoid,
    )
    weight_deltas = model.calc_weight_deltas()

    params = ci_params.get_parameters()
    optimizer = optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    n_params = len(params)
    n_layers = len(ci_params.ci_pre_sigmoid)
    total_elements = sum(p.numel() for p in params)
    print(f"\n{n_layers} layers, {n_params} parameter tensors, {total_elements} total elements")

    timings: dict[str, float] = {}
    log_freq = config.log_freq

    # Warmup step (GPU kernels, allocations)
    ci_outputs = ci_params.create_ci_outputs(model, DEVICE)
    recon_mask_infos = make_mask_infos(component_masks=ci_outputs.lower_leaky)
    with bf16_autocast():
        recon_out = model(tokens, mask_infos=recon_mask_infos)
    recon_loss = _compute_recon_loss(recon_out, config.loss_config, target_out, DEVICE)
    recon_loss.backward()
    optimizer.zero_grad()

    print(f"\nRunning {STEPS} steps (log_freq={log_freq})...")
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    for step in range(STEPS):
        with timed("zero_grad", timings):
            optimizer.zero_grad()

        with timed("create_ci_outputs", timings):
            ci_outputs = ci_params.create_ci_outputs(model, DEVICE)

        with timed("make_mask_infos", timings):
            recon_mask_infos = make_mask_infos(component_masks=ci_outputs.lower_leaky)

        with timed("model_forward", timings):
            with bf16_autocast():
                recon_out = model(tokens, mask_infos=recon_mask_infos)

        with timed("imp_min_loss", timings):
            imp_min_loss = importance_minimality_loss(
                ci_upper_leaky=ci_outputs.upper_leaky,
                current_frac_of_training=step / config.steps,
                pnorm=config.imp_min_config.pnorm,
                beta=config.imp_min_config.beta,
                eps=config.imp_min_config.eps,
                p_anneal_start_frac=config.imp_min_config.p_anneal_start_frac,
                p_anneal_final_p=config.imp_min_config.p_anneal_final_p,
                p_anneal_end_frac=config.imp_min_config.p_anneal_end_frac,
            )

        with timed("recon_loss", timings):
            recon_loss = _compute_recon_loss(recon_out, config.loss_config, target_out, DEVICE)
            total_loss = config.loss_config.coeff * recon_loss + config.imp_min_config.coeff * imp_min_loss

        # Logging
        if step % log_freq == 0 or step == STEPS - 1:
            with timed("logging", timings):
                l0_stats = compute_l0_stats(ci_outputs, ci_alive_threshold=0.0)
                with torch.no_grad():
                    ce_kl_stats = compute_specific_pos_ce_kl(
                        model=model,
                        batch=tokens,
                        target_out=target_out,
                        ci=ci_outputs.lower_leaky,
                        rounding_threshold=config.ce_kl_rounding_threshold,
                        loss_seq_pos=config.loss_config.position,
                    )

        with timed("backward", timings):
            total_loss.backward()

        with timed("optimizer_step", timings):
            optimizer.step()

    torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - total_start

    print(f"\n{'='*65}")
    print(f"Total: {total_elapsed:.3f}s ({total_elapsed/STEPS*1000:.1f}ms/step)")
    print(f"{'='*65}")
    print(f"\n{'Phase':<25} {'Total (s)':>10} {'Per step (ms)':>15} {'% total':>10}")
    print(f"{'-'*60}")
    for name, elapsed in sorted(timings.items(), key=lambda x: -x[1]):
        pct = 100 * elapsed / total_elapsed
        per_step = elapsed / STEPS * 1000
        print(f"{name:<25} {elapsed:>10.3f} {per_step:>15.1f} {pct:>9.1f}%")

    accounted = sum(timings.values())
    unaccounted = total_elapsed - accounted
    print(f"{'unaccounted':<25} {unaccounted:>10.3f} {unaccounted/STEPS*1000:>15.1f} {100*unaccounted/total_elapsed:>9.1f}%")

    n_log_steps = sum(1 for s in range(STEPS) if s % log_freq == 0 or s == STEPS - 1)
    print(f"\nLogging: {n_log_steps}/{STEPS} steps, {timings.get('logging', 0)/max(n_log_steps,1)*1000:.1f}ms each")


def profile_backward_breakdown():
    """Measure backward through recon_loss alone vs imp_min_loss alone."""
    print("\n\n" + "=" * 65)
    print("BACKWARD BREAKDOWN: recon_loss vs imp_min_loss")
    print("=" * 65)

    model = ComponentModel.from_pretrained(MODEL_PATH)
    model = model.to(DEVICE)
    model.requires_grad_(False)

    tokens = torch.randint(0, 1000, (1, SEQ_LEN), device=DEVICE)

    with torch.no_grad(), bf16_autocast():
        output_with_cache: OutputWithCache = model(tokens, cache_type="input")
        initial_ci_outputs = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling="binary",
            detach_inputs=False,
        )
        target_out = output_with_cache.output.detach()

    alive_info = compute_alive_info(initial_ci_outputs.lower_leaky)

    N = 10  # Iterations for timing

    # --- Measure recon_loss backward only ---
    recon_times = []
    for _ in range(N):
        ci_params = create_optimizable_ci_params(alive_info, initial_ci_outputs.pre_sigmoid)
        ci_outputs = ci_params.create_ci_outputs(model, DEVICE)
        recon_mask_infos = make_mask_infos(component_masks=ci_outputs.lower_leaky)
        with bf16_autocast():
            recon_out = model(tokens, mask_infos=recon_mask_infos)
        recon_loss = _compute_recon_loss(
            recon_out, CELossConfig(coeff=1.0, position=SEQ_LEN - 1, label_token=42), target_out, DEVICE
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        recon_loss.backward()
        torch.cuda.synchronize()
        recon_times.append(time.perf_counter() - t0)

    # --- Measure imp_min_loss backward only ---
    imp_min_times = []
    for _ in range(N):
        ci_params = create_optimizable_ci_params(alive_info, initial_ci_outputs.pre_sigmoid)
        ci_outputs = ci_params.create_ci_outputs(model, DEVICE)
        imp_min_loss = importance_minimality_loss(
            ci_upper_leaky=ci_outputs.upper_leaky,
            current_frac_of_training=0.5,
            pnorm=0.5,
            beta=0.0,
            eps=1e-8,
            p_anneal_start_frac=0.0,
            p_anneal_final_p=0.5,
            p_anneal_end_frac=1.0,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        imp_min_loss.backward()
        torch.cuda.synchronize()
        imp_min_times.append(time.perf_counter() - t0)

    # --- Measure combined backward ---
    combined_times = []
    for _ in range(N):
        ci_params = create_optimizable_ci_params(alive_info, initial_ci_outputs.pre_sigmoid)
        ci_outputs = ci_params.create_ci_outputs(model, DEVICE)
        recon_mask_infos = make_mask_infos(component_masks=ci_outputs.lower_leaky)
        with bf16_autocast():
            recon_out = model(tokens, mask_infos=recon_mask_infos)
        recon_loss = _compute_recon_loss(
            recon_out, CELossConfig(coeff=1.0, position=SEQ_LEN - 1, label_token=42), target_out, DEVICE
        )
        imp_min_loss = importance_minimality_loss(
            ci_upper_leaky=ci_outputs.upper_leaky,
            current_frac_of_training=0.5,
            pnorm=0.5,
            beta=0.0,
            eps=1e-8,
            p_anneal_start_frac=0.0,
            p_anneal_final_p=0.5,
            p_anneal_end_frac=1.0,
        )
        total_loss = recon_loss + 0.01 * imp_min_loss
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_loss.backward()
        torch.cuda.synchronize()
        combined_times.append(time.perf_counter() - t0)

    # --- Measure create_ci_outputs + backward through imp_min only (no model fwd) ---
    # This isolates the scatter backward cost
    ci_create_bwd_times = []
    for _ in range(N):
        ci_params = create_optimizable_ci_params(alive_info, initial_ci_outputs.pre_sigmoid)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ci_outputs = ci_params.create_ci_outputs(model, DEVICE)
        imp_min_loss = importance_minimality_loss(
            ci_upper_leaky=ci_outputs.upper_leaky,
            current_frac_of_training=0.5,
            pnorm=0.5,
            beta=0.0,
            eps=1e-8,
            p_anneal_start_frac=0.0,
            p_anneal_final_p=0.5,
            p_anneal_end_frac=1.0,
        )
        imp_min_loss.backward()
        torch.cuda.synchronize()
        ci_create_bwd_times.append(time.perf_counter() - t0)

    def fmt(times: list[float]) -> str:
        avg = sum(times) / len(times) * 1000
        mn = min(times) * 1000
        return f"{avg:.1f}ms (min={mn:.1f}ms)"

    print(f"\n  recon_loss.backward():     {fmt(recon_times)}")
    print(f"  imp_min_loss.backward():   {fmt(imp_min_times)}")
    print(f"  combined.backward():       {fmt(combined_times)}")
    print(f"  create_ci + imp_min + bwd: {fmt(ci_create_bwd_times)}")


def profile_create_ci_outputs_detail():
    """Break down create_ci_outputs: scatter loop vs sigmoid."""
    print("\n\n" + "=" * 65)
    print("CREATE_CI_OUTPUTS BREAKDOWN: scatter vs sigmoid")
    print("=" * 65)

    model = ComponentModel.from_pretrained(MODEL_PATH)
    model = model.to(DEVICE)
    model.requires_grad_(False)

    tokens = torch.randint(0, 1000, (1, SEQ_LEN), device=DEVICE)

    with torch.no_grad(), bf16_autocast():
        output_with_cache: OutputWithCache = model(tokens, cache_type="input")
        initial_ci_outputs = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling="binary",
            detach_inputs=False,
        )

    alive_info = compute_alive_info(initial_ci_outputs.lower_leaky)
    ci_params = create_optimizable_ci_params(alive_info, initial_ci_outputs.pre_sigmoid)

    N = 20
    scatter_times = []
    sigmoid_times = []

    for _ in range(N):
        pre_sigmoid: dict[str, torch.Tensor] = {}

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for layer_name, mask in ci_params.alive_info.alive_masks.items():
            full_pre_sigmoid = torch.zeros_like(mask, dtype=torch.float32, device=DEVICE)
            layer_pre_sigmoid_list = ci_params.ci_pre_sigmoid[layer_name]
            seq_len = mask.shape[1]
            for pos in range(seq_len):
                pos_mask = mask[0, pos, :]
                pos_pre_sigmoid = layer_pre_sigmoid_list[pos]
                full_pre_sigmoid[0, pos, pos_mask] = pos_pre_sigmoid
            pre_sigmoid[layer_name] = full_pre_sigmoid
        torch.cuda.synchronize()
        scatter_times.append(time.perf_counter() - t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lower_leaky = {k: model.lower_leaky_fn(v) for k, v in pre_sigmoid.items()}
        upper_leaky = {k: model.upper_leaky_fn(v) for k, v in pre_sigmoid.items()}
        torch.cuda.synchronize()
        sigmoid_times.append(time.perf_counter() - t0)

    def fmt(times: list[float]) -> str:
        avg = sum(times) / len(times) * 1000
        mn = min(times) * 1000
        return f"{avg:.1f}ms (min={mn:.1f}ms)"

    print(f"\n  scatter loop (24 layers × {SEQ_LEN} positions): {fmt(scatter_times)}")
    print(f"  sigmoid (lower + upper, 24 layers):          {fmt(sigmoid_times)}")

    # Now measure a full-tensor equivalent for comparison
    alive_mask_float = {k: v.float() for k, v in alive_info.alive_masks.items()}
    full_tensors = {}
    for layer_name, mask_f in alive_mask_float.items():
        tensor = initial_ci_outputs.pre_sigmoid[layer_name].clone().detach()
        tensor *= mask_f
        tensor.requires_grad_(True)
        full_tensors[layer_name] = tensor

    full_tensor_times = []
    for _ in range(N):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        masked = {k: v * alive_mask_float[k] for k, v in full_tensors.items()}
        lower = {k: model.lower_leaky_fn(v) for k, v in masked.items()}
        upper = {k: model.upper_leaky_fn(v) for k, v in masked.items()}
        torch.cuda.synchronize()
        full_tensor_times.append(time.perf_counter() - t0)

    print(f"  full-tensor equiv (multiply + sigmoid):      {fmt(full_tensor_times)}")


if __name__ == "__main__":
    profile_main_loop()
    profile_backward_breakdown()
    profile_create_ci_outputs_detail()
