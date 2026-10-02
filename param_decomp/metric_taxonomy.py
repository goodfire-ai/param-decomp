"""Pure metric naming grammar shared by logging and workspace presentation."""

import re

from param_decomp.metric_names import MetricKey

_PERFORMANCE = {
    "step_time_s",
    "hfu",
    "flops_per_step",
    "elapsed_s",
    "eta_s",
    "runtime_s",
    "training_time_s",
    "n_completed_steps",
    "window_training_time_s",
    "n_window_steps",
    "useful_model_flops",
    "useful_optimizer_flops",
    "useful_flops",
    "overall_mfu_without_optimizer",
    "overall_mfu_with_optimizer",
    "step_mfu_without_optimizer",
    "step_mfu_with_optimizer",
}
_CE_VARIANTS = {
    "ci_masked",
    "unmasked",
    "stoch_masked",
    "random_masked",
    "rounded_masked",
    "zero_masked",
}
_ROUTER_STRATEGIES = {"ci_masked", "stochastic", "fresh_pgd", "persistent"}
_UNIT_KINDS = {"neuron", "attention_head", "deltanet_head"}

PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

LABELS = {
    "self_attn.q_proj": "attn-q",
    "self_attn.k_proj": "attn-k",
    "self_attn.v_proj": "attn-v",
    "self_attn.o_proj": "attn-o",
    "mlp.gate_proj": "mlp-gate",
    "mlp.up_proj": "mlp-up",
    "mlp.down_proj": "mlp-down",
    "effective_use_count_per_subcomponent": "effective-use-per-subcomponent",
    "soft_use_count_relative_threshold_4": "soft-use-relative-threshold-4",
    "n_components": "component-count",
    "mean_ci_gt_0": "positive-mean-ci",
    "by_site": "sites",
    "by_unit_kind": "unit-kind",
    "group_sum": "group-sum",
    "site_reduction": "site-summary",
    "hidden_acts_reconstruction": "hidden-acts",
    "router_reconstruction": "router",
}

CI_FN_BLOCK_PARAMETERS = {
    "wq": "01-attn-q",
    "wk": "02-attn-k",
    "wv": "03-attn-v",
    "wo": "04-attn-o",
    "w1": "05-ffn-input-weight",
    "b1": "06-ffn-input-bias",
    "w2": "07-ffn-output-weight",
    "b2": "08-ffn-output-bias",
    "shared_gate": "13-shared-gate",
    "shared_up": "14-shared-up",
    "shared_down": "15-shared-down",
}
# Chunkwise and global transformers carry these under `backbone`, and a component-conditioned
# global transformer under `backbone.backbone`; the block-selected transformer carries them
# directly.
CI_FN_SHARED_LEAVES = frozenset(
    {
        "ci_fns.chunks.in_proj_w",
        "ci_fns.chunks.in_proj_b",
        "ci_fns.inv_freq",
        "ci_fns.backbone.chunks.in_proj_w",
        "ci_fns.backbone.chunks.in_proj_b",
        "ci_fns.backbone.in_proj_w",
        "ci_fns.backbone.in_proj_b",
        "ci_fns.backbone.inv_freq",
        "ci_fns.backbone.backbone.in_proj_w",
        "ci_fns.backbone.backbone.in_proj_b",
        "ci_fns.backbone.backbone.inv_freq",
    }
)
CI_FN_BLOCK_BRANCHES = {
    "gate": {"branch_00": "09-gate-weight", "branch_01": "10-gate-bias"},
    "norm_scales": {"branch_00": "11-attn-norm", "branch_01": "12-ffn-norm"},
}
WEIGHT_BIAS = {"weight": "01-weight", "bias": "02-bias"}
SUMMARY_ROWS = {"components": "01-components", "ci_fns": "02-ci-fns", "total": "03-total"}
SUMMARY_COLUMNS = {"window_min": "01-min", "window_median": "02-median", "window_max": "03-max"}


def _authored_label(value: str) -> str:
    # Loss, auxiliary, capture and group names are authored, not fixed metric vocabulary.
    if re.fullmatch(r"[a-z0-9]+(?:[_.-][a-z0-9]+)*", value) is None:
        raise ValueError(f"Invalid authored metric label: {value!r}")
    return value


def _threshold(value: str) -> str:
    if re.fullmatch(r"\d+(?:\.\d+)?(?:e-?\d+)?", value) is None or not 0 <= float(value) <= 1:
        raise ValueError(f"Expected a CI threshold between zero and one: {value!r}")
    return value


def label(value: str) -> str:
    if match := re.fullmatch(
        r"(layer|block|slot|depth|branch|selection|resid|group)[_.](\d+)", value
    ):
        return f"{match[1]}-{int(match[2]):02d}"
    return LABELS.get(value, value.replace("_", "-"))


def named(parts: tuple[str, ...], chart: str) -> MetricKey:
    return MetricKey("-".join(label(part) for part in parts), label(chart))


def site_named(parts: tuple[str, ...], scope: str, chart: str) -> MetricKey:
    if scope not in PROJECTIONS:
        return named((*parts, scope), chart)
    if (layer := re.fullmatch(r"layer_(\d+)", chart)) is None:
        raise ValueError(f"Expected a model layer for projection {scope}: {chart}")
    column = PROJECTIONS.index(scope) + 1
    return named(parts, f"layer-{int(layer[1]):02d}-{column:02d}-{label(scope)}")


def _factor(parts: tuple[str, ...]) -> MetricKey:
    match parts:
        case ("nano", *rest):
            key = _factor(tuple(rest))
            return MetricKey(f"nano-{key.group}", key.chart)
        case ("train", "loss", chart) | ("pretrain", "loss", chart) | ("eval", "loss", chart):
            return named(parts[:-1], chart)
        case ("train", "loss", "nontarget", chart):
            return named(("train", "loss", "nontarget"), chart)
        case (("train" | "pretrain") as phase, "perf", chart):
            return named((phase, "performance"), chart)
        case ("train", "mem", chart):
            return named(("train", "memory"), chart)
        case ("train", "ci_scaled_weight_decay", chart):
            return named(parts[:-1], chart)
        case ("train", "nonlinearity", "loss", chart):
            return named(("train", "nonlinearity", "loss"), chart)
        case ("train", "schedules", chart) | ("pretrain", "schedules", chart):
            return named(parts[:-1], chart)
        case ("train", "schedules", "lr", "src"):
            return MetricKey("train-lr", "ppgd-src")
        case ("train", "schedules", ("lr" | "source_lr" | "loss_coefficient") as measure, chart):
            return named(("train", measure), chart)
        case ("train", "schedules", "nontarget", "loss_coefficient", chart):
            return named(("train", "loss_coefficient", "nontarget"), chart)
        case ("train", "schedules", "auxiliary_coefficient", auxiliary, chart):
            return named(("train", "auxiliary_coefficient", auxiliary), chart)
        case ("train", "reconstruction", ("objective" | "output") as measure, chart):
            return named(("train", "reconstruction", measure), chart)
        case ("train", "reconstruction", "nontarget", "objective", chart):
            return named(("train", "reconstruction", "objective", "nontarget"), chart)
        case ("train", "reconstruction", "auxiliary_mean", auxiliary, chart):
            return named(("train", "reconstruction", auxiliary, "mean"), chart)
        case ("train", "reconstruction", "auxiliary", strategy, auxiliary, chart):
            return named(("train", "reconstruction", auxiliary, strategy), chart)
        case ("train", "grad_norm", ("u" | "v") as factor_name, scope, chart):
            return site_named(("train", "grad_norm", factor_name), scope, chart)
        case ("train", "grad_norm", "ci_fn", "component_readouts", parameter, scope, chart):
            return site_named(
                ("train", "grad_norm", "ci", "component_readouts", parameter), scope, chart
            )
        case (
            "train",
            "grad_norm",
            ("window_min" | "window_max" | "window_median") as statistic,
            chart,
        ):
            return named(
                ("train", "grad_norm", "summary"),
                f"{SUMMARY_ROWS[chart]}-{SUMMARY_COLUMNS[statistic]}",
            )
        case ("train", "grad_norm", "ci_fn", "shared", chart):
            return named(("train", "grad_norm", "ci", "shared"), chart)
        case ("train", "grad_norm", "ci_fn", "blocks", parameter, chart):
            return named(
                ("train", "grad_norm", "ci", "blocks"),
                f"{label(chart)}-{CI_FN_BLOCK_PARAMETERS[parameter]}",
            )
        case (
            "train",
            "grad_norm",
            "ci_fn",
            ("heads" | "selected_heads" | "head_stacks") as scope,
            parameter,
            chart,
        ):
            return named(
                ("train", "grad_norm", "ci", scope), f"{label(chart)}-{WEIGHT_BIAS[parameter]}"
            )
        case ("train", "grad_norm", "ci_fn", "global_mlp", parameter, chart):
            return named(
                ("train", "grad_norm", "ci", "global_mlp"),
                f"{label(chart.replace('layer_', 'depth_'))}-{WEIGHT_BIAS[parameter]}",
            )
        case (
            "train",
            "grad_norm",
            "ci_fn",
            "blocks",
            ("gate" | "norm_scales") as parameter,
            branch,
            chart,
        ):
            column = CI_FN_BLOCK_BRANCHES[parameter][branch]
            return named(("train", "grad_norm", "ci", "blocks"), f"{label(chart)}-{column}")
        case ("train", "grad_norm", "ci_fn", "depth_blocks", ("gate" | "norm_scales") as p, b):
            return named(("train", "grad_norm", "ci", "depth_blocks"), CI_FN_BLOCK_BRANCHES[p][b])
        case ("train", "grad_norm", "ci_fn", "depth_blocks", parameter):
            return named(
                ("train", "grad_norm", "ci", "depth_blocks"), CI_FN_BLOCK_PARAMETERS[parameter]
            )
        case (
            "train",
            "grad_norm",
            "ci_fn",
            "blocks",
            ("selected_gate" | "selected_up" | "selected_down") as parameter,
            block,
            selection,
        ):
            column = {
                "selected_gate": "01-gate",
                "selected_up": "02-up",
                "selected_down": "03-down",
            }[parameter]
            return named(
                ("train", "grad_norm", "ci", "selected_blocks"),
                f"{label(block)}-{label(selection)}-{column}",
            )
        case ("train", "grad_norm", "ci_fn", "site_mlp", parameter, depth, "by_site", chart):
            return named(
                ("train", "grad_norm", "ci", "site_mlp"),
                f"{label(depth)}-site-{label(chart)}-{WEIGHT_BIAS[parameter]}",
            )
        case (
            "eval",
            "reconstruction",
            ("kl" | "ce_difference" | "ce_unrecovered" | "hidden_mse_mean") as measure,
            chart,
        ):
            return named(("eval", "reconstruction", measure), chart)
        case ("eval", "reconstruction", "hidden_mse", strategy, projection, chart):
            return site_named(("eval", "reconstruction", "hidden_mse", strategy), projection, chart)
        case ("eval", "reconstruction", "pgd", chart):
            return named(("eval", "reconstruction", "pgd"), chart)
        case (
            "eval",
            "reconstruction",
            "pgd",
            ("objective" | "output" | "hidden_acts_mean") as measure,
            chart,
        ):
            return named(("eval", "reconstruction", "pgd", measure), chart)
        case ("eval", "reconstruction", "pgd", "hidden_acts", probe, chart):
            return named(("eval", "reconstruction", "pgd", "hidden_acts", probe), chart)
        case ("eval", "reconstruction", "pgd", "auxiliary", auxiliary, probe, chart):
            return named(("eval", "reconstruction", "pgd", auxiliary, probe), chart)
        case ("eval", "reconstruction", "pgd", "auxiliary_mean", auxiliary, chart):
            return named(("eval", "reconstruction", "pgd", auxiliary, "mean"), chart)
        case (
            "eval",
            "nonlinearity",
            (
                "effective_use_count_per_subcomponent"
                | "soft_use_count_relative_threshold_4"
                | "n_components"
            ) as measure,
            ("all" | "mean_ci_gt_0") as population,
            scope,
            chart,
        ):
            return site_named(("eval", "nonlinearity", measure, population), scope, chart)
        case ("eval", "attention_pattern", "kl", "all_layers", chart):
            return named(("eval", "attention_pattern", "kl", "all_layers"), chart)
        case ("eval", "attention_pattern", "kl", strategy, "self_attn.q_proj", chart):
            return named(("eval", "attention_pattern", "kl", strategy), chart)
        case ("eval", "router", ("kl" | "weight_mae" | "topk_overlap") as measure, scope, chart):
            return named(("eval", "router", measure, scope), chart)
        case ("eval", "uv_norm_ratio", scope, chart):
            return site_named(("eval", "uv_norm_ratio"), scope, chart)
        case ("eval", "ci", "l0", "charts", chart):
            return named(("eval", "ci", "figures"), f"l0-{label(chart)}")
        case (
            "eval",
            "ci",
            (
                "activation_density"
                | "component_mean"
                | "density"
                | "permuted_heatmap"
                | "value_histogram"
                | "mean_per_component"
            ) as measure,
            chart,
        ):
            return named(("eval", "ci", "figures"), f"{label(measure)}-{label(chart)}")
        case ("eval", "ci", "l0", threshold, scope, chart):
            return site_named(("eval", "ci", "l0", threshold), scope, chart)
        case (
            "eval",
            "ci",
            ("recovery_error" | "identity_ci_error" | "dense_ci_error") as measure,
            parameterization,
            scope,
            chart,
        ):
            return site_named(("eval", "ci", measure, parameterization), scope, chart)
        case ("eval", "components", "matrices", chart):
            return named(("eval", "component_figures"), chart)
        case ("eval", "well_temperedness", "fraction_well_ordered_pairs", scope, chart):
            return named(("eval", "well_temperedness", "ordered_pair_fraction", scope), chart)
        case ("eval", "well_temperedness", "figures", chart):
            return named(parts[:-1], chart)
        case ("eval", "active_component_count", threshold, "site_sum", "total"):
            return named(("eval", "active_component_count", "site_sum"), threshold)
        case (
            "eval",
            ("active_component_count" | "unplotted_active_component_count") as measure,
            threshold,
            projection,
            chart,
        ):
            return site_named(("eval", measure, threshold), projection, chart)
        case (
            "eval",
            "figures",
            ("activation_grid" | "ci_grid") as measure,
            threshold,
            projection,
            chart,
        ):
            return site_named(("eval", measure, threshold), projection, chart)
        case ("progress", chart):
            return named(("progress",), chart)
        case _:
            raise ValueError(f"Unclassified metric family: {parts}")


_LOSS_NAMES = {
    "FaithfulnessLoss": "faithfulness",
    "ImportanceMinimalityLoss": "importance_minimality",
    "FrequencyMinimalityLoss": "frequency_minimality",
    "FrequencyMinimalityLoss_batch": "frequency_minimality_batch",
    "NonlinearityLocalityLoss": "nonlinearity_locality",
    "CIMaskedReconLoss": "ci_masked",
    "CIMaskedReconSubsetLoss": "ci_masked_subset",
    "StochasticReconLoss": "stochastic",
    "StochasticReconSubsetLoss": "stochastic_subset",
    "UnmaskedReconLoss": "unmasked",
    "UnmaskedNoDeltaReconLoss": "unmasked_no_delta",
    "PGDReconLoss": "pgd",
    "PGDReconSubsetLoss": "pgd_subset",
    "PersistentPGDReconLoss": "persistent_pgd",
    "MergedStochasticSubsetPPGDReconLoss": "merged_stochastic_subset_ppgd",
    "MergedStochasticSubsetPooledPPGDReconLoss": "merged_stochastic_subset_pooled_ppgd",
    "recon_primary": "recon_primary",
    "recon_secondary": "recon_secondary",
}
_REGULARIZATION_LOSSES = {
    "FaithfulnessLoss",
    "ImportanceMinimalityLoss",
    "FrequencyMinimalityLoss",
    "FrequencyMinimalityLoss_batch",
    "NonlinearityLocalityLoss",
    "total",
}
_FIGURES = {
    "causal_importance_values": ("ci", "value_histogram", "lower_leaky"),
    "causal_importance_values_pre_sigmoid": ("ci", "value_histogram", "pre_sigmoid"),
    "component_activation_density": ("ci", "activation_density", "per_component"),
    "component_activation_density_groups": ("ci", "activation_density", "by_group"),
    "ci_mean_per_component": ("ci", "component_mean", "linear"),
    "ci_mean_per_component_log": ("ci", "component_mean", "log"),
    "ci_mean_per_component_groups": ("ci", "component_mean", "by_group"),
    "ci_density_heatmap": ("ci", "density", "heatmap"),
    "causal_importances": ("ci", "permuted_heatmap", "lower_leaky"),
    "causal_importances_upper_leaky": ("ci", "permuted_heatmap", "upper_leaky"),
    "uv_matrices": ("components", "matrices", "uv"),
}


def _loss_name(value: str) -> str:
    # Reconstruction terms and auxiliaries can be named by the run configuration.
    return _LOSS_NAMES[value] if value in _LOSS_NAMES else _authored_label(value)


def _reconstruction_name(value: str) -> str:
    if value in _REGULARIZATION_LOSSES or value.startswith("NonlinearityLocalityLoss_"):
        raise ValueError(f"Expected a reconstruction term, got scalar loss: {value}")
    return _loss_name(value)


def _site_parts(site: str) -> tuple[str, str]:
    if match := re.fullmatch(r"layers\.(\d+)\.(.+)", site):
        projection = match[2]
        if projection in PROJECTIONS or re.fullmatch(
            r"mlp_(?:in|out)|linear_attn\.(?:in_proj_qkv\.[qkv]|in_proj_[zba]|out_proj)"
            r"|mlp\.(?:experts|shared_expert)\.(?:gate|up|down)_proj",
            projection,
        ):
            return projection, f"layer_{int(match[1]):02d}"
    if match := re.fullmatch(r"h\.(\d+)\.(attn\.[qkvo]_proj|mlp\.(?:c_fc|down_proj))", site):
        return match[2], f"layer_{int(match[1]):02d}"
    if re.fullmatch(r"linear[12]|hidden_layers\.\d+", site):
        return "by_site", site
    raise ValueError(f"Unclassified model site: {site}")


def _gradient_parts(leaf: str) -> tuple[str, ...]:
    if match := re.fullmatch(r"components\.vu\['(.+)'\]\[([01])\]", leaf):
        return {"0": "v", "1": "u"}[match[2]], *_site_parts(match[1])
    if match := re.fullmatch(r"ci_fns\.(?:backbone\.)?chunks\.blocks\[(\d+)\]\.(\w+)", leaf):
        return "ci_fn", "blocks", match[2], f"block_{int(match[1]):02d}"
    if match := re.fullmatch(
        r"ci_fns\.(?:backbone\.)?chunks\.blocks\[(\d+)\]\.(gate|norm_scales)\[([01])\]", leaf
    ):
        return (
            "ci_fn",
            "blocks",
            match[2],
            f"branch_{int(match[3]):02d}",
            f"block_{int(match[1]):02d}",
        )
    if match := re.fullmatch(
        r"ci_fns\.chunks\.blocks\[(\d+)\]\.(selected_gate|selected_up|selected_down)\[(\d+)\]", leaf
    ):
        return (
            "ci_fn",
            "blocks",
            match[2],
            f"block_{int(match[1]):02d}",
            f"selection_{int(match[3]):02d}",
        )
    if match := re.fullmatch(r"ci_fns\.(?:backbone\.)?chunks\.(out_bs|out_ws)\[(\d+)\]", leaf):
        parameter = {"out_bs": "bias", "out_ws": "weight"}[match[1]]
        return "ci_fn", "heads", parameter, f"slot_{int(match[2]):02d}"
    if match := re.fullmatch(r"ci_fns\.chunks\.heads\[(\d+)\]\.(w|b)", leaf):
        return (
            "ci_fn",
            "selected_heads",
            {"w": "weight", "b": "bias"}[match[2]],
            f"slot_{int(match[1]):02d}",
        )
    if leaf in CI_FN_SHARED_LEAVES:
        return "ci_fn", "shared", leaf.rsplit(".", 1)[-1]
    # The global transformer stacks its blocks along depth and its heads by semantic group.
    global_backbone = r"ci_fns\.backbone\.(?:backbone\.)?"
    if match := re.fullmatch(global_backbone + r"blocks\.(gate|norm_scales)\[([01])\]", leaf):
        return "ci_fn", "depth_blocks", match[1], f"branch_{int(match[2]):02d}"
    if match := re.fullmatch(global_backbone + r"blocks\.(\w+)", leaf):
        return "ci_fn", "depth_blocks", match[1]
    if match := re.fullmatch(global_backbone + r"heads\[(\d+)\]\.(weights|biases)", leaf):
        parameter = {"weights": "weight", "biases": "bias"}[match[2]]
        return "ci_fn", "head_stacks", parameter, f"group_{int(match[1]):02d}"
    if match := re.fullmatch(
        r"ci_fns\.backbone\.readouts\['(.+)'\]"
        r"\.((?:positive|negative)\.(?:input_scale|input_bias|output_scale)|bias)",
        leaf,
    ):
        parameter = match[2].replace(".", "_")
        return "ci_fn", "component_readouts", parameter, *_site_parts(match[1])
    if match := re.fullmatch(r"ci_fns\.mlp\.(weights|biases)\[(\d+)\]", leaf):
        return (
            "ci_fn",
            "global_mlp",
            {"weights": "weight", "biases": "bias"}[match[1]],
            f"layer_{int(match[2]):02d}",
        )
    if match := re.fullmatch(r"ci_fns\.site_mlps\['(.+)'\]\.(weights|biases)\[(\d+)\]", leaf):
        _site_parts(match[1])
        return (
            "ci_fn",
            "site_mlp",
            {"weights": "weight", "biases": "bias"}[match[2]],
            f"depth_{int(match[3]):02d}",
            "by_site",
            match[1],
        )
    raise ValueError(f"Unclassified gradient parameter: {leaf}")


def _eval_loss_parts(probe: str, detail: tuple[str, ...]) -> tuple[str, ...]:
    if probe in ("CIMaskedAttnPatternsReconLoss", "StochasticAttnPatternsReconLoss"):
        strategy = {
            "CIMaskedAttnPatternsReconLoss": "ci_masked",
            "StochasticAttnPatternsReconLoss": "stochastic",
        }[probe]
        match detail:
            case ():
                return "attention_pattern", "kl", "all_layers", strategy
            case (site,):
                return "attention_pattern", "kl", strategy, *_site_parts(site)
            case _:
                raise ValueError(f"Unknown attention readout: {probe}/{detail}")
    if probe in ("StochasticHiddenActsReconLoss", "CIHiddenActsReconLoss"):
        strategy = {
            "StochasticHiddenActsReconLoss": "stochastic",
            "CIHiddenActsReconLoss": "ci_masked",
        }[probe]
        match detail:
            case ("total",):
                return "reconstruction", "hidden_mse_mean", strategy
            case (site,):
                return "reconstruction", "hidden_mse", strategy, *_site_parts(site)
            case _:
                raise ValueError(f"Unknown hidden activation readout: {probe}/{detail}")
    if probe in _REGULARIZATION_LOSSES or probe == "StochasticReconSubsetLoss":
        if detail:
            raise ValueError(f"Unexpected loss detail: {probe}/{detail}")
        return "loss", _loss_name(probe)
    if probe == "PGDReconLoss":
        name = "default"
    elif match := re.fullmatch(r"PGDReconLoss_(\d+)step", probe):
        name = f"steps_{int(match[1]):04d}"
    else:
        raise ValueError(f"Unclassified evaluation probe: {probe}")
    match detail:
        case ():
            return "reconstruction", "pgd", "objective", name
        case ("e2e",):
            return "reconstruction", "pgd", "output", name
        case ("hidden_acts_reconstruction",):
            return "reconstruction", "pgd", "hidden_acts_mean", name
        case ("hidden_acts_reconstruction", capture):
            return "reconstruction", "pgd", "hidden_acts", name, _authored_label(capture)
        case (auxiliary,):
            return "reconstruction", "pgd", "auxiliary_mean", _authored_label(auxiliary), name
        case (auxiliary, capture):
            return (
                "reconstruction",
                "pgd",
                "auxiliary",
                _authored_label(auxiliary),
                name,
                _authored_label(capture),
            )
        case _:
            raise ValueError(f"Unknown evaluation readout: {probe}/{detail}")


def _legacy_parts(key: str) -> tuple[str, ...]:
    match key.split("/"):
        case ["train", "loss", "nontarget", term]:
            if term in ("imp", "freq", "freq_batch", "total"):
                return (
                    "train",
                    "loss",
                    "nontarget",
                    {
                        "imp": "importance_minimality",
                        "freq": "frequency_minimality",
                        "freq_batch": "frequency_minimality_batch",
                        "total": "total",
                    }[term],
                )
            return "train", "reconstruction", "nontarget", "objective", _reconstruction_name(term)
        case ["train", "loss", term] if term in _REGULARIZATION_LOSSES:
            return "train", "loss", _loss_name(term)
        case ["train", "loss", term] if term.startswith("NonlinearityLocalityLoss_"):
            kind = term.removeprefix("NonlinearityLocalityLoss_")
            if kind not in _UNIT_KINDS:
                raise ValueError(f"Unclassified nonlinearity unit: {kind}")
            return "train", "nonlinearity", "loss", kind
        case ["train", "loss", term] if term in _LOSS_NAMES:
            return "train", "reconstruction", "objective", _loss_name(term)
        case ["train", "loss", term]:
            return "train", "loss", _authored_label(term)
        case ["train", "loss", term, "e2e"]:
            return "train", "reconstruction", "output", _reconstruction_name(term)
        case ["train", "loss", term, auxiliary]:
            return (
                "train",
                "reconstruction",
                "auxiliary_mean",
                _authored_label(auxiliary),
                _reconstruction_name(term),
            )
        case ["train", "loss", term, auxiliary, capture]:
            return (
                "train",
                "reconstruction",
                "auxiliary",
                _reconstruction_name(term),
                _authored_label(auxiliary),
                _authored_label(capture),
            )
        case ["train", "schedules", "coeff", "ImportanceMinimalityLoss", "frequency"]:
            return "train", "schedules", "loss_coefficient", "frequency_minimality"
        case ["train", "schedules", "coeff", "nontarget", "impmin"]:
            return "train", "schedules", "nontarget", "loss_coefficient", "importance_minimality"
        case ["train", "schedules", "coeff", "nontarget", term]:
            return "train", "schedules", "nontarget", "loss_coefficient", _reconstruction_name(term)
        case ["train", "schedules", "coeff", term]:
            return "train", "schedules", "loss_coefficient", _loss_name(term)
        case ["train", "schedules", "coeff", term, auxiliary]:
            return (
                "train",
                "schedules",
                "auxiliary_coefficient",
                _authored_label(auxiliary),
                _reconstruction_name(term),
            )
        case ["train", "schedules", "lr", "src", term]:
            return "train", "schedules", "source_lr", _reconstruction_name(term)
        case ["train", "schedules", "lr", "src" | "components" | "ci_fn"]:
            return tuple(key.split("/"))
        case ["train", "schedules", "gamma_imp" | "nonlinearity_relative_threshold"]:
            return tuple(key.split("/"))
        case ["train", "perf", metric] if metric in _PERFORMANCE:
            return "train", "perf", metric
        case ["train", "mem", "peak_gb_per_rank" | "peak_gb_min_device"]:
            return tuple(key.split("/"))
        case ["ci_scaled_weight_decay", "mean" | "max"]:
            return "train", *key.split("/")
        case ["train", "grad_norms", "summary", group, statistic]:
            return "train", "grad_norm", f"window_{statistic}", group
        case ["train", "grad_norms", leaf]:
            return "train", "grad_norm", *_gradient_parts(leaf)
        case ["eval", "nonlinearity", "sites", site, population, measure]:
            return "eval", "nonlinearity", measure, population, *_site_parts(site)
        case ["eval", "nonlinearity", "aggregates", kind, population, measure] if (
            kind in _UNIT_KINDS
        ):
            return "eval", "nonlinearity", measure, population, "by_unit_kind", kind
        case ["eval", "l0", "bar_chart"]:
            return "eval", "ci", "l0", "charts", "site_and_group_counts"
        case ["eval", "l0", "per_position"]:
            return "eval", "ci", "l0", "charts", "per_position"
        case ["eval", "l0", leaf]:
            threshold, separator, category = leaf.partition("_")
            if not separator:
                raise ValueError(f"Unknown CI threshold: {key}")
            is_site = category.startswith(("layers.", "h.", "hidden_layers.")) or re.fullmatch(
                r"linear\d+", category
            )
            scope = _site_parts(category) if is_site else ("group_sum", _authored_label(category))
            return "eval", "ci", "l0", f"threshold_{_threshold(threshold)}", *scope
        case ["eval", "ce_kl", leaf]:
            for measure in ("ce_difference", "ce_unrecovered", "kl"):
                if leaf.startswith(measure + "_"):
                    variant = leaf.removeprefix(measure + "_")
                    if variant not in _CE_VARIANTS:
                        raise ValueError(f"Unclassified reconstruction variant: {key}")
                    return "eval", "reconstruction", measure, variant
            raise ValueError(f"Unknown reconstruction measure: {key}")
        case ["eval", "loss", probe, *detail]:
            return "eval", *_eval_loss_parts(probe, tuple(detail))
        case ["eval", leaf] if leaf.startswith("uv_norm_ratio"):
            if match := re.fullmatch(r"uv_norm_ratio\['(.+)'\]", leaf):
                return "eval", "uv_norm_ratio", *_site_parts(match[1])
            if leaf in ("uv_norm_ratio_mean", "uv_norm_ratio_max"):
                return (
                    "eval",
                    "uv_norm_ratio",
                    "site_reduction",
                    leaf.removeprefix("uv_norm_ratio_"),
                )
            raise ValueError(f"Unknown norm ratio: {key}")
        case ["eval", "router_divergence", strategy, measure] if strategy in _ROUTER_STRATEGIES:
            return "eval", "router", measure, "layer_mean", strategy
        case ["eval", "router_divergence", strategy, measure, layer] if (
            strategy in _ROUTER_STRATEGIES and re.fullmatch(r"layer_\d+", layer)
        ):
            return "eval", "router", measure, strategy, layer
        case [
            "eval",
            "slow",
            "well_temperedness",
            "figures",
            "preactivation_vs_ablation_damage" as chart,
        ]:
            return "eval", "well_temperedness", "figures", chart
        case ["eval", "slow", "well_temperedness", scope, chart] if chart in {
            "fraction_well_ordered_pairs_below_zero",
            "fraction_well_ordered_pairs_zero_to_one",
            "fraction_well_ordered_pairs_above_one",
        }:
            return (
                "eval",
                "well_temperedness",
                "fraction_well_ordered_pairs",
                _authored_label(scope),
                chart.removeprefix("fraction_well_ordered_pairs_"),
            )
        case ["slow_eval", "figures", chart] if chart in _FIGURES:
            return "eval", *_FIGURES[chart]
        case ["eval", "slow", "IdentityCIError", site]:
            return "eval", "ci", "recovery_error", "upper_leaky", *_site_parts(site)
        case ["eval", "slow", "IdentityCIError"]:
            return "eval", "ci", "recovery_error", "upper_leaky", "site_sum", "total"
        case ["eval", "figures", ("ci_mean_per_component" | "ci_mean_per_component_log") as chart]:
            return (
                "eval",
                "ci",
                "mean_per_component",
                {"ci_mean_per_component": "linear", "ci_mean_per_component_log": "log"}[chart],
            )
        case ["eval", ("identity_ci_error" | "dense_ci_error") as measure, site]:
            return "eval", "ci", measure, "lower_leaky", *_site_parts(site)
        case ["eval", ("n_alive" | "n_dropped") as measure, threshold, site] if (
            threshold.startswith("thr")
        ):
            scope = ("site_sum", "total") if site == "total" else _site_parts(site)
            return (
                "eval",
                {
                    "n_alive": "active_component_count",
                    "n_dropped": "unplotted_active_component_count",
                }[measure],
                f"threshold_{_threshold(threshold.removeprefix('thr'))}",
                *scope,
            )
        case ["eval", "figures", ("activation_grid" | "ci_grid") as measure, threshold, site] if (
            threshold.startswith("thr")
        ):
            return (
                "eval",
                "figures",
                measure,
                f"threshold_{_threshold(threshold.removeprefix('thr'))}",
                *_site_parts(site),
            )
        case ["train_loss"]:
            return "pretrain", "loss", "train"
        case ["val_loss"]:
            return "pretrain", "loss", "validation"
        case ["lr"]:
            return "pretrain", "schedules", "lr"
        case ["step_time_s"]:
            return "pretrain", "perf", "step_time_s"
        case ["tok_per_s"]:
            return "pretrain", "perf", "tokens_per_s"
        case ["loss", ("faith" | "imp" | "stoch" | "ppgd") as term]:
            return (
                "nano",
                "train",
                "loss",
                {
                    "faith": "faithfulness",
                    "imp": "importance_minimality",
                    "stoch": "stochastic",
                    "ppgd": "persistent_pgd",
                }[term],
            )
        case ["lr", ("main" | "ppgd") as parameter]:
            return "nano", "train", "schedules", "lr", {"main": "main", "ppgd": "source"}[parameter]
        case ["step"]:
            return "progress", "step"
        case _:
            raise ValueError(f"Unclassified metric: {key}")


def grouped_metric_key(key: str) -> MetricKey:
    """Map an emitted legacy metric; ambiguous experiment context is never inferred."""
    if key.startswith("eval/arithmetic/"):
        inner = _factor(_legacy_parts("eval/" + key.removeprefix("eval/arithmetic/")))
        return MetricKey("eval-arithmetic-" + inner.group.removeprefix("eval-"), inner.chart)
    try:
        return _factor(_legacy_parts(key))
    except KeyError as error:
        raise ValueError(f"Unclassified metric category in {key}: {error}") from error


def grouped_axis(key: str) -> str:
    match key:
        case "_step":
            return key
        case "slow_eval/figure_step":
            return "axes/slow-figures-step"
        case "eval/arithmetic/figure_step":
            return "axes/arithmetic-figures-step"
        case "eval/slow/well_temperedness/figure_step":
            return "axes/well-temperedness-figures-step"
        case _:
            raise ValueError(f"Unclassified metric axis: {key}")
