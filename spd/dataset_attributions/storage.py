"""Storage classes for dataset attributions.

Stores raw (unnormalized) attribution sums. Normalization happens at query time using
stored metadata (CI sums, activation RMS, logit RMS).

Four edge types, each with its own shape:
- regular:        component → component  [tgt_c, src_c]  (signed + abs)
- embed:          embed → component      [tgt_c, vocab]  (signed + abs)
- unembed:        component → unembed    [d_model, src_c] (signed only, residual space)
- embed_unembed:  embed → unembed        [d_model, vocab] (signed only, residual space)

Abs variants are unavailable for unembed edges because abs is a nonlinear operation
incompatible with the residual-space storage trick.

Normalization formula:
    normed[t, s] = raw[t, s] / source_denom[s] / target_rms[t]
- source_denom is ci_sum[s] for component sources, embed_token_count[s] for embed sources
- target_rms is component activation RMS for component targets, logit RMS for output targets
"""

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from spd.log import logger

AttrMetric = Literal["attr", "attr_abs"]

EPS = 1e-10


@dataclass
class DatasetAttributionEntry:
    """A single entry in the attribution results (component + value)."""

    component_key: str
    layer: str
    component_idx: int
    value: float


class DatasetAttributionStorage:
    """Dataset-aggregated attribution strengths between components.

    All layer names use canonical addressing (e.g., "embed", "0.glu.up", "output").

    Internally stores raw sums — normalization applied at query time.
    Public interface: get_top_sources(), get_top_targets(), save/load/merge.

    Key formats:
        - embed tokens: "embed:{token_id}"
        - component layers: "canonical_layer:c_idx" (e.g., "0.glu.up:5")
        - output tokens: "output:{token_id}"
    """

    def __init__(
        self,
        regular_attr: dict[str, dict[str, Tensor]],
        regular_attr_abs: dict[str, dict[str, Tensor]],
        embed_attr: dict[str, Tensor],
        embed_attr_abs: dict[str, Tensor],
        unembed_attr: dict[str, Tensor],
        embed_unembed_attr: Tensor,
        w_unembed: Tensor,
        ci_sum: dict[str, Tensor],
        component_act_sq_sum: dict[str, Tensor],
        logit_sq_sum: Tensor,
        embed_token_count: Tensor,
        ci_threshold: float,
        n_tokens_processed: int,
    ):
        self._regular_attr = regular_attr
        self._regular_attr_abs = regular_attr_abs
        self._embed_attr = embed_attr
        self._embed_attr_abs = embed_attr_abs
        self._unembed_attr = unembed_attr
        self._embed_unembed_attr = embed_unembed_attr
        self._w_unembed = w_unembed
        self._ci_sum = ci_sum
        self._component_act_sq_sum = component_act_sq_sum
        self._logit_sq_sum = logit_sq_sum
        self._embed_token_count = embed_token_count
        self.ci_threshold = ci_threshold
        self.n_tokens_processed = n_tokens_processed

    @property
    def target_layers(self) -> set[str]:
        return self._regular_attr.keys() | self._embed_attr.keys()

    def _target_n_components(self, layer: str) -> int | None:
        if layer in self._embed_attr:
            return self._embed_attr[layer].shape[0]
        if layer in self._regular_attr:
            first_source = next(iter(self._regular_attr[layer].values()))
            return first_source.shape[0]
        return None

    @property
    def n_components(self) -> int:
        total = 0
        for layer in self.target_layers:
            n = self._target_n_components(layer)
            assert n is not None
            total += n
        return total

    @staticmethod
    def _parse_key(key: str) -> tuple[str, int]:
        layer, idx_str = key.rsplit(":", 1)
        return layer, int(idx_str)

    def _select_metric(
        self, metric: AttrMetric
    ) -> tuple[dict[str, dict[str, Tensor]], dict[str, Tensor]]:
        match metric:
            case "attr":
                return self._regular_attr, self._embed_attr
            case "attr_abs":
                return self._regular_attr_abs, self._embed_attr_abs

    def _component_activation_rms(self, layer: str) -> Tensor:
        """RMS activation for a component layer. Shape (n_components,)."""
        return (self._component_act_sq_sum[layer] / self.n_tokens_processed).sqrt().clamp(min=EPS)

    def _logit_activation_rms(self) -> Tensor:
        """RMS logit per token. Shape (vocab,)."""
        return (self._logit_sq_sum / self.n_tokens_processed).sqrt().clamp(min=EPS)

    def _layer_ci_sum(self, layer: str) -> Tensor:
        """CI sum for a source layer, clamped. Shape (n_components,)."""
        return self._ci_sum[layer].clamp(min=EPS)

    def _embed_count(self) -> Tensor:
        """Per-token occurrence count, clamped. Shape (vocab,)."""
        return self._embed_token_count.float().clamp(min=EPS)

    def get_top_sources(
        self,
        target_key: str,
        k: int,
        sign: Literal["positive", "negative"],
        metric: AttrMetric,
    ) -> list[DatasetAttributionEntry]:
        target_layer, target_idx = self._parse_key(target_key)

        value_segments: list[Tensor] = []
        layer_names: list[str] = []
        if target_layer == "embed":
            return []

        if target_layer == "output":
            if metric == "attr_abs":
                return []
            w = self._w_unembed[:, target_idx].to(self._embed_unembed_attr.device)
            target_act_rms = self._logit_activation_rms()[target_idx]

            for source_layer, attr_matrix in self._unembed_attr.items():
                raw = w @ attr_matrix  # (src_c,)
                value_segments.append(raw / self._layer_ci_sum(source_layer) / target_act_rms)
                layer_names.append(source_layer)

            raw = w @ self._embed_unembed_attr  # (vocab,)
            value_segments.append(raw / self._embed_count() / target_act_rms)
            layer_names.append("embed")
        else:
            regular_attr, embed_target_attr = self._select_metric(metric)
            target_act_rms = self._component_activation_rms(target_layer)[target_idx]

            if target_layer in regular_attr:
                for source_layer, attr_matrix in regular_attr[target_layer].items():
                    raw = attr_matrix[target_idx, :]  # (src_c,)
                    value_segments.append(raw / self._layer_ci_sum(source_layer) / target_act_rms)
                    layer_names.append(source_layer)

            if target_layer in embed_target_attr:
                raw = embed_target_attr[target_layer][target_idx, :]  # (vocab,)
                value_segments.append(raw / self._embed_count() / target_act_rms)
                layer_names.append("embed")

        return self._top_k_from_segments(value_segments, layer_names, k, sign)

    def get_top_targets(
        self,
        source_key: str,
        k: int,
        sign: Literal["positive", "negative"],
        metric: AttrMetric,
        include_outputs: bool = True,
    ) -> list[DatasetAttributionEntry]:
        source_layer, source_idx = self._parse_key(source_key)

        value_segments: list[Tensor] = []
        layer_names: list[str] = []

        if source_layer == "output":
            return []
        elif source_layer == "embed":
            regular, embed = self._select_metric(metric)
            embed_count = self._embed_count()[source_idx]

            for target_layer, attr_matrix in embed.items():
                raw = attr_matrix[:, source_idx]  # (tgt_c,)
                value_segments.append(
                    raw / embed_count / self._component_activation_rms(target_layer)
                )
                layer_names.append(target_layer)

            if include_outputs and metric == "attr":
                residual = self._embed_unembed_attr[:, source_idx]  # (d_model,)
                raw = residual @ self._w_unembed  # (vocab,)
                value_segments.append(raw / embed_count / self._logit_activation_rms())
                layer_names.append("output")
        else:
            regular, embed = self._select_metric(metric)
            ci = self._layer_ci_sum(source_layer)[source_idx]

            for target_layer, sources in regular.items():
                if source_layer not in sources:
                    continue
                raw = sources[source_layer][:, source_idx]  # (tgt_c,)
                value_segments.append(raw / ci / self._component_activation_rms(target_layer))
                layer_names.append(target_layer)

            if include_outputs and metric == "attr" and source_layer in self._unembed_attr:
                residual = self._unembed_attr[source_layer][:, source_idx]  # (d_model,)
                raw = residual @ self._w_unembed  # (vocab,)
                value_segments.append(raw / ci / self._logit_activation_rms())
                layer_names.append("output")

        return self._top_k_from_segments(value_segments, layer_names, k, sign)

    def _top_k_from_segments(
        self,
        value_segments: list[Tensor],
        layer_names: list[str],
        k: int,
        sign: Literal["positive", "negative"],
    ) -> list[DatasetAttributionEntry]:
        if not value_segments:
            return []

        all_values = torch.cat(value_segments)
        offsets = [0]
        for seg in value_segments:
            offsets.append(offsets[-1] + len(seg))

        is_positive = sign == "positive"
        top_vals, top_idxs = torch.topk(all_values, min(k, len(all_values)), largest=is_positive)

        mask = top_vals > 0 if is_positive else top_vals < 0
        top_vals, top_idxs = top_vals[mask], top_idxs[mask]

        results = []
        for flat_idx, val in zip(top_idxs.tolist(), top_vals.tolist(), strict=True):
            seg_idx = bisect.bisect_right(offsets, flat_idx) - 1
            local_idx = flat_idx - offsets[seg_idx]
            layer = layer_names[seg_idx]
            results.append(
                DatasetAttributionEntry(
                    component_key=f"{layer}:{local_idx}",
                    layer=layer,
                    component_idx=local_idx,
                    value=val,
                )
            )
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "regular_attr": _to_cpu_nested(self._regular_attr),
                "regular_attr_abs": _to_cpu_nested(self._regular_attr_abs),
                "embed_attr": _to_cpu(self._embed_attr),
                "embed_attr_abs": _to_cpu(self._embed_attr_abs),
                "unembed_attr": _to_cpu(self._unembed_attr),
                "embed_unembed_attr": self._embed_unembed_attr.detach().cpu(),
                "w_unembed": self._w_unembed.detach().cpu(),
                "ci_sum": _to_cpu(self._ci_sum),
                "component_act_sq_sum": _to_cpu(self._component_act_sq_sum),
                "logit_sq_sum": self._logit_sq_sum.detach().cpu(),
                "embed_token_count": self._embed_token_count.detach().cpu(),
                "ci_threshold": self.ci_threshold,
                "n_tokens_processed": self.n_tokens_processed,
            },
            path,
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved dataset attributions to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: Path) -> "DatasetAttributionStorage":
        data = torch.load(path, weights_only=True)
        return cls(
            regular_attr=data["regular_attr"],
            regular_attr_abs=data["regular_attr_abs"],
            embed_attr=data["embed_attr"],
            embed_attr_abs=data["embed_attr_abs"],
            unembed_attr=data["unembed_attr"],
            embed_unembed_attr=data["embed_unembed_attr"],
            w_unembed=data["w_unembed"],
            ci_sum=data["ci_sum"],
            component_act_sq_sum=data["component_act_sq_sum"],
            logit_sq_sum=data["logit_sq_sum"],
            embed_token_count=data["embed_token_count"],
            ci_threshold=data["ci_threshold"],
            n_tokens_processed=data["n_tokens_processed"],
        )

    @classmethod
    def merge(cls, paths: list[Path]) -> "DatasetAttributionStorage":
        """Merge partial attribution files from parallel workers.

        All stored values are raw sums — merge is element-wise addition.
        """
        assert paths, "No files to merge"

        merged = cls.load(paths[0])

        for path in paths[1:]:
            other = cls.load(path)
            assert other.ci_threshold == merged.ci_threshold, "CI threshold mismatch"

            for target, sources in other._regular_attr.items():
                for source, tensor in sources.items():
                    merged._regular_attr[target][source] += tensor
                    merged._regular_attr_abs[target][source] += other._regular_attr_abs[target][
                        source
                    ]

            for target, tensor in other._embed_attr.items():
                merged._embed_attr[target] += tensor
                merged._embed_attr_abs[target] += other._embed_attr_abs[target]

            for source, tensor in other._unembed_attr.items():
                merged._unembed_attr[source] += tensor

            merged._embed_unembed_attr += other._embed_unembed_attr

            for layer in other._ci_sum:
                merged._ci_sum[layer] += other._ci_sum[layer]

            for layer in other._component_act_sq_sum:
                merged._component_act_sq_sum[layer] += other._component_act_sq_sum[layer]

            merged._logit_sq_sum += other._logit_sq_sum
            merged._embed_token_count += other._embed_token_count
            merged.n_tokens_processed += other.n_tokens_processed

        return merged


def _to_cpu_nested(d: dict[str, dict[str, Tensor]]) -> dict[str, dict[str, Tensor]]:
    return {
        target: {source: v.detach().cpu() for source, v in sources.items()}
        for target, sources in d.items()
    }


def _to_cpu(d: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: v.detach().cpu() for k, v in d.items()}
