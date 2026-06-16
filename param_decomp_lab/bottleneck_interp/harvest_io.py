"""Shared loaders and helpers for the bottleneck-interp tooling.

Consolidates what the interpret / viewer drivers all need from a context-preserving
harvest (`harvest_codes.py` output): the codes, the token sequences, the tokenizer, and
the helpers for region labelling and token thumbnails.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Int
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import MiniBatchKMeans
from torch import Tensor
from transformers import AutoTokenizer

from param_decomp_lab.bottleneck_interp.geometry import load_codes

# Distinct RGB triples for colouring regions / groups in the viewers.
REGION_PALETTE: list[list[float]] = [
    [0.12, 0.47, 0.71],
    [1.00, 0.50, 0.05],
    [0.17, 0.63, 0.17],
    [0.84, 0.15, 0.16],
    [0.58, 0.40, 0.74],
    [0.55, 0.34, 0.29],
    [0.89, 0.47, 0.76],
    [0.50, 0.50, 0.50],
    [0.74, 0.74, 0.13],
    [0.09, 0.75, 0.81],
    [0.68, 0.78, 0.91],
    [1.00, 0.73, 0.47],
    [0.60, 0.87, 0.54],
    [1.00, 0.60, 0.59],
    [0.77, 0.69, 0.84],
    [0.77, 0.61, 0.58],
    [0.97, 0.71, 0.82],
    [0.78, 0.78, 0.78],
    [0.86, 0.86, 0.55],
    [0.62, 0.85, 0.90],
]


@dataclass
class Harvest:
    """A loaded context-preserving harvest, with `flat_tokens` aligned to `codes`.

    `module_frac` is `[n, M]` float in [0, 1] (per position, the fraction of each module's
    components that are CI-active), present only if the harvest stored it.
    """

    meta: dict[str, Any]
    seq_len: int
    tokenizer: Any
    sequences: Int[Tensor, "n_seq seq_len"]
    codes: Float[Tensor, "n D"]
    flat_tokens: Int[Tensor, " n"]
    module_names: list[str] | None
    module_frac: Float[Tensor, "n M"] | None
    # Active-component COO (present only with a --components harvest): `active_components`
    # is {"point": [nnz], "comp": [nnz]} flat indices into positions / global component
    # ids; `component_names` maps a global component id to "module#local".
    component_names: list[str] | None
    active_components: dict[str, Tensor] | None


def load_harvest(code_dir: Path, max_positions: int) -> Harvest:
    meta = json.loads((code_dir / "meta.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(meta["tokenizer_name"])
    sequences = torch.load(code_dir / "sequences.pt")
    codes = load_codes(code_dir, max_positions)[:max_positions].float()
    flat_tokens = sequences.reshape(-1)[: codes.shape[0]]

    frac_path = code_dir / "module_frac.pt"
    module_frac = (
        torch.load(frac_path)[: codes.shape[0]].float() / 255.0 if frac_path.exists() else None
    )

    comp_path = code_dir / "active_components.pt"
    names_path = code_dir / "component_names.json"
    active_components = torch.load(comp_path) if comp_path.exists() else None
    component_names = json.loads(names_path.read_text()) if names_path.exists() else None
    return Harvest(
        meta=meta,
        seq_len=meta["seq_len"],
        tokenizer=tokenizer,
        sequences=sequences,
        codes=codes,
        flat_tokens=flat_tokens,
        module_names=meta.get("module_names"),
        module_frac=module_frac,
        component_names=component_names,
        active_components=active_components,
    )


def kmeans_regions(codes: Float[Tensor, "n D"], n_regions: int, seed: int = 0) -> np.ndarray:
    """Assign each code vector to one of `n_regions` MiniBatchKMeans clusters."""
    km = MiniBatchKMeans(n_clusters=n_regions, random_state=seed, n_init="auto", batch_size=4096)
    return km.fit_predict(codes.numpy())


def top_tokens(token_ids: Int[Tensor, " n"], tokenizer: Any, k: int) -> list[tuple[str, int]]:
    """The `k` most common tokens (decoded, repr'd) among `token_ids`, with counts."""
    counts = Counter(token_ids.tolist())
    return [(repr(tokenizer.decode([t])), n) for t, n in counts.most_common(k)]


def token_thumbnails(token_ids: list[int], tokenizer: Any, patch: int) -> np.ndarray:
    """(N, 3, patch, patch) float[0,1] images, each rendering one token's text."""
    font = ImageFont.load_default(size=max(10, patch // 3))
    out = np.zeros((len(token_ids), 3, patch, patch), dtype=np.float32)
    for i, tid in enumerate(token_ids):
        text = tokenizer.decode([tid]).strip()[:6] or "·"
        img = Image.new("RGB", (patch, patch), (24, 24, 28))
        ImageDraw.Draw(img).text((2, patch // 3), text, fill=(235, 235, 235), font=font)
        out[i] = np.transpose(np.asarray(img, dtype=np.float32) / 255.0, (2, 0, 1))
    return out
