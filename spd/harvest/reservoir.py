"""Activation examples reservoir backed by dense tensors.

Stores [n_components, k, window] activation example windows using Algorithm R
for sampling and Efraimidis-Spirakis for merging parallel reservoirs.
"""

import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spd.harvest.schemas import ActivationExample
from spd.utils.general_utils import runtime_cast

WINDOW_PAD_SENTINEL = -1


@dataclass
class ActivationWindows:
    component_idx: Int[Tensor, " n_firings"]
    token_windows: Int[Tensor, "n_firings window_size"]
    firing_windows: Bool[Tensor, "n_firings window_size"]
    activation_windows: dict[str, Float[Tensor, "n_firings window_size"]]


class ActivationExamplesReservoir:
    """Fixed-capacity reservoir of activation example windows per component.

    Each component slot holds up to `k` windows of size `w`, where each window
    contains (token_ids, activation_values, component_acts) aligned by position.

    Use create() for fresh allocation, from_state_dict() for deserialization.
    """

    def __init__(
        self,
        n_components: int,
        k: int,
        window: int,
        device: torch.device,
        tokens: Int[Tensor, "C k w"],
        firings: Bool[Tensor, "C k w"],
        acts: dict[str, Float[Tensor, "C k w"]],
        n_items: Int[Tensor, " C"],
        n_seen: Int[Tensor, " C"],
    ):
        self.n_components = n_components
        self.k = k
        self.window = window
        self.device = device
        self.tokens = tokens
        self.firings = firings
        self.acts = acts
        self.n_items = n_items
        self.n_seen = n_seen

    @classmethod
    def create(
        cls,
        n_components: int,
        k: int,
        window: int,
        device: torch.device,
    ) -> "ActivationExamplesReservoir":
        return cls(
            n_components=n_components,
            k=k,
            window=window,
            device=device,
            tokens=torch.full(
                (n_components, k, window), WINDOW_PAD_SENTINEL, dtype=torch.long, device=device
            ),
            firings=torch.full((n_components, k, window), False, dtype=torch.bool, device=device),
            acts=defaultdict(lambda: torch.zeros(n_components, k, window, device=device)),
            n_items=torch.zeros(n_components, dtype=torch.long, device=device),
            n_seen=torch.zeros(n_components, dtype=torch.long, device=device),
        )

    @classmethod
    def from_state_dict(
        cls, d: dict[str, object], device: torch.device
    ) -> "ActivationExamplesReservoir":
        tokens = runtime_cast(Tensor, d["tokens"])

        acts = runtime_cast(dict, d["acts"])
        acts = {act_type: runtime_cast(Tensor, acts[act_type]).to(device) for act_type in acts}

        return cls(
            n_components=tokens.shape[0],
            k=runtime_cast(int, d["k"]),
            window=runtime_cast(int, d["window"]),
            device=device,
            tokens=tokens.to(device),
            firings=runtime_cast(Tensor, d["firings"]).to(device),
            acts=acts,
            n_items=runtime_cast(Tensor, d["n_items"]).to(device),
            n_seen=runtime_cast(Tensor, d["n_seen"]).to(device),
        )

    def add(self, activation_windows: ActivationWindows) -> None:
        """Add firing windows via Algorithm R.

        Bookkeeping on CPU (cheap integer ops), then batch-write to device.
        """
        device = activation_windows.component_idx.device
        comps = activation_windows.component_idx.cpu().tolist()
        items_cpu = self.n_items.cpu()
        seen_cpu = self.n_seen.cpu()

        write_comps: list[int] = []
        write_slots: list[int] = []
        write_srcs: list[int] = []

        for i, c in enumerate(comps):
            n = int(seen_cpu[c])
            if items_cpu[c] < self.k:
                write_comps.append(c)
                write_slots.append(int(items_cpu[c]))
                write_srcs.append(i)
                items_cpu[c] += 1
            else:
                j = random.randint(0, n)
                if j < self.k:
                    write_comps.append(c)
                    write_slots.append(j)
                    write_srcs.append(i)
            seen_cpu[c] += 1

        self.n_items.copy_(items_cpu)
        self.n_seen.copy_(seen_cpu)

        if write_comps:
            c_t = torch.tensor(write_comps, dtype=torch.long, device=device)
            s_t = torch.tensor(write_slots, dtype=torch.long, device=device)
            f_t = torch.tensor(write_srcs, dtype=torch.long, device=device)

            self.tokens[c_t, s_t] = activation_windows.token_windows[f_t]
            self.firings[c_t, s_t] = activation_windows.firing_windows[f_t]
            for act_type in activation_windows.activation_windows:
                self.acts[act_type][c_t, s_t] = activation_windows.activation_windows[act_type][f_t]

    def merge(self, other: "ActivationExamplesReservoir") -> None:
        """Merge other's reservoir into self via Efraimidis-Spirakis.

        Computes selection indices on small [C, 2k] tensors, then gathers
        from self/other based on whether each selected index came from self or other.
        """
        assert other.n_components == self.n_components
        assert other.k == self.k
        device = self.device
        n_comp = self.n_components

        idx = rearrange(torch.arange(self.k, device=device), "k -> 1 k")
        valid_self = idx < rearrange(self.n_items, "c -> c 1")
        valid_other = idx < rearrange(other.n_items, "c -> c 1")
        valid = torch.cat([valid_self, valid_other], dim=1)

        weights = torch.zeros(n_comp, 2 * self.k, device=device)
        weights[:, : self.k] = rearrange(self.n_seen.float(), "c -> c 1")
        weights[:, self.k :] = rearrange(other.n_seen.float(), "c -> c 1")
        weights[~valid] = 0.0

        rand = torch.rand(n_comp, 2 * self.k, device=device).clamp(min=1e-30)
        keys = rand.pow(1.0 / weights.clamp(min=1.0))
        keys[~valid] = -1.0

        _, top_indices = keys.topk(self.k, dim=1)

        from_self = top_indices < self.k
        self_indices = top_indices.clamp(max=self.k - 1)
        other_indices = (top_indices - self.k).clamp(min=0)

        si = repeat(self_indices, "c k -> c k w", w=self.window)
        oi = repeat(other_indices, "c k -> c k w", w=self.window)
        mask = repeat(from_self, "c k -> c k w", w=self.window)

        self.tokens = torch.where(mask, self.tokens.gather(1, si), other.tokens.gather(1, oi))

        self.firings = torch.where(mask, self.firings.gather(1, si), other.firings.gather(1, oi))

        for act_type in self.acts:
            self.acts[act_type] = torch.where(
                mask,
                self.acts[act_type].gather(1, si),
                other.acts[act_type].gather(1, oi),
            )

        self.n_items = valid.sum(dim=1).clamp(max=self.k)
        self.n_seen = self.n_seen + other.n_seen

    def examples(self, component: int) -> Iterator[ActivationExample]:
        """Yield (token_ids, component_acts), sentinel-filtered."""
        n = int(self.n_items[component])
        for j in range(n):
            toks = self.tokens[component, j]
            firings = self.firings[component, j]
            acts = {act_type: self.acts[act_type][component, j] for act_type in self.acts}

            mask = toks != WINDOW_PAD_SENTINEL  # TODO(oli) not sure this is actually needed

            toks = toks[mask].tolist()
            firings = firings[mask].tolist()
            acts = {act_type: acts[act_type][mask].tolist() for act_type in acts}

            yield ActivationExample(token_ids=toks, firings=firings, activations=acts)

    def to(self, device: torch.device) -> "ActivationExamplesReservoir":
        return ActivationExamplesReservoir(
            n_components=self.n_components,
            k=self.k,
            window=self.window,
            device=device,
            tokens=self.tokens.to(device),
            firings=self.firings.to(device),
            acts={act_type: self.acts[act_type].to(device) for act_type in self.acts},
            n_items=self.n_items.to(device),
            n_seen=self.n_seen.to(device),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "window": self.window,
            "tokens": self.tokens.cpu(),
            "firings": self.firings.cpu(),
            "acts": {act_type: self.acts[act_type].cpu() for act_type in self.acts},
            "n_items": self.n_items.cpu(),
            "n_seen": self.n_seen.cpu(),
        }
