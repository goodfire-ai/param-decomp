"""Activation examples reservoir backed by dense arrays.

Stores [n_components, k, window] activation example windows using Algorithm R
for sampling and Efraimidis-Spirakis for merging parallel reservoirs.
"""

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int

from param_decomp_config.base import runtime_cast
from param_decomp_lab.harvest.schemas import ActivationExample

WINDOW_PAD_SENTINEL = -1


@dataclass
class ActivationWindows:
    component_idx: Int[np.ndarray, " n_firings"]
    token_windows: Int[np.ndarray, "n_firings window_size"]
    firing_windows: Bool[np.ndarray, "n_firings window_size"]
    activation_windows: dict[str, Float[np.ndarray, "n_firings window_size"]]


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
        tokens: Int[np.ndarray, "C k w"],
        firings: Bool[np.ndarray, "C k w"],
        acts: dict[str, Float[np.ndarray, "C k w"]],
        n_items: Int[np.ndarray, " C"],
        n_seen: Int[np.ndarray, " C"],
    ):
        self.n_components = n_components
        self.k = k
        self.window = window
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
    ) -> "ActivationExamplesReservoir":
        return cls(
            n_components=n_components,
            k=k,
            window=window,
            tokens=np.full((n_components, k, window), WINDOW_PAD_SENTINEL, dtype=np.int64),
            firings=np.zeros((n_components, k, window), dtype=np.bool_),
            acts=defaultdict(lambda: np.zeros((n_components, k, window), dtype=np.float32)),
            n_items=np.zeros(n_components, dtype=np.int64),
            n_seen=np.zeros(n_components, dtype=np.int64),
        )

    @classmethod
    def from_state_dict(cls, d: dict[str, object]) -> "ActivationExamplesReservoir":
        tokens = runtime_cast(np.ndarray, d["tokens"])

        acts = runtime_cast(dict, d["acts"])
        acts = {act_type: runtime_cast(np.ndarray, acts[act_type]) for act_type in acts}

        return cls(
            n_components=tokens.shape[0],
            k=runtime_cast(int, d["k"]),
            window=runtime_cast(int, d["window"]),
            tokens=tokens,
            firings=runtime_cast(np.ndarray, d["firings"]),
            acts=acts,
            n_items=runtime_cast(np.ndarray, d["n_items"]),
            n_seen=runtime_cast(np.ndarray, d["n_seen"]),
        )

    def add(self, activation_windows: ActivationWindows, rng: np.random.Generator) -> None:
        """Add firing windows via Algorithm R."""
        comps = activation_windows.component_idx.tolist()

        write_comps: list[int] = []
        write_slots: list[int] = []
        write_srcs: list[int] = []

        for i, c in enumerate(comps):
            n = int(self.n_seen[c])
            if self.n_items[c] < self.k:
                write_comps.append(c)
                write_slots.append(int(self.n_items[c]))
                write_srcs.append(i)
                self.n_items[c] += 1
            else:
                j = int(rng.integers(0, n + 1))
                if j < self.k:
                    write_comps.append(c)
                    write_slots.append(j)
                    write_srcs.append(i)
            self.n_seen[c] += 1

        if write_comps:
            c_t = np.array(write_comps, dtype=np.int64)
            s_t = np.array(write_slots, dtype=np.int64)
            f_t = np.array(write_srcs, dtype=np.int64)

            self.tokens[c_t, s_t] = activation_windows.token_windows[f_t]
            self.firings[c_t, s_t] = activation_windows.firing_windows[f_t]
            for act_type in activation_windows.activation_windows:
                self.acts[act_type][c_t, s_t] = activation_windows.activation_windows[act_type][f_t]

    def merge(self, other: "ActivationExamplesReservoir", rng: np.random.Generator) -> None:
        """Merge other's reservoir into self via Efraimidis-Spirakis.

        Computes selection indices on small [C, 2k] arrays, then gathers
        from self/other based on whether each selected index came from self or other.
        """
        assert other.n_components == self.n_components
        assert other.k == self.k
        n_comp = self.n_components

        idx = np.arange(self.k).reshape(1, self.k)
        valid_self = idx < self.n_items.reshape(n_comp, 1)
        valid_other = idx < other.n_items.reshape(n_comp, 1)
        valid = np.concatenate([valid_self, valid_other], axis=1)

        weights = np.zeros((n_comp, 2 * self.k), dtype=np.float64)
        weights[:, : self.k] = self.n_seen.astype(np.float64).reshape(n_comp, 1)
        weights[:, self.k :] = other.n_seen.astype(np.float64).reshape(n_comp, 1)
        weights[~valid] = 0.0

        rand = np.clip(rng.random((n_comp, 2 * self.k)), 1e-30, None)
        keys = rand ** (1.0 / np.clip(weights, 1.0, None))
        keys[~valid] = -1.0

        top_indices = np.argsort(keys, axis=1)[:, ::-1][:, : self.k]

        from_self = top_indices < self.k
        self_indices = np.clip(top_indices, None, self.k - 1)
        other_indices = np.clip(top_indices - self.k, 0, None)

        si = np.broadcast_to(self_indices[:, :, None], (n_comp, self.k, self.window))
        oi = np.broadcast_to(other_indices[:, :, None], (n_comp, self.k, self.window))
        mask = np.broadcast_to(from_self[:, :, None], (n_comp, self.k, self.window))

        self.tokens = np.where(
            mask,
            np.take_along_axis(self.tokens, si, axis=1),
            np.take_along_axis(other.tokens, oi, axis=1),
        )
        self.firings = np.where(
            mask,
            np.take_along_axis(self.firings, si, axis=1),
            np.take_along_axis(other.firings, oi, axis=1),
        )
        for act_type in self.acts:
            self.acts[act_type] = np.where(
                mask,
                np.take_along_axis(self.acts[act_type], si, axis=1),
                np.take_along_axis(other.acts[act_type], oi, axis=1),
            )

        self.n_items = np.clip(valid.sum(axis=1), None, self.k)
        self.n_seen = self.n_seen + other.n_seen

    def examples(self, component: int) -> Iterator[ActivationExample]:
        """Yield (token_ids, component_acts), sentinel-filtered."""
        n = int(self.n_items[component])
        for j in range(n):
            toks = self.tokens[component, j]
            firings = self.firings[component, j]
            acts = {act_type: self.acts[act_type][component, j] for act_type in self.acts}

            mask = toks != WINDOW_PAD_SENTINEL

            toks_kept = toks[mask].tolist()
            firings_kept = firings[mask].tolist()
            acts_kept = {act_type: acts[act_type][mask].tolist() for act_type in acts}

            yield ActivationExample(
                token_ids=toks_kept, firings=firings_kept, activations=acts_kept
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "window": self.window,
            "tokens": self.tokens,
            "firings": self.firings,
            "acts": {act_type: self.acts[act_type] for act_type in self.acts},
            "n_items": self.n_items,
            "n_seen": self.n_seen,
        }
