"""Deterministic synthetic ScopeDataSource for developing the scope UI without real artifacts."""

import random
import time
import zlib
from dataclasses import dataclass

import numpy as np

from param_decomp_lab.scope.backend.data_source import (
    ActivationExample,
    CatalogResponse,
    ComponentDetail,
    ComponentLabel,
    ComponentListResponse,
    ComponentRow,
    RunEntry,
    ScopeNotFoundError,
    SiteEntry,
    SortKey,
    SubrunEntry,
)

WORDS = [
    "the",
    "of",
    "and",
    "to",
    "in",
    "for",
    "with",
    "on",
    "as",
    "by",
    "from",
    "at",
    "this",
    "that",
    "not",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "river",
    "bridge",
    "stone",
    "garden",
    "winter",
    "summer",
    "harvest",
    "letter",
    "window",
    "candle",
    "journey",
    "market",
    "horse",
    "soldier",
    "captain",
    "doctor",
    "village",
    "castle",
    "forest",
    "mountain",
    "valley",
    "ocean",
    "island",
    "music",
    "silence",
    "memory",
    "shadow",
    "morning",
    "evening",
    "twilight",
    "thunder",
    "whisper",
    "promise",
    "treaty",
    "empire",
    "republic",
    "senate",
    "council",
    "verdict",
    "witness",
    "testimony",
    "statute",
    "clause",
    "molecule",
    "protein",
    "enzyme",
    "neuron",
    "synapse",
    "lattice",
    "quantum",
    "vector",
    "tensor",
    "gradient",
    "theorem",
    "lemma",
    "proof",
    "axiom",
    "integer",
    "prime",
    "matrix",
    "kernel",
    "manifold",
    "topology",
    "sonnet",
    "stanza",
    "metaphor",
    "irony",
    "satire",
    "chronicle",
    "parable",
    "preface",
    "epilogue",
    "glossary",
    "copper",
    "iron",
    "mercury",
    "sulfur",
    "amber",
    "ivory",
    "linen",
    "velvet",
    "parchment",
    "vellum",
    "anchor",
    "compass",
    "rudder",
    "mast",
    "harbor",
    "lighthouse",
    "voyage",
    "cargo",
    "manifest",
    "ledger",
    "violin",
    "cello",
    "oboe",
    "chorus",
    "overture",
    "cadence",
    "refrain",
    "crescendo",
    "aria",
    "libretto",
]

THEMES = (
    "legal and contractual",
    "maritime navigation",
    "biochemical pathway",
    "nineteenth-century epistolary",
    "mathematical proof",
    "musical notation",
    "agricultural almanac",
    "parliamentary procedure",
    "weather forecasting",
    "culinary instruction",
)

FIXTURE_LABEL_MODEL = "claude-sonnet-4-5"
FIXTURE_LABEL_COST_USD = 0.03
LABELED_FRACTION = 0.6
MAX_EXAMPLES = 30
EXAMPLE_WINDOW = 41
PMI_TOP_K = 20


def _seed(*parts: str | int) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


@dataclass(frozen=True)
class _SubrunSpec:
    subrun_id: str
    n_batches: int
    progress0: float
    progress_per_s: float


@dataclass(frozen=True)
class _SiteSpec:
    site: str
    n_components: int
    subruns: tuple[_SubrunSpec, ...]


@dataclass(frozen=True)
class _RunSpec:
    run_id: str
    sites: tuple[_SiteSpec, ...]


def _present(site_idx: int) -> tuple[_SubrunSpec, ...]:
    return (_SubrunSpec(f"h-{site_idx:02d}a", 400, 1.0, 0.0),)


def _in_flight(site_idx: int, progress0: float, progress_per_s: float) -> tuple[_SubrunSpec, ...]:
    return (_SubrunSpec(f"h-{site_idx:02d}a", 400, progress0, progress_per_s),)


def _fixture_runs() -> tuple[_RunSpec, ...]:
    big_run = _RunSpec(
        "p-aa11bb22",
        tuple(
            _SiteSpec(f"model.layers.18.mlp.{proj}", 30_000, _present(i))
            for i, proj in enumerate(["gate_proj", "up_proj", "down_proj"])
        ),
    )

    kinds = ["self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    # Per (layer, kind) cell: present / in-flight(progress0, rate) / absent (no subruns).
    # Two fast in-flight sites (~0.2%/s) visibly tick and eventually flip to present
    # while the catalog page polls; the slow ones stay in flight for hours.
    statuses: dict[tuple[int, int], tuple[float, float] | str] = {
        (0, 0): "present",
        (0, 1): "present",
        (0, 2): "present",
        (0, 3): "two_subruns",
        (1, 0): "present",
        (1, 1): (0.62, 0.00002),
        (1, 2): (0.35, 0.00002),
        (1, 3): "present",
        (2, 0): (0.85, 0.002),
        (2, 1): "present",
        (2, 2): (0.15, 0.002),
        (2, 3): "absent",
        (3, 0): "absent",
        (3, 1): (0.05, 0.00002),
        (3, 2): "present",
        (3, 3): (0.50, 0.00005),
        (4, 0): "present",
        (4, 1): "absent",
        (4, 2): (0.70, 0.00002),
        (4, 3): "present",
    }
    sites = []
    for (layer, kind_i), status in sorted(statuses.items()):
        site_idx = layer * len(kinds) + kind_i
        n_components = 2_000 + 500 * site_idx
        match status:
            case "present":
                subruns = _present(site_idx)
            case "absent":
                subruns = ()
            case "two_subruns":
                subruns = (
                    *_present(site_idx),
                    _SubrunSpec(f"h-{site_idx:02d}b", 800, 0.4, 0.00003),
                )
            case (progress0, rate):
                subruns = _in_flight(site_idx, progress0, rate)
            case _:
                raise ValueError(f"bad status spec: {status}")
        sites.append(_SiteSpec(f"model.layers.{layer}.{kinds[kind_i]}", n_components, subruns))
    mixed_run = _RunSpec("p-cc33dd44", tuple(sites))

    return (big_run, mixed_run)


@dataclass
class _SiteColumns:
    """Compact per-site columns — the fixture analogue of the future mmap column store."""

    densities: np.ndarray
    mean_cis: np.ndarray
    max_acts: np.ndarray
    labels: list[str | None]


def _component_semantics(run_id: str, site: str, idx: int) -> tuple[list[str], str, str]:
    """(trigger tokens, theme, label text) for one component — shared by listing and detail."""
    rng = random.Random(_seed(run_id, site, idx, "sem"))
    triggers = rng.sample(WORDS, 3)
    theme = rng.choice(THEMES)
    text = f'fires on "{triggers[0]}" and "{triggers[1]}" in {theme} contexts'
    return triggers, theme, text


def _build_columns(run_id: str, spec: _SiteSpec) -> _SiteColumns:
    n = spec.n_components
    rng = np.random.default_rng(_seed(run_id, spec.site, "columns"))
    zipf_ranks = rng.permutation(n).astype(np.float64)
    densities = 0.35 * (zipf_ranks + 1.0) ** -0.75 * rng.uniform(0.8, 1.2, size=n)
    mean_cis = np.clip(densities / densities.max() * rng.uniform(0.7, 1.0, size=n) + 0.05, 0.0, 1.0)
    max_acts = rng.lognormal(mean=1.2, sigma=0.6, size=n)
    labeled = rng.random(n) < LABELED_FRACTION
    labels: list[str | None] = [
        _component_semantics(run_id, spec.site, idx)[2] if labeled[idx] else None
        for idx in range(n)
    ]
    return _SiteColumns(densities=densities, mean_cis=mean_cis, max_acts=max_acts, labels=labels)


class FixtureDataSource:
    """Seeded synthetic data: every value is a pure function of (run, site, idx) plus
    wall-clock time for in-flight subrun progress. Component details are generated on
    demand for a single idx; only compact listing columns are cached per site."""

    def __init__(self) -> None:
        self._t0 = time.time()
        self._runs = {run.run_id: run for run in _fixture_runs()}
        self._columns_cache: dict[tuple[str, str], _SiteColumns] = {}
        self._posted_labels: dict[tuple[str, str, int], ComponentLabel] = {}

    def _site_spec(self, run_id: str, site: str) -> _SiteSpec:
        if run_id not in self._runs:
            raise ScopeNotFoundError(f"unknown run {run_id}")
        specs = [s for s in self._runs[run_id].sites if s.site == site]
        if not specs:
            raise ScopeNotFoundError(f"unknown site {site} in run {run_id}")
        return specs[0]

    def _subrun_entry(self, spec: _SubrunSpec) -> SubrunEntry:
        progress = min(1.0, spec.progress0 + spec.progress_per_s * (time.time() - self._t0))
        status = "present" if progress >= 1.0 else "in_flight"
        return SubrunEntry(
            subrun_id=spec.subrun_id,
            status=status,
            n_batches=spec.n_batches,
            progress=round(progress, 4),
        )

    def _browsable_columns(self, run_id: str, site: str) -> tuple[_SiteSpec, _SiteColumns]:
        """Columns for a site that has at least one present subrun; 404 otherwise."""
        spec = self._site_spec(run_id, site)
        entries = [self._subrun_entry(s) for s in spec.subruns]
        if not any(e.status == "present" for e in entries):
            raise ScopeNotFoundError(f"site {site} of run {run_id} has no present subrun yet")
        key = (run_id, site)
        if key not in self._columns_cache:
            self._columns_cache[key] = _build_columns(run_id, spec)
        return spec, self._columns_cache[key]

    def _label_text(self, run_id: str, site: str, idx: int, columns: _SiteColumns) -> str | None:
        posted = self._posted_labels.get((run_id, site, idx))
        if posted is not None:
            return posted.text
        return columns.labels[idx]

    def catalog(self) -> CatalogResponse:
        runs = []
        for run in self._runs.values():
            sites = []
            for spec in run.sites:
                entries = [self._subrun_entry(s) for s in spec.subruns]
                if any(e.status == "present" for e in entries):
                    _, columns = self._browsable_columns(run.run_id, spec.site)
                    base = sum(label is not None for label in columns.labels)
                    posted_unlabeled = sum(
                        columns.labels[i] is None
                        for (r, s, i) in self._posted_labels
                        if (r, s) == (run.run_id, spec.site)
                    )
                    n_labeled = base + posted_unlabeled
                else:
                    n_labeled = 0
                sites.append(
                    SiteEntry(
                        site=spec.site,
                        n_components=spec.n_components,
                        n_labeled=n_labeled,
                        subruns=entries,
                    )
                )
            runs.append(RunEntry(run_id=run.run_id, sites=sites))
        return CatalogResponse(runs=runs)

    def list_components(
        self, run_id: str, site: str, sort: SortKey, page: int, page_size: int, q: str
    ) -> ComponentListResponse:
        assert page >= 0 and 1 <= page_size <= 200
        spec, columns = self._browsable_columns(run_id, site)
        n = spec.n_components

        def label_at(idx: int) -> str | None:
            return self._label_text(run_id, site, idx, columns)

        if q:
            needle = q.lower()
            candidate_idxs = np.array(
                [i for i in range(n) if (lbl := label_at(i)) is not None and needle in lbl.lower()],
                dtype=np.int64,
            )
        else:
            candidate_idxs = np.arange(n, dtype=np.int64)

        match sort:
            case "mean_ci":
                order = np.argsort(-columns.mean_cis[candidate_idxs], kind="stable")
            case "density":
                order = np.argsort(-columns.densities[candidate_idxs], kind="stable")
            case "max_act":
                order = np.argsort(-columns.max_acts[candidate_idxs], kind="stable")
            case "unlabeled_first":
                unlabeled_last = np.array(
                    [label_at(int(i)) is not None for i in candidate_idxs], dtype=np.int8
                )
                order = np.argsort(unlabeled_last, kind="stable")
        ranked = candidate_idxs[order]

        page_idxs = ranked[page * page_size : (page + 1) * page_size]
        items = [
            ComponentRow(
                idx=int(i),
                mean_ci=round(float(columns.mean_cis[i]), 6),
                density=round(float(columns.densities[i]), 6),
                max_act=round(float(columns.max_acts[i]), 4),
                label=label_at(int(i)),
            )
            for i in page_idxs
        ]
        return ComponentListResponse(total=len(ranked), page=page, items=items)

    def component_detail(self, run_id: str, site: str, idx: int) -> ComponentDetail:
        spec, columns = self._browsable_columns(run_id, site)
        if not 0 <= idx < spec.n_components:
            raise ScopeNotFoundError(
                f"component {idx} out of range for {site} ({spec.n_components})"
            )

        triggers, _theme, _text = _component_semantics(run_id, site, idx)
        rng = random.Random(_seed(run_id, site, idx, "detail"))
        max_act = float(columns.max_acts[idx])

        def pmi_list(direction: str) -> list[tuple[str, float]]:
            drng = random.Random(_seed(run_id, site, idx, "pmi", direction))
            non_triggers = [w for w in WORDS if w not in triggers]
            tokens = triggers + drng.sample(non_triggers, PMI_TOP_K - len(triggers))
            score = drng.uniform(6.0, 10.0)
            scored = []
            for token in tokens:
                scored.append((f" {token}", round(score, 3)))
                score *= drng.uniform(0.85, 0.97)
            return scored

        n_examples = rng.randint(8, MAX_EXAMPLES)
        example_peaks = sorted(
            (max_act * rng.uniform(0.5, 1.0) for _ in range(n_examples)), reverse=True
        )
        examples = [self._make_example(rng, triggers, peak) for peak in example_peaks]

        posted = self._posted_labels.get((run_id, site, idx))
        stored_text = columns.labels[idx]
        if posted is not None:
            label = posted
        elif stored_text is not None:
            lrng = random.Random(_seed(run_id, site, idx, "label_meta"))
            label = ComponentLabel(
                text=stored_text,
                model=FIXTURE_LABEL_MODEL,
                cost_usd=FIXTURE_LABEL_COST_USD,
                created_at=f"2026-05-{lrng.randint(1, 28):02d}T{lrng.randint(0, 23):02d}:{lrng.randint(0, 59):02d}:00Z",
            )
        else:
            label = None

        order = np.argsort(-columns.mean_cis, kind="stable")
        rank_of = int(np.nonzero(order == idx)[0][0])
        return ComponentDetail(
            idx=idx,
            rank=rank_of,
            prev_idx=int(order[rank_of - 1]) if rank_of > 0 else None,
            next_idx=int(order[rank_of + 1]) if rank_of + 1 < len(order) else None,
            density=round(float(columns.densities[idx]), 6),
            max_act=round(max_act, 4),
            mean_ci=round(float(columns.mean_cis[idx]), 6),
            label=label,
            input_pmi=pmi_list("input"),
            output_pmi=pmi_list("output"),
            examples=examples,
        )

    def _make_example(
        self, rng: random.Random, triggers: list[str], peak: float
    ) -> ActivationExample:
        center = EXAMPLE_WINDOW // 2
        tokens = [f" {rng.choice(WORDS)}" for _ in range(EXAMPLE_WINDOW)]
        tokens[center] = f" {rng.choice(triggers)}"

        acts = []
        for pos in range(EXAMPLE_WINDOW):
            envelope = peak * 2.0 ** (-abs(pos - center) / rng.uniform(1.0, 3.0))
            background = peak * 0.05 * rng.random() if rng.random() < 0.15 else 0.0
            acts.append(round(max(envelope * rng.uniform(0.5, 1.0), background), 3))
        acts[center] = round(peak, 3)

        cis = [
            round(min(1.0, max(0.0, (a / peak) * rng.uniform(0.6, 1.2))), 3) if a > 0 else 0.0
            for a in acts
        ]
        return ActivationExample(tokens=tokens, acts=acts, cis=cis, max_act=round(peak, 3))

    def create_label(self, run_id: str, site: str, idx: int) -> ComponentLabel:
        spec, _ = self._browsable_columns(run_id, site)
        assert 0 <= idx < spec.n_components, f"component {idx} out of range"
        time.sleep(1.0)  # simulated LLM latency
        triggers, theme, _ = _component_semantics(run_id, site, idx)
        label = ComponentLabel(
            text=f'fires on "{triggers[0]}" and "{triggers[2]}" in {theme} contexts (relabeled)',
            model=FIXTURE_LABEL_MODEL,
            cost_usd=FIXTURE_LABEL_COST_USD,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._posted_labels[(run_id, site, idx)] = label
        return label
