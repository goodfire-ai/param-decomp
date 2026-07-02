"""Synthetic scope shards for developing the viewer without a real harvest.

`write_fixture_store()` writes published (and one in-flight) site shards under
PARAM_DECOMP_OUT_DIR in the exact on-disk format harvest produces, so the viewer's one
read path (`backend.store.ScopeStore`) exercises real mmap seeks, sqlite sort/filter, and
gpt2 detokenization. Point PARAM_DECOMP_OUT_DIR at a throwaway dir first (run_scope.py does).
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from param_decomp_lab.scope.artifacts import (
    FORMAT_VERSION,
    ComponentExamples,
    SiteMeta,
    SiteShardWriter,
    open_labels_db,
    scope_dir,
)
from param_decomp_lab.tokenizer_display import AppTokenizer

TOKENIZER = "gpt2"
K_EXAMPLES = 30
WINDOW = 41
PMI_TOP_K = 20
N_TOKENS_SEEN = 5_000_000
LABELED_FRACTION = 0.6
DEAD_FRACTION = 0.1
LABEL_MODEL = "claude-sonnet-4-5"
LABEL_COST_USD = 0.03

_CORPUS = (
    "the river carried stone past the winter garden while a soldier read a letter by "
    "candlelight; the treaty bound the empire and the republic, and the senate weighed the "
    "verdict of the witness whose testimony filled the ledger. a molecule folds into a "
    "protein, an enzyme cleaves a bond, a neuron fires across a synapse. the theorem rests "
    "on a lemma, the proof on an axiom, the manifold on its topology. the violin answered "
    "the cello over the harbor as the lighthouse marked the voyage home."
)


@dataclass(frozen=True)
class _SubrunPlan:
    subrun_id: str
    published: bool


@dataclass(frozen=True)
class _SitePlan:
    site: str
    n_components: int
    subruns: tuple[_SubrunPlan, ...]


@dataclass(frozen=True)
class _RunPlan:
    run_id: str
    sites: tuple[_SitePlan, ...]


def _plans() -> tuple[_RunPlan, ...]:
    return (
        _RunPlan(
            "p-fixture",
            (
                _SitePlan(
                    "blocks.6.mlp.gate_proj",
                    1200,
                    (_SubrunPlan("h-20260101_000000", published=True),),
                ),
                _SitePlan(
                    "blocks.6.mlp.up_proj",
                    600,
                    (
                        _SubrunPlan("h-20260101_000000", published=True),
                        _SubrunPlan("h-20260102_120000", published=True),
                    ),
                ),
                _SitePlan(
                    "blocks.6.mlp.down_proj",
                    400,
                    (_SubrunPlan("h-inflight", published=False),),
                ),
            ),
        ),
        _RunPlan(
            "p-fixture-attn",
            (
                _SitePlan(
                    "blocks.3.attn.o_proj",
                    300,
                    (_SubrunPlan("h-20260103_000000", published=True),),
                ),
            ),
        ),
    )


def _pmi(rng: np.random.Generator, pool: np.ndarray, trigger: int) -> list[tuple[int, float]]:
    tokens = [trigger, *rng.choice(pool, size=PMI_TOP_K - 1, replace=False).tolist()]
    score = float(rng.uniform(6.0, 10.0))
    pairs: list[tuple[int, float]] = []
    for token in tokens:
        pairs.append((int(token), round(score, 3)))
        score *= float(rng.uniform(0.85, 0.97))
    return pairs


def _examples(
    rng: np.random.Generator, pool: np.ndarray, trigger: int, peak0: float
) -> ComponentExamples:
    n = int(rng.integers(8, K_EXAMPLES + 1))
    lengths = rng.integers(25, WINDOW + 1, size=n).astype(np.uint16)
    token_ids = np.zeros((n, WINDOW), np.uint32)
    firings = np.zeros((n, WINDOW), np.uint8)
    ci = np.zeros((n, WINDOW), np.float16)
    act = np.zeros((n, WINDOW), np.float16)
    for j in range(n):
        length = int(lengths[j])
        center = length // 2
        ids = rng.choice(pool, size=length)
        ids[center] = trigger
        peak = peak0 * float(rng.uniform(0.5, 1.0))
        positions = np.arange(length)
        envelope = peak * 2.0 ** (-np.abs(positions - center) / float(rng.uniform(1.0, 3.0)))
        values = envelope * rng.uniform(0.5, 1.0, size=length)
        values[center] = peak
        token_ids[j, :length] = ids
        act[j, :length] = values
        ci[j, :length] = np.clip(values / peak * rng.uniform(0.6, 1.2, size=length), 0.0, 1.0)
        firings[j, :length] = 1
    return ComponentExamples(token_ids=token_ids, firings=firings, ci=ci, act=act, lengths=lengths)


def _dead_examples() -> ComponentExamples:
    return ComponentExamples(
        token_ids=np.zeros((0, WINDOW), np.uint32),
        firings=np.zeros((0, WINDOW), np.uint8),
        ci=np.zeros((0, WINDOW), np.float16),
        act=np.zeros((0, WINDOW), np.float16),
        lengths=np.zeros((0,), np.uint16),
    )


def _write_shard(plan: _RunPlan, site: _SitePlan, subrun_id: str, pool: np.ndarray) -> None:
    rng = np.random.default_rng(abs(hash((plan.run_id, site.site, subrun_id))) % (2**32))
    n = site.n_components
    ranks = rng.permutation(n).astype(np.float64)
    densities = 0.35 * (ranks + 1.0) ** -0.75 * rng.uniform(0.8, 1.2, size=n)
    mean_cis = np.clip(densities / densities.max() * rng.uniform(0.7, 1.0, size=n) + 0.05, 0.0, 1.0)
    dead = rng.random(n) < DEAD_FRACTION

    meta = SiteMeta(
        format_version=FORMAT_VERSION,
        run_id=plan.run_id,
        site=site.site,
        subrun_id=subrun_id,
        n_components=n,
        k_examples=K_EXAMPLES,
        window=WINDOW,
        tokenizer_name=TOKENIZER,
        n_tokens_seen=N_TOKENS_SEEN,
        pmi_top_k=PMI_TOP_K,
        provenance="fixture",
        created_at=datetime.now(UTC).isoformat(),
    )
    writer = SiteShardWriter(meta)
    for idx in range(n):
        if dead[idx]:
            writer.write_component(
                idx,
                _dead_examples(),
                firing_count=0,
                firing_density=0.0,
                max_act=0.0,
                mean_ci=0.0,
                mean_act=0.0,
                input_pmi=[],
                output_pmi=[],
            )
            continue
        trigger = int(rng.choice(pool))
        peak0 = float(rng.lognormal(mean=1.2, sigma=0.6))
        examples = _examples(rng, pool, trigger, peak0)
        firing_count = int(examples.lengths.sum())
        writer.write_component(
            idx,
            examples,
            firing_count=firing_count,
            firing_density=float(densities[idx]),
            max_act=float(examples.act.max()),
            mean_ci=float(mean_cis[idx]),
            mean_act=float(examples.act.sum() / firing_count),
            input_pmi=_pmi(rng, pool, trigger),
            output_pmi=_pmi(rng, pool, trigger),
        )
    writer.publish()


def _write_labels(plan: _RunPlan, site: _SitePlan, tok: AppTokenizer, pool: np.ndarray) -> None:
    conn = open_labels_db(plan.run_id, readonly=False)
    rng = np.random.default_rng(abs(hash((plan.run_id, site.site, "labels"))) % (2**32))
    created = datetime.now(UTC).isoformat()
    rows = []
    for idx in range(site.n_components):
        if rng.random() >= LABELED_FRACTION:
            continue
        word = tok.get_tok_display(int(rng.choice(pool))).strip()
        rows.append(
            (site.site, idx, f'fires on "{word}"', LABEL_MODEL, LABEL_COST_USD, created, "fixture")
        )
    conn.executemany(
        "INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def write_fixture_store() -> None:
    tok = AppTokenizer.from_pretrained(TOKENIZER)
    pool = np.array(sorted(set(tok.encode(_CORPUS))), dtype=np.uint32)
    for plan in _plans():
        for site in plan.sites:
            for subrun in site.subruns:
                if subrun.published:
                    _write_shard(plan, site, subrun.subrun_id, pool)
                else:
                    (scope_dir(plan.run_id) / site.site / f".tmp-{subrun.subrun_id}").mkdir(
                        parents=True
                    )
            if any(s.published for s in site.subruns):
                _write_labels(plan, site, tok, pool)


if __name__ == "__main__":
    write_fixture_store()
    print(json.dumps({"out_dir": str(scope_dir("p-fixture").parent.parent)}, indent=2))
