"""Harvest data repository.

Single read/write entry for one harvest subrun of a decomposition. Per-component
example + PMI + scalar data lives in the scope artifact shards
(`scope/<site>/<subrun>/`, one per site); this repo reconstructs the harvest-consumer
types (`ComponentData` / `ComponentSummary`) from those shards on demand. The small
global provenance (config) and the LLM-eval byproducts (intruder scores + prompts) live
in a slim `harvest.db` beside the tensor sidecars (`component_correlations.npz`,
`token_stats.npz`).

Shards store ALL of a site's components (dead ones too, for stable idx); the
harvest-consumer read API filters to fired components (`firing_count > 0`) to preserve
the old only-fired semantics.

Layout: runs/<decomposition_id>/harvest/h-YYYYMMDD_HHMMSS/{harvest.db, *.npz}
        runs/<decomposition_id>/scope/<site>/h-YYYYMMDD_HHMMSS/{examples.bin, site.db, meta.json}
"""

import json

import numpy as np

from param_decomp.log import logger
from param_decomp_lab.harvest.accumulator import Harvester
from param_decomp_lab.harvest.config import HarvestConfig
from param_decomp_lab.harvest.db import HarvestDB
from param_decomp_lab.harvest.schemas import (
    ActivationExample,
    ComponentData,
    ComponentSummary,
    ComponentTokenPMI,
    get_harvest_dir,
    get_harvest_subrun_dir,
)
from param_decomp_lab.harvest.scope_writer import (
    CI_ACT_TYPE,
    COMPONENT_ACT_TYPE,
    write_scope_shards,
)
from param_decomp_lab.harvest.storage import CorrelationStorage, TokenStatsStorage
from param_decomp_lab.scope.artifacts import SiteShardReader, scope_dir


def _pmi_from_json(raw: str) -> ComponentTokenPMI:
    return ComponentTokenPMI(top=[(int(t), float(p)) for t, p in json.loads(raw)], bottom=[])


class HarvestRepo:
    """Access to harvest data for a single harvest subrun of a decomposition."""

    def __init__(self, decomposition_id: str, subrun_id: str, readonly: bool) -> None:
        self.decomposition_id = decomposition_id
        self.subrun_id = subrun_id
        self._dir = get_harvest_subrun_dir(decomposition_id, subrun_id)
        self._db = HarvestDB(self._dir / "harvest.db", readonly=readonly)
        self._readers: dict[str, SiteShardReader] = {}

    @classmethod
    def open_most_recent(
        cls,
        decomposition_id: str,
        readonly: bool = True,
    ) -> "HarvestRepo | None":
        """Open harvest data. Returns None if no harvest data exists."""
        decomposition_subruns_dir = get_harvest_dir(decomposition_id)
        if not decomposition_subruns_dir.exists():
            return None

        subrun_candidates = sorted(
            d for d in decomposition_subruns_dir.iterdir() if d.is_dir() and d.name.startswith("h-")
        )
        if not subrun_candidates:
            return None

        subrun_dir = subrun_candidates[-1]
        if not (subrun_dir / "harvest.db").exists():
            logger.info(f"No harvest data found for {decomposition_id}")
            return None

        logger.info(f"Opening harvest data for {decomposition_id} from {subrun_dir}")
        return cls(decomposition_id=decomposition_id, subrun_id=subrun_dir.name, readonly=readonly)

    @staticmethod
    def save_results(
        harvester: Harvester,
        config: HarvestConfig,
        run_id: str,
        subrun_id: str,
        tokenizer_name: str,
    ) -> None:
        """Write a finished harvester: scope shards (per site) + slim harvest.db + npz."""
        output_dir = get_harvest_subrun_dir(run_id, subrun_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        db = HarvestDB(output_dir / "harvest.db")
        db.save_config(config)
        db.close()

        logger.info("Writing scope shards...")
        write_scope_shards(harvester, run_id, subrun_id, tokenizer_name, config.pmi_token_top_k)

        component_keys = harvester.component_keys
        if harvester.cooccurrence_counts is not None:
            CorrelationStorage(
                component_keys=component_keys,
                count_i=harvester.firing_counts,
                count_ij=harvester.cooccurrence_counts,
                count_total=harvester.total_tokens_processed,
            ).save(output_dir / "component_correlations.npz")

        TokenStatsStorage(
            component_keys=component_keys,
            vocab_size=harvester.vocab_size,
            n_tokens=harvester.total_tokens_processed,
            input_counts=harvester.input_cooccurrence,
            input_totals=harvester.input_marginals.astype(np.float64),
            output_counts=harvester.output_cooccurrence,
            output_totals=harvester.output_marginals,
            firing_counts=harvester.firing_counts,
        ).save(output_dir / "token_stats.npz")
        logger.info(f"Saved harvest results to {output_dir} + scope shards")

    # -- Scope shard access ----------------------------------------------------

    def _sites(self) -> list[str]:
        base = scope_dir(self.decomposition_id)
        if not base.exists():
            return []
        return sorted(
            d.name
            for d in base.iterdir()
            if d.is_dir() and (d / self.subrun_id / "meta.json").exists()
        )

    def _reader(self, site: str) -> SiteShardReader:
        if site not in self._readers:
            self._readers[site] = SiteShardReader(
                scope_dir(self.decomposition_id) / site / self.subrun_id
            )
        return self._readers[site]

    def _reconstruct(self, site: str, idx: int) -> ComponentData | None:
        """Reconstruct a fired component; None for an unknown or dead (firing_count == 0)
        idx — shards store dead components too (stable idx), the facade hides them."""
        reader = self._reader(site)
        row = reader.db.execute(
            "SELECT firing_count, firing_density, mean_ci, mean_act, input_pmi, output_pmi "
            "FROM components WHERE idx = ?",
            (idx,),
        ).fetchone()
        if row is None or row[0] == 0:
            return None
        _, density, mean_ci, mean_act, in_pmi, out_pmi = row
        ex = reader.examples(idx)
        examples = [
            ActivationExample(
                token_ids=[int(t) for t in ex.token_ids[j, : ex.lengths[j]]],
                firings=[bool(f) for f in ex.firings[j, : ex.lengths[j]]],
                activations={
                    CI_ACT_TYPE: [float(c) for c in ex.ci[j, : ex.lengths[j]]],
                    COMPONENT_ACT_TYPE: [float(a) for a in ex.act[j, : ex.lengths[j]]],
                },
            )
            for j in range(ex.token_ids.shape[0])
        ]
        return ComponentData(
            component_key=f"{site}:{idx}",
            layer=site,
            component_idx=idx,
            mean_activations={CI_ACT_TYPE: mean_ci, COMPONENT_ACT_TYPE: mean_act},
            firing_density=density,
            activation_examples=examples,
            input_token_pmi=_pmi_from_json(in_pmi),
            output_token_pmi=_pmi_from_json(out_pmi),
        )

    # -- Provenance ------------------------------------------------------------

    def get_config(self) -> dict[str, object]:
        return self._db.get_config_dict()

    def get_component_count(self) -> int:
        total = 0
        for site in self._sites():
            (n,) = (
                self._reader(site)
                .db.execute("SELECT COUNT(*) FROM components WHERE firing_count > 0")
                .fetchone()
            )
            total += n
        return total

    # -- Component data (reconstructed from scope shards) ----------------------

    def get_summary(self) -> dict[str, ComponentSummary]:
        out: dict[str, ComponentSummary] = {}
        for site in self._sites():
            rows = (
                self._reader(site)
                .db.execute(
                    "SELECT idx, firing_density, mean_ci, mean_act FROM components WHERE firing_count > 0"
                )
                .fetchall()
            )
            for idx, density, mean_ci, mean_act in rows:
                out[f"{site}:{idx}"] = ComponentSummary(
                    layer=site,
                    component_idx=idx,
                    firing_density=density,
                    mean_activations={CI_ACT_TYPE: mean_ci, COMPONENT_ACT_TYPE: mean_act},
                )
        return out

    def get_component(self, component_key: str) -> ComponentData | None:
        site, _, idx = component_key.rpartition(":")
        if site not in self._sites():
            return None
        return self._reconstruct(site, int(idx))

    def get_components_bulk(self, component_keys: list[str]) -> dict[str, ComponentData]:
        out: dict[str, ComponentData] = {}
        for key in component_keys:
            comp = self.get_component(key)
            if comp is not None:
                out[key] = comp
        return out

    def get_all_components(self) -> list[ComponentData]:
        out: list[ComponentData] = []
        for site in self._sites():
            rows = (
                self._reader(site)
                .db.execute("SELECT idx FROM components WHERE firing_count > 0 ORDER BY idx")
                .fetchall()
            )
            for (idx,) in rows:
                comp = self._reconstruct(site, idx)
                assert comp is not None
                out.append(comp)
        return out

    def get_component_keys(self) -> list[str]:
        keys: list[str] = []
        for site in self._sites():
            rows = (
                self._reader(site)
                .db.execute("SELECT idx FROM components WHERE firing_count > 0")
                .fetchall()
            )
            keys.extend(f"{site}:{idx}" for (idx,) in rows)
        return keys

    def get_eligible_component_keys(self, min_examples: int) -> list[str]:
        keys: list[str] = []
        for site in self._sites():
            rows = (
                self._reader(site)
                .db.execute("SELECT idx FROM components WHERE n_examples >= ?", (min_examples,))
                .fetchall()
            )
            keys.extend(f"{site}:{idx}" for (idx,) in rows)
        return keys

    def get_component_densities(self, min_examples: int) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for site in self._sites():
            rows = (
                self._reader(site)
                .db.execute(
                    "SELECT idx, firing_density FROM components WHERE n_examples >= ?",
                    (min_examples,),
                )
                .fetchall()
            )
            out.extend((f"{site}:{idx}", density) for idx, density in rows)
        return out

    # -- Correlations & token stats (tensor data) ------------------------------

    def get_correlations(self) -> CorrelationStorage | None:
        path = self._dir / "component_correlations.npz"
        if not path.exists():
            return None
        return CorrelationStorage.load(path)

    def get_token_stats(self) -> TokenStatsStorage | None:
        path = self._dir / "token_stats.npz"
        if not path.exists():
            return None
        return TokenStatsStorage.load(path)

    # -- Eval scores + prompts (e.g. intruder) ---------------------------------

    def save_score(self, component_key: str, score_type: str, score: float, details: str) -> None:
        self._db.save_score(component_key, score_type, score, details)

    def get_scores(self, score_type: str) -> dict[str, float]:
        return self._db.get_scores(score_type)

    def save_intruder_prompt(self, trial_key: str, prompt: str) -> None:
        self._db.save_intruder_prompt(trial_key, prompt)
