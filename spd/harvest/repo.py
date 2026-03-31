"""Harvest data repository.

Owns SPD_OUT_DIR/harvest/<decomposition_id>/ and provides read/write access to all
harvest artifacts. No in-memory caching -- reads go through on every call.
Component data backed by SQLite; correlations and token stats remain as .pt files.

Layout: harvest/<decomposition_id>/h-YYYYMMDD_HHMMSS/{harvest.db, *.pt}
"""

from pathlib import Path

from spd.harvest.config import HarvestConfig
from spd.harvest.db import HarvestDB
from spd.harvest.harvester import Harvester
from spd.harvest.schemas import (
    ComponentData,
    ComponentSummary,
    get_harvest_dir,
)
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage
from spd.log import logger


class HarvestRepo:
    """Access to harvest data for a single harvest subrun of a decomposition."""

    def __init__(self, decomposition_id: str, subrun_id: str, readonly: bool) -> None:
        self.subrun_id = subrun_id
        self._dir = get_harvest_dir(decomposition_id) / subrun_id
        self._db = HarvestDB(self._dir / "harvest.db", readonly=readonly)

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
            [
                d
                for d in decomposition_subruns_dir.iterdir()
                if d.is_dir() and d.name.startswith("h-")
            ],
            key=lambda d: d.name,
        )
        if not subrun_candidates:
            return None

        subrun_dir = subrun_candidates[-1]

        db_path = subrun_dir / "harvest.db"
        if not db_path.exists():
            logger.info(f"No harvest data found for {decomposition_id}")
            return None

        logger.info(f"Opening harvest data for {decomposition_id} from {subrun_dir}")
        subrun_id = subrun_dir.name

        return cls(decomposition_id=decomposition_id, subrun_id=subrun_id, readonly=readonly)

    @staticmethod
    def save_results(harvester: Harvester, config: HarvestConfig, output_dir: Path) -> None:
        """Build and save all harvest results to disk.

        Components are streamed to the DB one at a time to avoid holding all
        ComponentData objects in memory simultaneously.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Building and saving component results...")
        db_path = output_dir / "harvest.db"
        db = HarvestDB(db_path)
        db.save_config(config)
        components_iter = harvester.build_results(pmi_top_k_tokens=config.pmi_token_top_k)
        n_saved = db.save_components_iter(components_iter)
        db.close()
        logger.info(f"Saved {n_saved} components to {db_path}")

        component_keys = harvester.component_keys

        correlations = CorrelationStorage(
            component_keys=component_keys,
            count_i=harvester.firing_counts.long().cpu(),
            count_ij=harvester.cooccurrence_counts.long().cpu(),
            count_total=harvester.total_tokens_processed,
        )
        correlations.save(output_dir / "component_correlations.pt")

        token_stats = TokenStatsStorage(
            component_keys=component_keys,
            vocab_size=harvester.vocab_size,
            n_tokens=harvester.total_tokens_processed,
            input_counts=harvester.input_cooccurrence.cpu(),
            input_totals=harvester.input_marginals.float().cpu(),
            output_counts=harvester.output_cooccurrence.cpu(),
            output_totals=harvester.output_marginals.cpu(),
            firing_counts=harvester.firing_counts.cpu(),
        )
        token_stats.save(output_dir / "token_stats.pt")

    # -- Provenance ------------------------------------------------------------

    def get_config(self) -> dict[str, object]:
        return self._db.get_config_dict()

    def get_component_count(self) -> int:
        return self._db.get_component_count()

    # -- Activation contexts ---------------------------------------------------

    def get_summary(self) -> dict[str, ComponentSummary]:
        return self._db.get_summary()

    def get_component(self, component_key: str) -> ComponentData | None:
        return self._db.get_component(component_key)

    def get_components_bulk(self, component_keys: list[str]) -> dict[str, ComponentData]:
        return self._db.get_components_bulk(component_keys)

    def get_all_components(self) -> list[ComponentData]:
        return self._db.get_all_components()

    def get_component_keys(self) -> list[str]:
        return self._db.get_component_keys()

    def get_eligible_component_keys(self, min_examples: int) -> list[str]:
        return self._db.get_eligible_component_keys(min_examples)

    # -- Correlations & token stats (tensor data) ------------------------------

    def get_correlations(self) -> CorrelationStorage | None:
        path = self._dir / "component_correlations.pt"
        if not path.exists():
            return None
        return CorrelationStorage.load(path)

    def get_token_stats(self) -> TokenStatsStorage | None:
        path = self._dir / "token_stats.pt"
        if not path.exists():
            return None
        return TokenStatsStorage.load(path)

    # -- Eval scores (e.g. intruder) -------------------------------------------

    def save_score(self, component_key: str, score_type: str, score: float, details: str) -> None:
        self._db.save_score(component_key, score_type, score, details)

    def get_scores(self, score_type: str) -> dict[str, float]:
        return self._db.get_scores(score_type)
