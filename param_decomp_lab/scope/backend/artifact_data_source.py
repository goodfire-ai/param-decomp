"""ScopeDataSource over the real artifact store (see param_decomp_lab/scope/artifacts.py).

Reads every run under `PARAM_DECOMP_OUT_DIR/runs/*/scope/`. Newest complete subrun of a
site wins. Listing queries run against the site.db indexes with labels ATTACHed; detail
reads are one mmap seek + one row + detokenization via a cached AppTokenizer.
"""

import json
import sqlite3
from functools import lru_cache
from pathlib import Path

from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.scope.artifacts import SiteShardReader, find_subruns, scope_dir
from param_decomp_lab.scope.backend.data_source import (
    ActivationExample,
    CatalogResponse,
    ComponentDetail,
    ComponentLabel,
    ComponentListResponse,
    ComponentRow,
    CurvePoint,
    RunEntry,
    ScopeNotFoundError,
    SiteCurve,
    SiteEntry,
    SortKey,
    SubrunEntry,
)
from param_decomp_lab.tokenizer_display import AppTokenizer

_SORT_SQL: dict[str, str] = {
    "density": "c.firing_density DESC",
    "max_act": "c.max_act DESC",
    "unlabeled_first": "(l.label IS NULL) DESC, c.firing_density DESC",
}


@lru_cache(maxsize=64)
def _reader(subrun_dir: str) -> SiteShardReader:
    return SiteShardReader(Path(subrun_dir))


@lru_cache(maxsize=64)
def _rank_order(subrun_dir: str) -> tuple[list[int], dict[int, int], list[float]]:
    """(idx_by_rank, rank_by_idx, mean_ci_by_rank), ordered by mean_ci DESC then idx."""
    rows = (
        _reader(subrun_dir)
        .db.execute("SELECT idx, mean_ci FROM components ORDER BY mean_ci DESC, idx ASC")
        .fetchall()
    )
    idx_by_rank = [idx for idx, _ in rows]
    return idx_by_rank, {idx: r for r, idx in enumerate(idx_by_rank)}, [ci for _, ci in rows]


@lru_cache(maxsize=8)
def _tokenizer(name: str) -> AppTokenizer:
    return AppTokenizer.from_pretrained(name)


def _labels_db_path(run_id: str) -> Path:
    return scope_dir(run_id) / "labels.db"


class ArtifactDataSource:
    def _latest_subrun(self, run_id: str, site: str) -> SiteShardReader:
        subruns = find_subruns(run_id, site)
        if not subruns:
            raise ScopeNotFoundError(f"no published subrun for {run_id}/{site}")
        return _reader(str(subruns[-1]))

    def _query_conn(self, run_id: str, site: str) -> tuple[sqlite3.Connection, SiteShardReader]:
        """site.db connection with the run's labels.db attached (if it exists)."""
        reader = self._latest_subrun(run_id, site)
        conn = reader.db
        (attached,) = conn.execute(
            "SELECT COUNT(*) FROM pragma_database_list WHERE name='lbl'"
        ).fetchone()
        if not attached:
            labels = _labels_db_path(run_id)
            if labels.exists():
                conn.execute("ATTACH DATABASE ? AS lbl", (f"file:{labels}?mode=ro",))
            else:
                conn.execute("ATTACH DATABASE ':memory:' AS lbl")
                conn.execute("CREATE TABLE lbl.labels (site TEXT, component_idx INT, label TEXT)")
        return conn, reader

    # -- ScopeDataSource -------------------------------------------------------

    def catalog(self) -> CatalogResponse:
        runs = []
        runs_dir = PARAM_DECOMP_OUT_DIR / "runs"
        for run_dir in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
            sdir = run_dir / "scope"
            if not sdir.is_dir():
                continue
            labeled_by_site = self._labeled_counts(run_dir.name)
            sites = []
            for site_dir in sorted(d for d in sdir.iterdir() if d.is_dir()):
                published = find_subruns(run_dir.name, site_dir.name)
                in_flight = sorted(d.name for d in site_dir.iterdir() if d.name.startswith(".tmp-"))
                if not published and not in_flight:
                    continue
                subruns = [
                    SubrunEntry(
                        subrun_id=p.name,
                        status="present",
                        n_batches=0,
                        progress=1.0,
                    )
                    for p in published
                ] + [
                    SubrunEntry(subrun_id=t, status="in_flight", n_batches=0, progress=0.0)
                    for t in in_flight
                ]
                n_components = _reader(str(published[-1])).meta.n_components if published else 0
                sites.append(
                    SiteEntry(
                        site=site_dir.name,
                        n_components=n_components,
                        n_labeled=labeled_by_site.get(site_dir.name, 0),
                        subruns=subruns,
                    )
                )
            if sites:
                runs.append(RunEntry(run_id=run_dir.name, sites=sites))
        return CatalogResponse(runs=runs)

    def _labeled_counts(self, run_id: str) -> dict[str, int]:
        path = _labels_db_path(run_id)
        if not path.exists():
            return {}
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        counts = dict(conn.execute("SELECT site, COUNT(*) FROM labels GROUP BY site"))
        conn.close()
        return counts

    def list_components(
        self, run_id: str, site: str, sort: SortKey, page: int, page_size: int, q: str
    ) -> ComponentListResponse:
        conn, _ = self._query_conn(run_id, site)
        join = "LEFT JOIN lbl.labels l ON l.site = :site AND l.component_idx = c.idx"
        where = "WHERE l.label LIKE :pat" if q else ""
        params = {"site": site, "pat": f"%{q}%", "limit": page_size, "offset": page * page_size}
        (total,) = conn.execute(
            f"SELECT COUNT(*) FROM components c {join} {where}", params
        ).fetchone()
        rows = conn.execute(
            f"""SELECT c.idx, c.firing_density, c.max_act, l.label
                FROM components c {join} {where}
                ORDER BY {_SORT_SQL[sort]} LIMIT :limit OFFSET :offset""",
            params,
        ).fetchall()
        items = [
            ComponentRow(idx=idx, density=density, max_act=max_act, label=label)
            for idx, density, max_act, label in rows
        ]
        return ComponentListResponse(total=total, page=page, items=items)

    def site_curve(self, run_id: str, site: str) -> SiteCurve:
        subruns = find_subruns(run_id, site)
        if not subruns:
            raise ScopeNotFoundError(f"no published subrun for {run_id}/{site}")
        idx_by_rank, _, mean_ci_by_rank = _rank_order(str(subruns[-1]))
        n = len(idx_by_rank)
        sample = sorted(
            {
                0,
                n - 1,
                *(round(1.6**k) for k in range(1, 200) if 1.6**k < n - 1),
                *range(0, n, max(1, n // 360)),
            }
        )
        points = [
            CurvePoint(rank=r, idx=idx_by_rank[r], mean_ci=mean_ci_by_rank[r]) for r in sample
        ]
        return SiteCurve(total=n, points=points)

    def component_detail(self, run_id: str, site: str, idx: int) -> ComponentDetail:
        conn, reader = self._query_conn(run_id, site)
        row = conn.execute(
            """SELECT c.firing_density, c.max_act, c.mean_ci, c.input_pmi, c.output_pmi,
                      l.label, l.model, l.cost_usd, l.created_at
               FROM components c
               LEFT JOIN lbl.labels l ON l.site = :site AND l.component_idx = c.idx
               WHERE c.idx = :idx""",
            {"site": site, "idx": idx},
        ).fetchone()
        if row is None:
            raise ScopeNotFoundError(f"no component {idx} in {run_id}/{site}")
        (
            density,
            max_act,
            mean_ci,
            in_pmi_json,
            out_pmi_json,
            label_text,
            model,
            cost_usd,
            created_at,
        ) = row

        tok = _tokenizer(reader.meta.tokenizer_name)
        examples = reader.examples(idx)
        rendered = [
            ActivationExample(
                tokens=tok.get_spans([int(t) for t in examples.token_ids[i]]),
                acts=[float(a) for a in examples.act[i]],
                cis=[float(c) for c in examples.ci[i]],
                max_act=float(examples.ci[i].max()),
            )
            for i in range(examples.token_ids.shape[0])
        ]

        def pmi_pairs(raw: str) -> list[tuple[str, float]]:
            return [(tok.get_tok_display(t), p) for t, p in json.loads(raw)]

        label = (
            ComponentLabel(
                text=label_text,
                model=model,
                cost_usd=cost_usd if cost_usd is not None else 0.0,
                created_at=created_at,
            )
            if label_text is not None
            else None
        )
        idx_by_rank, rank_by_idx, _ = _rank_order(str(find_subruns(run_id, site)[-1]))
        rank = rank_by_idx[idx]
        return ComponentDetail(
            idx=idx,
            rank=rank,
            prev_idx=idx_by_rank[rank - 1] if rank > 0 else None,
            next_idx=idx_by_rank[rank + 1] if rank + 1 < len(idx_by_rank) else None,
            density=density,
            max_act=max_act,
            mean_ci=mean_ci,
            label=label,
            input_pmi=pmi_pairs(in_pmi_json),
            output_pmi=pmi_pairs(out_pmi_json),
            examples=rendered,
        )

    def create_label(self, run_id: str, site: str, idx: int) -> ComponentLabel:
        raise NotImplementedError(
            f"label-on-demand is not wired for the artifact source yet (requested "
            f"{run_id}/{site}/{idx}); labels are imported from legacy autointerp runs"
        )
