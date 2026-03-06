"""SQLite database for prompt attribution data.

Stores runs, prompts, and attribution graphs.
Activation contexts and correlations are stored in the harvest pipeline output at
SPD_OUT_DIR/harvest/<run_id>/.
Interpretations are stored separately at SPD_OUT_DIR/autointerp/<run_id>/.
"""

import fcntl
import hashlib
import io
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel

from spd.app.backend.compute import Edge, Node
from spd.app.backend.optim_cis import (
    CELossConfig,
    KLLossConfig,
    LogitLossConfig,
    MaskType,
    PositionalLossConfig,
)
from spd.settings import SPD_OUT_DIR

GraphType = Literal["standard", "optimized", "manual"]

_DEFAULT_DB_PATH = SPD_OUT_DIR / "app" / "prompt_attr.db"


def get_default_db_path() -> Path:
    """Get the default database path.

    Checks env vars in order:
    1. SPD_INVESTIGATION_DIR - investigation mode, db at dir/app.db
    2. SPD_APP_DB_PATH - explicit override
    3. Default: SPD_OUT_DIR/app/prompt_attr.db
    """
    investigation_dir = os.environ.get("SPD_INVESTIGATION_DIR")
    if investigation_dir:
        return Path(investigation_dir) / "app.db"
    env_path = os.environ.get("SPD_APP_DB_PATH")
    if env_path:
        return Path(env_path)
    return _DEFAULT_DB_PATH


class Run(BaseModel):
    """A run record."""

    id: int
    wandb_path: str


class PromptRecord(BaseModel):
    """A stored prompt record containing token IDs."""

    id: int
    run_id: int
    token_ids: list[int]
    is_custom: bool = False


class PgdConfig(BaseModel):
    n_steps: int
    step_size: float


class OptimizationParams(BaseModel):
    """Optimization parameters that affect graph computation."""

    imp_min_coeff: float
    steps: int
    pnorm: float
    beta: float
    mask_type: MaskType
    loss: PositionalLossConfig
    pgd: PgdConfig | None = None
    # Computed metrics (persisted for display on reload)
    ci_masked_label_prob: float | None = None
    stoch_masked_label_prob: float | None = None
    adv_pgd_label_prob: float | None = None


class StoredGraph(BaseModel):
    """A stored attribution graph."""

    model_config = {"arbitrary_types_allowed": True}

    id: int = -1  # -1 for unsaved graphs, set by DB on save
    graph_type: GraphType = "standard"

    # Core graph data (all types)
    edges: list[Edge]
    edges_abs: list[Edge] | None = (
        None  # absolute-target variant (∂|y|/∂x · x), None for old graphs
    )
    ci_masked_out_logits: torch.Tensor  # [seq, vocab]
    target_out_logits: torch.Tensor  # [seq, vocab]
    node_ci_vals: dict[str, float]  # layer:seq:c_idx -> ci_val (required for all graphs)
    node_subcomp_acts: dict[str, float] = {}  # layer:seq:c_idx -> subcomp act (v_i^T @ a)

    # Optimized-specific (None for other types)
    optimization_params: OptimizationParams | None = None

    # Manual-specific (None for other types)
    included_nodes: list[str] | None = None  # Nodes included in this graph


class InterventionRunRecord(BaseModel):
    """A stored intervention run."""

    id: int
    graph_id: int
    selected_nodes: list[str]  # node keys that were selected
    result_json: str  # JSON-encoded InterventionResult
    created_at: str


class ForkedInterventionRunRecord(BaseModel):
    """A forked intervention run with modified tokens (currently unused)."""

    id: int
    intervention_run_id: int
    token_replacements: list[tuple[int, int]]  # [(seq_pos, new_token_id), ...]
    result_json: str
    created_at: str


class PromptAttrDB:
    """SQLite database for storing and querying prompt attribution data.

    Schema:
    - runs: One row per SPD run (keyed by wandb_path)
    - prompts: One row per stored prompt (token sequence), keyed by run_id
    - graphs: Attribution graphs for prompts

    Attribution graphs are computed on-demand and cached.
    """

    def __init__(self, db_path: Path | None = None, check_same_thread: bool = True):
        self.db_path = db_path or get_default_db_path()
        self._lock_path = self.db_path.with_suffix(".db.lock")
        self._check_same_thread = check_same_thread
        self._conn: sqlite3.Connection | None = None

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "PromptAttrDB":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @contextmanager
    def _write_lock(self):
        """Acquire an exclusive file lock for write operations (NFS-safe)."""
        with open(self._lock_path, "w") as lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

    # -------------------------------------------------------------------------
    # Schema initialization
    # -------------------------------------------------------------------------

    def init_schema(self) -> None:
        """Initialize the database schema. Safe to call multiple times."""
        conn = self._get_conn()
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY,
                wandb_path TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES runs(id),
                token_ids TEXT NOT NULL,
                context_length INTEGER NOT NULL,
                is_custom INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_prompts_run_id
                ON prompts(run_id);

            CREATE TABLE IF NOT EXISTS graphs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id INTEGER NOT NULL REFERENCES prompts(id),
                graph_type TEXT NOT NULL,  -- 'standard', 'optimized', 'manual'

                -- Optimization params (NULL for non-optimized graphs)
                imp_min_coeff REAL,
                steps INTEGER,
                pnorm REAL,
                beta REAL,
                mask_type TEXT,
                loss_config TEXT,  -- JSON: {type: "ce"|"kl", coeff, position, label_token?}
                loss_config_hash TEXT,  -- SHA256 hash for uniqueness indexing
                adv_pgd_n_steps INTEGER,
                adv_pgd_step_size REAL,

                -- Optimization metrics (NULL for non-optimized graphs)
                ci_masked_label_prob REAL,
                stoch_masked_label_prob REAL,
                adv_pgd_label_prob REAL,

                -- Manual graph params (NULL for non-manual graphs)
                included_nodes TEXT,  -- JSON array of node keys in this graph
                included_nodes_hash TEXT,  -- SHA256 hash of sorted JSON for uniqueness

                -- The actual graph data (JSON)
                edges_data TEXT NOT NULL,
                -- Absolute-target edges (∂|y|/∂x · x), NULL for old graphs
                edges_data_abs TEXT,
                -- Node CI values: "layer:seq:c_idx" -> ci_val (required for all graphs)
                node_ci_vals TEXT NOT NULL,
                -- Node subcomponent activations: "layer:seq:c_idx" -> v_i^T @ a
                node_subcomp_acts TEXT NOT NULL DEFAULT '{}',
                -- Output logits: torch.save({ci_masked, target}) as blob
                output_logits BLOB NOT NULL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- One standard graph per prompt
            CREATE UNIQUE INDEX IF NOT EXISTS idx_graphs_standard
                ON graphs(prompt_id)
                WHERE graph_type = 'standard';

            -- One optimized graph per unique parameter combination
            CREATE UNIQUE INDEX IF NOT EXISTS idx_graphs_optimized
                ON graphs(prompt_id, imp_min_coeff, steps, pnorm, beta, mask_type, loss_config_hash, adv_pgd_n_steps, adv_pgd_step_size)
                WHERE graph_type = 'optimized';

            -- One manual graph per unique node set (using hash for reliable uniqueness)
            CREATE UNIQUE INDEX IF NOT EXISTS idx_graphs_manual
                ON graphs(prompt_id, included_nodes_hash)
                WHERE graph_type = 'manual';

            CREATE INDEX IF NOT EXISTS idx_graphs_prompt
                ON graphs(prompt_id);

            CREATE TABLE IF NOT EXISTS intervention_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_id INTEGER NOT NULL REFERENCES graphs(id),
                selected_nodes TEXT NOT NULL,  -- JSON array of node keys
                result TEXT NOT NULL,  -- JSON InterventionResult
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_intervention_runs_graph
                ON intervention_runs(graph_id);

            CREATE TABLE IF NOT EXISTS forked_intervention_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intervention_run_id INTEGER NOT NULL REFERENCES intervention_runs(id) ON DELETE CASCADE,
                token_replacements TEXT NOT NULL,  -- JSON array of [seq_pos, new_token_id] tuples
                result TEXT NOT NULL,  -- JSON InterventionResponse
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_forked_intervention_runs_parent
                ON forked_intervention_runs(intervention_run_id);
        """)

        # Migration: add edges_data_abs column if missing (backwards compat with existing DBs)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(graphs)").fetchall()}
        if "edges_data_abs" not in columns:
            conn.execute("ALTER TABLE graphs ADD COLUMN edges_data_abs TEXT")

        conn.commit()

    # -------------------------------------------------------------------------
    # Run operations
    # -------------------------------------------------------------------------

    def create_run(self, wandb_path: str) -> int:
        """Create a new run. Returns the run ID."""
        with self._write_lock():
            conn = self._get_conn()
            cursor = conn.execute(
                "INSERT INTO runs (wandb_path) VALUES (?)",
                (wandb_path,),
            )
            conn.commit()
            run_id = cursor.lastrowid
            assert run_id is not None
            return run_id

    def get_run_by_wandb_path(self, wandb_path: str) -> Run | None:
        """Get a run by its wandb path."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, wandb_path FROM runs WHERE wandb_path = ?",
            (wandb_path,),
        ).fetchone()
        if row is None:
            return None
        return Run(id=row["id"], wandb_path=row["wandb_path"])

    def get_run(self, run_id: int) -> Run | None:
        """Get a run by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, wandb_path FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return Run(id=row["id"], wandb_path=row["wandb_path"])

    # -------------------------------------------------------------------------
    # Prompt operations
    # -------------------------------------------------------------------------

    def find_prompt_by_token_ids(
        self,
        run_id: int,
        token_ids: list[int],
        context_length: int,
    ) -> int | None:
        """Find an existing prompt with the same token_ids."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id FROM prompts WHERE run_id = ? AND token_ids = ? AND context_length = ?",
            (run_id, json.dumps(token_ids), context_length),
        ).fetchone()
        return row[0] if row else None

    def add_custom_prompt(
        self,
        run_id: int,
        token_ids: list[int],
        context_length: int,
    ) -> int:
        """Add a custom prompt to the database, or return existing if duplicate.

        Args:
            run_id: The run this prompt belongs to.
            token_ids: The token IDs for the prompt.
            context_length: The context length setting.

        Returns:
            The prompt ID (existing or newly created).
        """
        with self._write_lock():
            existing_id = self.find_prompt_by_token_ids(run_id, token_ids, context_length)
            if existing_id is not None:
                return existing_id

            conn = self._get_conn()
            cursor = conn.execute(
                "INSERT INTO prompts (run_id, token_ids, context_length, is_custom) VALUES (?, ?, ?, 1)",
                (run_id, json.dumps(token_ids), context_length),
            )
            prompt_id = cursor.lastrowid
            assert prompt_id is not None
            conn.commit()
            return prompt_id

    def get_prompt(self, prompt_id: int) -> PromptRecord | None:
        """Get a prompt by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, run_id, token_ids, is_custom FROM prompts WHERE id = ?",
            (prompt_id,),
        ).fetchone()
        if row is None:
            return None

        return PromptRecord(
            id=row["id"],
            run_id=row["run_id"],
            token_ids=json.loads(row["token_ids"]),
            is_custom=bool(row["is_custom"]),
        )

    def get_prompt_count(self, run_id: int, context_length: int) -> int:
        """Get total number of prompts for a run with a specific context length."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM prompts WHERE run_id = ? AND context_length = ?",
            (run_id, context_length),
        ).fetchone()
        return row["cnt"]

    def get_all_prompt_ids(self, run_id: int, context_length: int) -> list[int]:
        """Get all prompt IDs for a run with a specific context length."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id FROM prompts WHERE run_id = ? AND context_length = ? ORDER BY id",
            (run_id, context_length),
        ).fetchall()
        return [row["id"] for row in rows]

    # -------------------------------------------------------------------------
    # Graph operations
    # -------------------------------------------------------------------------

    def save_graph(
        self,
        prompt_id: int,
        graph: StoredGraph,
    ) -> int:
        """Save a computed graph for a prompt.

        Args:
            prompt_id: The prompt ID.
            graph: The graph to save.

        Returns:
            The database ID of the saved graph.
        """
        conn = self._get_conn()

        def _node_to_dict(n: Node) -> dict[str, str | int]:
            return {
                "layer": n.layer,
                "seq_pos": n.seq_pos,
                "component_idx": n.component_idx,
            }

        def _edges_to_json(edges: list[Edge]) -> str:
            return json.dumps(
                [
                    {
                        "source": _node_to_dict(e.source),
                        "target": _node_to_dict(e.target),
                        "strength": e.strength,
                        "is_cross_seq": e.is_cross_seq,
                    }
                    for e in edges
                ]
            )

        edges_json = _edges_to_json(graph.edges)
        edges_abs_json = _edges_to_json(graph.edges_abs) if graph.edges_abs is not None else None
        buf = io.BytesIO()
        logits_dict: dict[str, torch.Tensor] = {
            "ci_masked": graph.ci_masked_out_logits,
            "target": graph.target_out_logits,
        }
        torch.save(logits_dict, buf)
        output_logits_blob = buf.getvalue()
        node_ci_vals_json = json.dumps(graph.node_ci_vals)
        node_subcomp_acts_json = json.dumps(graph.node_subcomp_acts)

        # Extract optimization-specific values (NULL for non-optimized graphs)
        imp_min_coeff = None
        steps = None
        pnorm = None
        beta = None
        mask_type = None
        loss_config_json: str | None = None
        loss_config_hash: str | None = None
        adv_pgd_n_steps = None
        adv_pgd_step_size = None
        ci_masked_label_prob = None
        stoch_masked_label_prob = None
        adv_pgd_label_prob = None

        if graph.optimization_params:
            imp_min_coeff = graph.optimization_params.imp_min_coeff
            steps = graph.optimization_params.steps
            pnorm = graph.optimization_params.pnorm
            beta = graph.optimization_params.beta
            mask_type = graph.optimization_params.mask_type
            loss_config_json = graph.optimization_params.loss.model_dump_json()
            loss_config_hash = hashlib.sha256(loss_config_json.encode()).hexdigest()
            adv_pgd_n_steps = (
                graph.optimization_params.pgd.n_steps if graph.optimization_params.pgd else None
            )
            adv_pgd_step_size = (
                graph.optimization_params.pgd.step_size if graph.optimization_params.pgd else None
            )
            ci_masked_label_prob = graph.optimization_params.ci_masked_label_prob
            stoch_masked_label_prob = graph.optimization_params.stoch_masked_label_prob
            adv_pgd_label_prob = graph.optimization_params.adv_pgd_label_prob

        # Extract manual-specific values (NULL for non-manual graphs)
        # Sort included_nodes and compute hash for reliable uniqueness
        included_nodes_json: str | None = None
        included_nodes_hash: str | None = None
        if graph.included_nodes:
            included_nodes_json = json.dumps(sorted(graph.included_nodes))
            included_nodes_hash = hashlib.sha256(included_nodes_json.encode()).hexdigest()

        with self._write_lock():
            try:
                cursor = conn.execute(
                    """INSERT INTO graphs
                       (prompt_id, graph_type,
                        imp_min_coeff, steps, pnorm, beta, mask_type,
                        loss_config, loss_config_hash,
                        adv_pgd_n_steps, adv_pgd_step_size,
                        ci_masked_label_prob, stoch_masked_label_prob, adv_pgd_label_prob,
                        included_nodes, included_nodes_hash,
                        edges_data, edges_data_abs, output_logits, node_ci_vals, node_subcomp_acts)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        prompt_id,
                        graph.graph_type,
                        imp_min_coeff,
                        steps,
                        pnorm,
                        beta,
                        mask_type,
                        loss_config_json,
                        loss_config_hash,
                        adv_pgd_n_steps,
                        adv_pgd_step_size,
                        ci_masked_label_prob,
                        stoch_masked_label_prob,
                        adv_pgd_label_prob,
                        included_nodes_json,
                        included_nodes_hash,
                        edges_json,
                        edges_abs_json,
                        output_logits_blob,
                        node_ci_vals_json,
                        node_subcomp_acts_json,
                    ),
                )
                conn.commit()
                graph_id = cursor.lastrowid
                assert graph_id is not None
                return graph_id
            except sqlite3.IntegrityError as e:
                match graph.graph_type:
                    case "standard":
                        raise ValueError(
                            f"Standard graph already exists for prompt_id={prompt_id}. "
                            "Use get_graphs() to retrieve existing graph or delete it first."
                        ) from e
                    case "optimized":
                        raise ValueError(
                            f"Optimized graph with same parameters already exists for prompt_id={prompt_id}."
                        ) from e
                    case "manual":
                        conn.rollback()
                        row = conn.execute(
                            """SELECT id FROM graphs
                               WHERE prompt_id = ? AND graph_type = 'manual'
                               AND included_nodes_hash = ?""",
                            (prompt_id, included_nodes_hash),
                        ).fetchone()
                        if row:
                            return row["id"]
                        raise ValueError(
                            "A manual graph with the same nodes already exists."
                        ) from e

    def _row_to_stored_graph(self, row: sqlite3.Row) -> StoredGraph:
        """Convert a database row to a StoredGraph."""

        def _node_from_dict(d: dict[str, str | int]) -> Node:
            return Node(
                layer=str(d["layer"]),
                seq_pos=int(d["seq_pos"]),
                component_idx=int(d["component_idx"]),
            )

        def _parse_edges(data: str) -> list[Edge]:
            return [
                Edge(
                    source=_node_from_dict(e["source"]),
                    target=_node_from_dict(e["target"]),
                    strength=float(e["strength"]),
                    is_cross_seq=bool(e["is_cross_seq"]),
                )
                for e in json.loads(data)
            ]

        edges = _parse_edges(row["edges_data"])
        edges_abs = _parse_edges(row["edges_data_abs"]) if row["edges_data_abs"] else None
        logits_data = torch.load(io.BytesIO(row["output_logits"]), weights_only=True)
        ci_masked_out_logits: torch.Tensor = logits_data["ci_masked"]
        target_out_logits: torch.Tensor = logits_data["target"]
        node_ci_vals: dict[str, float] = json.loads(row["node_ci_vals"])
        node_subcomp_acts: dict[str, float] = json.loads(row["node_subcomp_acts"] or "{}")

        opt_params: OptimizationParams | None = None
        if row["graph_type"] == "optimized":
            loss_config_data = json.loads(row["loss_config"])
            loss_type = loss_config_data["type"]
            assert loss_type in ("ce", "kl", "logit"), f"Unknown loss type: {loss_type}"
            loss_config: PositionalLossConfig
            match loss_type:
                case "ce":
                    loss_config = CELossConfig(**loss_config_data)
                case "kl":
                    loss_config = KLLossConfig(**loss_config_data)
                case "logit":
                    loss_config = LogitLossConfig(**loss_config_data)
            pgd = None
            if row["adv_pgd_n_steps"] is not None:
                pgd = PgdConfig(n_steps=row["adv_pgd_n_steps"], step_size=row["adv_pgd_step_size"])
            opt_params = OptimizationParams(
                imp_min_coeff=row["imp_min_coeff"],
                steps=row["steps"],
                pnorm=row["pnorm"],
                beta=row["beta"],
                mask_type=row["mask_type"],
                loss=loss_config,
                pgd=pgd,
                ci_masked_label_prob=row["ci_masked_label_prob"],
                stoch_masked_label_prob=row["stoch_masked_label_prob"],
                adv_pgd_label_prob=row["adv_pgd_label_prob"],
            )

        # Parse manual-specific fields
        included_nodes: list[str] | None = None
        if row["included_nodes"]:
            included_nodes = json.loads(row["included_nodes"])

        return StoredGraph(
            id=row["id"],
            graph_type=row["graph_type"],
            edges=edges,
            edges_abs=edges_abs,
            ci_masked_out_logits=ci_masked_out_logits,
            target_out_logits=target_out_logits,
            node_ci_vals=node_ci_vals,
            node_subcomp_acts=node_subcomp_acts,
            optimization_params=opt_params,
            included_nodes=included_nodes,
        )

    def get_graphs(self, prompt_id: int) -> list[StoredGraph]:
        """Retrieve all stored graphs for a prompt.

        Args:
            prompt_id: The prompt ID.

        Returns:
            List of stored graphs (standard, optimized, and manual).
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, graph_type, edges_data, edges_data_abs, output_logits, node_ci_vals,
                      node_subcomp_acts, imp_min_coeff, steps, pnorm, beta, mask_type,
                      loss_config, adv_pgd_n_steps, adv_pgd_step_size, included_nodes,
                      ci_masked_label_prob, stoch_masked_label_prob, adv_pgd_label_prob
               FROM graphs
               WHERE prompt_id = ?
               ORDER BY
                   CASE graph_type WHEN 'standard' THEN 0 WHEN 'optimized' THEN 1 ELSE 2 END,
                   created_at""",
            (prompt_id,),
        ).fetchall()
        return [self._row_to_stored_graph(row) for row in rows]

    def get_graph(self, graph_id: int) -> tuple[StoredGraph, int] | None:
        """Retrieve a single graph by its ID. Returns (graph, prompt_id) or None."""
        conn = self._get_conn()
        row = conn.execute(
            """SELECT id, prompt_id, graph_type, edges_data, edges_data_abs, output_logits,
                      node_ci_vals, node_subcomp_acts, imp_min_coeff, steps, pnorm, beta,
                      mask_type, loss_config, adv_pgd_n_steps, adv_pgd_step_size,
                      included_nodes, ci_masked_label_prob, stoch_masked_label_prob,
                      adv_pgd_label_prob
               FROM graphs
               WHERE id = ?""",
            (graph_id,),
        ).fetchone()
        if row is None:
            return None
        return (self._row_to_stored_graph(row), row["prompt_id"])

    def delete_prompt(self, prompt_id: int) -> None:
        """Delete a prompt and all its graphs, intervention runs, and forked runs."""
        with self._write_lock():
            conn = self._get_conn()
            graph_ids_query = "SELECT id FROM graphs WHERE prompt_id = ?"
            intervention_ids_query = (
                f"SELECT id FROM intervention_runs WHERE graph_id IN ({graph_ids_query})"
            )
            conn.execute(
                f"DELETE FROM forked_intervention_runs WHERE intervention_run_id IN ({intervention_ids_query})",
                (prompt_id,),
            )
            conn.execute(
                f"DELETE FROM intervention_runs WHERE graph_id IN ({graph_ids_query})",
                (prompt_id,),
            )
            conn.execute("DELETE FROM graphs WHERE prompt_id = ?", (prompt_id,))
            conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
            conn.commit()

    def delete_graphs_for_prompt(self, prompt_id: int) -> int:
        """Delete all graphs for a prompt. Returns the number of deleted rows."""
        with self._write_lock():
            conn = self._get_conn()
            cursor = conn.execute("DELETE FROM graphs WHERE prompt_id = ?", (prompt_id,))
            conn.commit()
            return cursor.rowcount

    def delete_graphs_for_run(self, run_id: int) -> int:
        """Delete all graphs for all prompts in a run. Returns the number of deleted rows."""
        with self._write_lock():
            conn = self._get_conn()
            cursor = conn.execute(
                """DELETE FROM graphs
                   WHERE prompt_id IN (SELECT id FROM prompts WHERE run_id = ?)""",
                (run_id,),
            )
            conn.commit()
            return cursor.rowcount

    # -------------------------------------------------------------------------
    # Intervention run operations
    # -------------------------------------------------------------------------

    def save_intervention_run(
        self,
        graph_id: int,
        selected_nodes: list[str],
        result_json: str,
    ) -> int:
        """Save an intervention run.

        Args:
            graph_id: The graph ID this run belongs to.
            selected_nodes: List of node keys that were selected.
            result_json: JSON-encoded InterventionResult.

        Returns:
            The intervention run ID.
        """
        with self._write_lock():
            conn = self._get_conn()
            cursor = conn.execute(
                """INSERT INTO intervention_runs (graph_id, selected_nodes, result)
                   VALUES (?, ?, ?)""",
                (graph_id, json.dumps(selected_nodes), result_json),
            )
            conn.commit()
            run_id = cursor.lastrowid
            assert run_id is not None
            return run_id

    def get_intervention_runs(self, graph_id: int) -> list[InterventionRunRecord]:
        """Get all intervention runs for a graph.

        Args:
            graph_id: The graph ID.

        Returns:
            List of intervention run records, ordered by creation time.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, graph_id, selected_nodes, result, created_at
               FROM intervention_runs
               WHERE graph_id = ?
               ORDER BY created_at""",
            (graph_id,),
        ).fetchall()

        return [
            InterventionRunRecord(
                id=row["id"],
                graph_id=row["graph_id"],
                selected_nodes=json.loads(row["selected_nodes"]),
                result_json=row["result"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def delete_intervention_run(self, run_id: int) -> None:
        """Delete an intervention run."""
        with self._write_lock():
            conn = self._get_conn()
            conn.execute("DELETE FROM intervention_runs WHERE id = ?", (run_id,))
            conn.commit()

    def delete_intervention_runs_for_graph(self, graph_id: int) -> int:
        """Delete all intervention runs for a graph. Returns count deleted."""
        with self._write_lock():
            conn = self._get_conn()
            cursor = conn.execute("DELETE FROM intervention_runs WHERE graph_id = ?", (graph_id,))
            conn.commit()
            return cursor.rowcount

    # -------------------------------------------------------------------------
    # Forked intervention run operations
    # -------------------------------------------------------------------------

    def save_forked_intervention_run(
        self,
        intervention_run_id: int,
        token_replacements: list[tuple[int, int]],
        result_json: str,
    ) -> int:
        """Save a forked intervention run.

        Args:
            intervention_run_id: The parent intervention run ID.
            token_replacements: List of (seq_pos, new_token_id) tuples.
            result_json: JSON-encoded InterventionResponse.

        Returns:
            The forked intervention run ID.
        """
        with self._write_lock():
            conn = self._get_conn()
            cursor = conn.execute(
                """INSERT INTO forked_intervention_runs (intervention_run_id, token_replacements, result)
                   VALUES (?, ?, ?)""",
                (intervention_run_id, json.dumps(token_replacements), result_json),
            )
            conn.commit()
            fork_id = cursor.lastrowid
            assert fork_id is not None
            return fork_id

    def get_forked_intervention_runs(
        self, intervention_run_id: int
    ) -> list[ForkedInterventionRunRecord]:
        """Get all forked runs for an intervention run.

        Args:
            intervention_run_id: The parent intervention run ID.

        Returns:
            List of forked intervention run records, ordered by creation time.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, intervention_run_id, token_replacements, result, created_at
               FROM forked_intervention_runs
               WHERE intervention_run_id = ?
               ORDER BY created_at""",
            (intervention_run_id,),
        ).fetchall()

        return [
            ForkedInterventionRunRecord(
                id=row["id"],
                intervention_run_id=row["intervention_run_id"],
                token_replacements=json.loads(row["token_replacements"]),
                result_json=row["result"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_intervention_run(self, run_id: int) -> InterventionRunRecord | None:
        """Get a single intervention run by ID."""
        conn = self._get_conn()
        row = conn.execute(
            """SELECT id, graph_id, selected_nodes, result, created_at
               FROM intervention_runs
               WHERE id = ?""",
            (run_id,),
        ).fetchone()

        if row is None:
            return None

        return InterventionRunRecord(
            id=row["id"],
            graph_id=row["graph_id"],
            selected_nodes=json.loads(row["selected_nodes"]),
            result_json=row["result"],
            created_at=row["created_at"],
        )

    def delete_forked_intervention_run(self, fork_id: int) -> None:
        """Delete a forked intervention run."""
        with self._write_lock():
            conn = self._get_conn()
            conn.execute("DELETE FROM forked_intervention_runs WHERE id = ?", (fork_id,))
            conn.commit()
