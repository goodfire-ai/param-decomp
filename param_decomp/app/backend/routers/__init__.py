"""FastAPI routers for the PD backend API."""

from param_decomp.app.backend.routers.activation_contexts import (
    router as activation_contexts_router,
)
from param_decomp.app.backend.routers.agents import router as agents_router
from param_decomp.app.backend.routers.autointerp_compare import router as autointerp_compare_router
from param_decomp.app.backend.routers.clusters import router as clusters_router
from param_decomp.app.backend.routers.correlations import router as correlations_router
from param_decomp.app.backend.routers.data_sources import router as data_sources_router
from param_decomp.app.backend.routers.dataset_attributions import (
    router as dataset_attributions_router,
)
from param_decomp.app.backend.routers.dataset_search import router as dataset_search_router
from param_decomp.app.backend.routers.graph_interp import router as graph_interp_router
from param_decomp.app.backend.routers.graphs import router as graphs_router
from param_decomp.app.backend.routers.intervention import router as intervention_router
from param_decomp.app.backend.routers.investigations import router as investigations_router
from param_decomp.app.backend.routers.mcp import router as mcp_router
from param_decomp.app.backend.routers.pretrain_info import router as pretrain_info_router
from param_decomp.app.backend.routers.prompts import router as prompts_router
from param_decomp.app.backend.routers.run_registry import router as run_registry_router
from param_decomp.app.backend.routers.runs import router as runs_router

__all__ = [
    "activation_contexts_router",
    "agents_router",
    "autointerp_compare_router",
    "clusters_router",
    "correlations_router",
    "data_sources_router",
    "dataset_attributions_router",
    "dataset_search_router",
    "graph_interp_router",
    "graphs_router",
    "intervention_router",
    "investigations_router",
    "mcp_router",
    "pretrain_info_router",
    "prompts_router",
    "run_registry_router",
    "runs_router",
]
