"""FastAPI server for scope: `python -m param_decomp_lab.scope.backend.server --port N`."""

import argparse
import gzip
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from param_decomp_lab.scope.backend.data_source import (
    CatalogResponse,
    ComponentDetail,
    ComponentLabel,
    ComponentListResponse,
    ScopeDataSource,
    ScopeNotFoundError,
    SiteCurve,
    SortKey,
)
from param_decomp_lab.scope.backend.fixture_data_source import FixtureDataSource

MAX_GZIPPED_RESPONSE_BYTES = 50_000

app = FastAPI(title="scope")
app.add_middleware(GZipMiddleware, minimum_size=500)

data_source: ScopeDataSource = FixtureDataSource()


def use_artifact_source() -> None:
    from param_decomp_lab.scope.backend.artifact_data_source import ArtifactDataSource

    global data_source
    data_source = ArtifactDataSource()


@app.exception_handler(ScopeNotFoundError)
def not_found_handler(_request: Request, exc: ScopeNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


def budgeted_json(model: BaseModel) -> Response:
    """Serialize, asserting the gzipped payload stays within the per-response budget."""
    body = model.model_dump_json()
    gzipped_size = len(gzip.compress(body.encode()))
    assert gzipped_size <= MAX_GZIPPED_RESPONSE_BYTES, (
        f"response is {gzipped_size}B gzipped, budget is {MAX_GZIPPED_RESPONSE_BYTES}B"
    )
    return Response(content=body, media_type="application/json")


@app.get("/api/catalog")
def get_catalog() -> CatalogResponse:
    return data_source.catalog()


@app.get("/api/runs/{run_id}/sites/{site}/components")
def list_components(
    run_id: str,
    site: str,
    sort: SortKey = "density",
    page: Annotated[int, Query(ge=0)] = 0,
    page_size: Annotated[int, Query(ge=1, le=200)] = 50,
    q: str = "",
) -> Response:
    listing: ComponentListResponse = data_source.list_components(
        run_id, site, sort, page, page_size, q
    )
    return budgeted_json(listing)


@app.get("/api/runs/{run_id}/sites/{site}/components/{idx}")
def get_component_detail(run_id: str, site: str, idx: int) -> Response:
    detail: ComponentDetail = data_source.component_detail(run_id, site, idx)
    return budgeted_json(detail)


@app.get("/api/runs/{run_id}/sites/{site}/curve")
def get_site_curve(run_id: str, site: str) -> Response:
    curve: SiteCurve = data_source.site_curve(run_id, site)
    return budgeted_json(curve)


@app.post("/api/runs/{run_id}/sites/{site}/components/{idx}/label")
def create_label(run_id: str, site: str, idx: int) -> ComponentLabel:
    return data_source.create_label(run_id, site, idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="scope backend")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--data-source", choices=["fixture", "artifacts"], default="fixture")
    args = parser.parse_args()
    if args.data_source == "artifacts":
        use_artifact_source()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
