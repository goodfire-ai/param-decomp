"""Circuit-builder endpoints: browse subcomponents, compute j-vectors, assemble rank-1
LoRAs, and compare base vs edited model. See backend/circuit_builder.py for the math.

State model: one CircuitBuilderContext (real run or mock) + a dict of LoraSpecs +
a cache of computed j-vectors, all in-memory on the StateManager singleton (this is a
hands-on exploration tool; nothing persists across server restarts)."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from param_decomp_lab.app.backend.circuit_builder import (
    LOGITS_SITE,
    CompareResult,
    JVectorResult,
    LoraSpec,
    SubcomponentRef,
    build_lora,
    compare_models,
    compute_j_vectors,
    downstream_sites,
    site_rank,
    u_norm_absorbed,
)
from param_decomp_lab.app.backend.dependencies import DepStateManager
from param_decomp_lab.app.backend.mock_run import CircuitBuilderContext, load_mock_context
from param_decomp_lab.app.backend.utils import log_errors

router = APIRouter(prefix="/api/circuit_builder", tags=["circuit_builder"])


# =============================================================================
# In-memory circuit-builder state (attached to StateManager dynamically)
# =============================================================================


class _CBState:
    def __init__(self, ctx: CircuitBuilderContext) -> None:
        self.ctx = ctx
        self.loras: dict[str, LoraSpec] = {}
        # cache: (read_site, site, idx, n_prompts) -> JVectorResult
        self.j_cache: dict[tuple[str, str, int, int], JVectorResult] = {}


def _get_cb_state(manager: object) -> _CBState:
    state = getattr(manager, "_circuit_builder_state", None)
    if state is None:
        raise HTTPException(status_code=400, detail="Circuit builder not loaded. POST /api/circuit_builder/load first.")
    return state


# =============================================================================
# Schemas
# =============================================================================


class LoadRequest(BaseModel):
    source: str = "mock"  # "mock" | "run"
    run_ref: str | None = None  # run dir / wandb ref, e.g. "p-55ea3f9b" (source="run")
    seed: int = 0


class SiteInfo(BaseModel):
    site: str
    C: int
    d_in: int
    d_out: int
    rank: int


class SubcomponentInfo(BaseModel):
    site: str
    idx: int
    label: str | None
    label_source: str | None  # "canon" | "fallback" | "mock"
    u_norm_absorbed: float
    examples: list[dict]


class SearchHit(BaseModel):
    site: str
    idx: int
    label: str
    label_source: str


class ComponentDetail(BaseModel):
    site: str
    idx: int
    label: str | None
    label_source: str | None
    reasoning: str | None
    u_norm_absorbed: float
    examples: list[dict]


class JVectorRequest(BaseModel):
    read_site: str
    targets: list[dict]  # [{site, idx}]
    n_prompts: int = 16


class JVectorInfo(BaseModel):
    site: str
    idx: int
    raw_norm: float


class CompareRequest(BaseModel):
    prompt: str
    top_k: int = 10
    max_new_tokens: int = 32
    temperature: float = 0.8
    seed: int = 0


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/load")
@log_errors
def load(request: LoadRequest, manager: DepStateManager) -> dict:
    """Load the circuit-builder context: source='mock' or source='run' + run_ref."""
    existing = getattr(manager, "_circuit_builder_state", None)
    if (
        existing is not None
        and request.source == "run"
        and existing.ctx.run_id == request.run_ref
    ):
        return {"run_id": existing.ctx.run_id, "status": "already_loaded"}
    if request.source == "mock":
        ctx = load_mock_context(seed=request.seed)
    else:
        assert request.source == "run", f"unknown source {request.source!r}"
        assert request.run_ref, "source='run' requires run_ref (e.g. 'p-55ea3f9b')"
        from param_decomp_lab.app.backend.circuit_builder_loader import load_run_context

        with getattr(manager, "gpu_lock")():
            ctx = load_run_context(request.run_ref)
    manager._circuit_builder_state = _CBState(ctx)  # type: ignore[attr-defined]
    return {"run_id": ctx.run_id, "status": "loaded"}


@router.get("/sites")
@log_errors
def sites(manager: DepStateManager) -> list[SiteInfo]:
    state = _get_cb_state(manager)
    model = state.ctx.model
    out = []
    for site in sorted(model.target_module_paths, key=site_rank):
        comps = model.components[site]
        d_in, C = comps.V.shape
        _, d_out = comps.U.shape
        out.append(SiteInfo(site=site, C=C, d_in=d_in, d_out=d_out, rank=site_rank(site)))
    return out


@router.get("/downstream/{read_site:path}")
@log_errors
def downstream(read_site: str, manager: DepStateManager) -> list[str]:
    state = _get_cb_state(manager)
    return downstream_sites(state.ctx.model, read_site)


@router.get("/subcomponents/{site:path}")
@log_errors
def subcomponents(
    site: str, manager: DepStateManager, offset: int = 0, limit: int = 50, examples: int = 0
) -> list[SubcomponentInfo]:
    state = _get_cb_state(manager)
    model = state.ctx.model
    assert site in model.target_module_paths, f"unknown site {site}"
    C = model.components[site].V.shape[1]
    out = []
    for idx in range(offset, min(offset + limit, C)):
        labeled = state.ctx.info.label(site, idx)
        out.append(
            SubcomponentInfo(
                site=site,
                idx=idx,
                label=labeled[0] if labeled else None,
                label_source=labeled[1] if labeled else None,
                u_norm_absorbed=u_norm_absorbed(model, SubcomponentRef(site, idx)),
                examples=state.ctx.info.activating_examples(site, idx, examples) if examples else [],
            )
        )
    return out


@router.get("/search")
@log_errors
def search(
    q: str, manager: DepStateManager, limit: int = 50, downstream_of: str | None = None
) -> list[SearchHit]:
    """Substring search over component labels; optionally only sites downstream of a site."""
    state = _get_cb_state(manager)
    assert q.strip(), "empty search query"
    hits = state.ctx.info.search_labels(q.strip(), limit=500)
    if downstream_of is not None:
        allowed = set(downstream_sites(state.ctx.model, downstream_of))
        hits = [h for h in hits if h["site"] in allowed]
    known = set(state.ctx.model.target_module_paths)
    hits = [h for h in hits if h["site"] in known]
    return [SearchHit(**h) for h in hits[:limit]]


class TokenInfo(BaseModel):
    token_id: int
    token: str


@router.get("/tokens")
@log_errors
def tokens(text: str, manager: DepStateManager) -> list[TokenInfo]:
    """Tokenize text so the user can pick token ids for logit write terms."""
    state = _get_cb_state(manager)
    assert text, "empty text"
    ids = state.ctx.tokenizer.encode(text)
    spans = state.ctx.tokenizer.decode_tokens(ids)
    return [TokenInfo(token_id=i, token=t) for i, t in zip(ids, spans, strict=True)]


@router.get("/component/{site:path}/{idx}")
@log_errors
def component_detail(
    site: str, idx: int, manager: DepStateManager, examples: int = 8
) -> ComponentDetail:
    """Full autointerp explanation + activating examples for one component."""
    state = _get_cb_state(manager)
    model = state.ctx.model
    assert site in model.target_module_paths, f"unknown site {site}"
    interp = state.ctx.info.interpretation(site, idx)
    return ComponentDetail(
        site=site,
        idx=idx,
        label=interp["label"] if interp else None,
        label_source=interp["label_source"] if interp else None,
        reasoning=interp["reasoning"] if interp else None,
        u_norm_absorbed=u_norm_absorbed(model, SubcomponentRef(site, idx)),
        examples=state.ctx.info.activating_examples(site, idx, examples),
    )


@router.post("/j_vectors")
@log_errors
def j_vectors(request: JVectorRequest, manager: DepStateManager) -> list[JVectorInfo]:
    """Compute (and cache) j-vectors for downstream targets w.r.t. a read site."""
    state = _get_cb_state(manager)
    ctx = state.ctx
    refs = [SubcomponentRef(t["site"], int(t["idx"])) for t in request.targets]
    missing = [
        r for r in refs
        if (request.read_site, r.site, r.idx, request.n_prompts) not in state.j_cache
    ]
    if missing:
        results = compute_j_vectors(
            ctx.model,
            request.read_site,
            missing,
            ctx.token_provider.batches(ctx.batch_size, ctx.seq_len),
            n_prompts=request.n_prompts,
        )
        for r in results:
            state.j_cache[(request.read_site, r.ref.site, r.ref.idx, request.n_prompts)] = r
    return [
        JVectorInfo(
            site=r.site,
            idx=r.idx,
            raw_norm=state.j_cache[(request.read_site, r.site, r.idx, request.n_prompts)].raw_norm,
        )
        for r in refs
    ]


@router.get("/loras")
@log_errors
def list_loras(manager: DepStateManager) -> list[LoraSpec]:
    return list(_get_cb_state(manager).loras.values())


@router.put("/loras/{name}")
@log_errors
def put_lora(name: str, spec: LoraSpec, manager: DepStateManager) -> LoraSpec:
    state = _get_cb_state(manager)
    assert spec.name == name, f"name mismatch: url={name!r} body={spec.name!r}"
    model = state.ctx.model
    assert spec.read_site in model.target_module_paths, f"unknown site {spec.read_site}"
    allowed = set(downstream_sites(model, spec.read_site))
    for term in spec.writes:
        if term.kind == "u":
            assert term.site == spec.read_site, (
                f"U write term {term.site}:{term.idx} must be at the read site {spec.read_site}"
            )
        elif term.kind == "logit":
            assert term.site == LOGITS_SITE, f"logit terms use site={LOGITS_SITE!r}"
        else:
            assert term.site in allowed, f"{term.site} is not downstream of {spec.read_site}"
    state.loras[name] = spec
    return spec


@router.delete("/loras/{name}")
@log_errors
def delete_lora(name: str, manager: DepStateManager) -> dict:
    state = _get_cb_state(manager)
    if name not in state.loras:
        raise HTTPException(status_code=404, detail=f"no LoRA named {name!r}")
    del state.loras[name]
    return {"deleted": name}


@router.post("/compare")
@log_errors
def compare(request: CompareRequest, manager: DepStateManager) -> CompareResult:
    """Run the prompt through base and LoRA'd model; return logit diffs + continuations."""
    state = _get_cb_state(manager)
    ctx = state.ctx
    built = []
    with getattr(manager, "gpu_lock")():
        for spec in state.loras.values():
            if not spec.enabled:
                continue
            refs = [SubcomponentRef(t.site, t.idx) for t in spec.writes if t.kind in ("j", "logit")]
            missing = [
                r for r in refs
                if (spec.read_site, r.site, r.idx, spec.n_prompts) not in state.j_cache
            ]
            if missing:
                results = compute_j_vectors(
                    ctx.model, spec.read_site, missing,
                    ctx.token_provider.batches(ctx.batch_size, ctx.seq_len),
                    n_prompts=spec.n_prompts,
                )
                for r in results:
                    state.j_cache[(spec.read_site, r.ref.site, r.ref.idx, spec.n_prompts)] = r
            j_results = [
                state.j_cache[(spec.read_site, r.site, r.idx, spec.n_prompts)] for r in refs
            ]
            built.append(build_lora(ctx.model, spec, j_results))
        return compare_models(
            ctx.model, built, ctx.tokenizer, request.prompt,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            seed=request.seed,
        )
