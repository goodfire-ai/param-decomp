"""Make huggingface_hub's HTTP backend resilient to a cold-cache startup burst.

The JAX trainer normally loads the frozen target's weights from a pre-warmed local
HF snapshot (`llama8b._hf_snapshot_dir` asserts the snapshot exists, no network call).
But at production topology (8 ranks/node x N nodes) a *cold* cache turns startup into an
8N-rank simultaneous Hub burst, and a single `ReadTimeout` on one rank tears the whole
job down before training begins. This mounts a retrying adapter on huggingface_hub's
session factory so connect/read timeouts and 5xx/429 are retried with jittered backoff,
de-synchronizing the simultaneous retries of many ranks.

`huggingface_hub` is not a hard dependency of this distribution (weights come via
`safetensors` from the local cache), so this is a no-op when it isn't importable.
"""

import importlib.util

_configured = False


def configure_hf_http_retries(*, total_retries: int = 5, backoff_factor: float = 1.5) -> None:
    """Install a retrying HTTP backend on huggingface_hub (idempotent, process-global).

    `backoff_factor` with full jitter spaces retries at roughly 0, 1.5, 3, 6, 12s; the
    jitter de-synchronizes the simultaneous retries of many ranks. Only idempotent
    methods (GET/HEAD) are retried, so non-idempotent writes are untouched. No-op when
    `huggingface_hub` is not installed in this venv.
    """
    global _configured
    if _configured:
        return
    _configured = True

    if importlib.util.find_spec("huggingface_hub") is None:
        return

    import requests
    from huggingface_hub import configure_http_backend
    from huggingface_hub.utils._http import UniqueRequestIdAdapter
    from urllib3.util.retry import Retry

    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        backoff_jitter=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    def backend_factory() -> "requests.Session":
        session = requests.Session()
        adapter = UniqueRequestIdAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    configure_http_backend(backend_factory=backend_factory)
    print(f"Configured huggingface_hub HTTP retries (total={total_retries})", flush=True)
