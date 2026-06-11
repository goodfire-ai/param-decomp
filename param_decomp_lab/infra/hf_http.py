"""Make huggingface_hub's HTTP backend resilient to transient network flakiness.

HF's default session retries file *downloads* (via `http_backoff`) but mounts a bare
adapter for metadata calls like `HfApi.repo_info` — the call `datasets` makes to resolve
a streaming dataset's layout at startup. A single `ReadTimeout` there raises, and in a
DDP job that one rank's failure tears down every rank before training begins. This mounts
a retrying adapter on the session factory so connect/read timeouts and 5xx/429 are
retried with jittered backoff across *all* Hub HTTP calls (dataset, tokenizer, model).
"""

import requests
from huggingface_hub import configure_http_backend
from huggingface_hub.utils._http import UniqueRequestIdAdapter
from urllib3.util.retry import Retry

from param_decomp.log import logger

_configured = False


def configure_hf_http_retries(*, total_retries: int = 5, backoff_factor: float = 1.5) -> None:
    """Install a retrying HTTP backend on huggingface_hub (idempotent, process-global).

    `backoff_factor` with full jitter spaces retries at roughly 0, 1.5, 3, 6, 12s; the
    jitter de-synchronizes the simultaneous retries of many DDP ranks. Only idempotent
    methods (GET/HEAD) are retried, so non-idempotent writes are untouched.
    """
    global _configured
    if _configured:
        return

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

    def backend_factory() -> requests.Session:
        session = requests.Session()
        adapter = UniqueRequestIdAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    configure_http_backend(backend_factory=backend_factory)
    _configured = True
    logger.info("Configured huggingface_hub HTTP retries (total=%d)", total_retries)
