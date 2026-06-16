"""Make huggingface_hub's HTTP backend resilient to transient network flakiness.

HF retries file *downloads* (via `http_backoff`) but `HfApi.repo_info` — the call
`datasets` makes to resolve a streaming dataset's layout at startup — goes straight
through `get_session().get(...)` with no retry. A single timeout there raises, and in a
DDP job that one rank's failure tears down every rank before training begins. This
installs a client factory whose transport retries connect/read timeouts, network errors,
and 5xx/429 with jittered backoff across *all* Hub HTTP calls (dataset, tokenizer, model).

huggingface_hub >=1.0 moved from `requests` to `httpx`: the old `configure_http_backend`
hook is gone, replaced by `set_client_factory`.
"""

import random
import time
from typing import override

import httpx
from huggingface_hub.utils import set_client_factory
from huggingface_hub.utils._http import hf_request_event_hook

from param_decomp.log import logger

_configured = False

# Mirror huggingface_hub's own retry policy (`_http._DEFAULT_RETRY_ON_*`).
_RETRY_ON_EXCEPTIONS = (httpx.TimeoutException, httpx.NetworkError)
_RETRY_ON_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# Only idempotent methods are safe to replay; writes are sent once.
_RETRY_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class _RetryTransport(httpx.HTTPTransport):
    def __init__(self, *, total_retries: int, backoff_factor: float) -> None:
        super().__init__()
        self._total_retries = total_retries
        self._backoff_factor = backoff_factor

    @override
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method not in _RETRY_METHODS:
            return super().handle_request(request)
        for attempt in range(self._total_retries + 1):
            last = attempt == self._total_retries
            try:
                response = super().handle_request(request)
            except _RETRY_ON_EXCEPTIONS as err:
                if last:
                    raise
                self._sleep(attempt, reason=repr(err))
                continue
            if response.status_code in _RETRY_ON_STATUS_CODES and not last:
                response.close()
                self._sleep(attempt, reason=f"status {response.status_code}")
                continue
            return response
        raise AssertionError("unreachable")

    def _sleep(self, attempt: int, *, reason: str) -> None:
        # Full-jitter exponential backoff (~0, 1.5, 3, 6, 12s at backoff_factor=1.5); the
        # jitter de-synchronizes the simultaneous retries of many DDP ranks.
        delay = self._backoff_factor * (2**attempt) * random.random()
        logger.warning("HF Hub request failed (%s); retrying in %.1fs", reason, delay)
        time.sleep(delay)


def configure_hf_http_retries(*, total_retries: int = 5, backoff_factor: float = 1.5) -> None:
    """Install a retrying HTTP client factory on huggingface_hub (idempotent, process-global)."""
    global _configured
    if _configured:
        return

    def client_factory() -> httpx.Client:
        # Match huggingface_hub's `default_client_factory` apart from the retrying transport.
        return httpx.Client(
            transport=_RetryTransport(total_retries=total_retries, backoff_factor=backoff_factor),
            event_hooks={"request": [hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
        )

    set_client_factory(client_factory)
    _configured = True
    logger.info("Configured huggingface_hub HTTP retries (total=%d)", total_retries)
