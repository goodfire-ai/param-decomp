"""Make HuggingFace HTTP reads resilient to transient CDN errors.

`datasets` streaming retries `HfHubHTTPError` only for 503/429; a 408 ("Request
Time-out") from the xet-bridge CDN falls through and kills the whole DDP job mid-run.
Installing a `requests` retry policy at the `huggingface_hub` backend level retries
408s (and other transient statuses) transparently for every Hub read, including the
fsspec range reads `datasets` issues while streaming parquet shards.
"""

import huggingface_hub
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 408 is the one that bites here; the rest are the usual transient-server set.
_RETRY_STATUSES = (408, 429, 500, 502, 503, 504)


def _session_factory() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=8,
        backoff_factor=1.0,  # 0s, 2s, 4s, 8s, ... between attempts
        status_forcelist=_RETRY_STATUSES,
        allowed_methods=None,  # retry all verbs, including the GET range reads
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def configure_hf_retries() -> None:
    """Route all `huggingface_hub` (and `datasets`-streaming) HTTP through a retrying session."""
    huggingface_hub.configure_http_backend(backend_factory=_session_factory)
