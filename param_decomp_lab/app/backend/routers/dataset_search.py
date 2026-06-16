"""Dataset search endpoints.

Provides search functionality for the training dataset of the loaded run.
The dataset name and text column are read from the run's config.
Results are cached in memory for pagination.

Currently restricted to SimpleStories runs (see `_assert_simplestories`).
"""

import random
import time
from typing import Annotated, Any

from datasets import Dataset, load_dataset
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from param_decomp.log import logger
from param_decomp_lab.app.backend.dependencies import DepLoadedRun, DepStateManager
from param_decomp_lab.app.backend.inference import next_token_probs
from param_decomp_lab.app.backend.schemas import DatasetSearchMetadata, DatasetSearchResult
from param_decomp_lab.app.backend.state import DatasetSearchState
from param_decomp_lab.app.backend.utils import log_errors

# =============================================================================
# Schemas
# =============================================================================


class TokenizedSearchResult(BaseModel):
    """A tokenized search result with per-token probability."""

    tokens: list[str]
    next_token_probs: list[float | None]
    occurrence_count: int
    metadata: dict[str, str]


class DatasetSearchPage(BaseModel):
    """Paginated results from a dataset search."""

    results: list[DatasetSearchResult]
    page: int
    page_size: int
    total_results: int
    total_pages: int


class TokenizedSearchPage(BaseModel):
    """Paginated tokenized results from a dataset search."""

    results: list[TokenizedSearchResult]
    query: str
    page: int
    page_size: int
    total_results: int
    total_pages: int


router = APIRouter(prefix="/api/dataset", tags=["dataset"])


def _assert_simplestories(dataset_name: str) -> None:
    """Raise 400 unless the run's dataset_name is a SimpleStories variant.

    The dataset explorer currently relies on SimpleStories's text format and
    full-in-memory load; other datasets (e.g. pile-tokenized-streaming) aren't
    supported.
    """
    if "simplestories" not in dataset_name.lower():
        raise HTTPException(
            status_code=400,
            detail=(f"Currently only simplestories is supported; got {dataset_name}"),
        )


@router.post("/search")
@log_errors
def search_dataset(
    query: Annotated[str, Query(min_length=1)],
    loaded: DepLoadedRun,
    manager: DepStateManager,
    split: Annotated[str, Query(pattern="^(train|test)$")] = "train",
) -> DatasetSearchMetadata:
    """Case-insensitive substring search of the run's training dataset.

    Reads `dataset_name` / `column_name` from the loaded run's config; caches results
    for pagination via `/results`.
    """
    dataset_name = loaded.lm_data.dataset_name
    text_column = loaded.lm_data.column_name
    _assert_simplestories(dataset_name)

    start_time = time.time()
    search_query = query.lower()

    logger.info(f"Loading dataset {dataset_name} (split={split})...")
    dataset = load_dataset(dataset_name, split=split)
    assert isinstance(dataset, Dataset), f"Expected Dataset, got {type(dataset)}"

    total_rows = len(dataset)
    logger.info(f"Searching {total_rows} rows for '{query}'...")

    filtered = dataset.filter(
        lambda x: search_query in x[text_column].lower(),
        num_proc=8,
    )

    # Collect extra string columns as metadata (skip the text column itself)
    column_names = dataset.column_names
    metadata_columns = [c for c in column_names if c != text_column]

    results: list[DatasetSearchResult] = []
    for item in filtered:
        item_dict: dict[str, Any] = dict(item)
        text: str = item_dict[text_column]
        row_metadata = {
            col: str(item_dict[col]) for col in metadata_columns if item_dict.get(col) is not None
        }
        results.append(
            DatasetSearchResult(
                text=text,
                occurrence_count=text.lower().count(search_query),
                metadata=row_metadata,
            )
        )

    search_time = time.time() - start_time

    search_metadata = DatasetSearchMetadata(
        query=query,
        split=split,
        dataset_name=dataset_name,
        total_results=len(results),
        search_time_seconds=search_time,
    )
    manager.state.dataset_search_state = DatasetSearchState(
        results=results,
        metadata=search_metadata,
    )

    logger.info(f"Found {len(results)} results in {search_time:.2f}s (searched {total_rows} rows)")

    return search_metadata


@router.get("/results")
@log_errors
def get_dataset_results(
    manager: DepStateManager,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> DatasetSearchPage:
    """Paginated results from the last dataset search."""
    search_state = manager.state.dataset_search_state
    if search_state is None:
        raise HTTPException(
            status_code=404,
            detail="No search results available. Perform a search first.",
        )

    total_results = len(search_state.results)
    total_pages = max(1, (total_results + page_size - 1) // page_size)

    if page > total_pages and total_results > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Page {page} exceeds total pages {total_pages}",
        )

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_results = search_state.results[start_idx:end_idx]

    return DatasetSearchPage(
        results=page_results,
        page=page,
        page_size=page_size,
        total_results=total_results,
        total_pages=total_pages,
    )


@router.get("/results_tokenized")
@log_errors
def get_tokenized_results(
    loaded: DepLoadedRun,
    manager: DepStateManager,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=20)] = 10,
    max_tokens: Annotated[int, Query(ge=16, le=512)] = 256,
) -> TokenizedSearchPage:
    """Paginated tokenized results with per-token next-token probability.

    Requires a loaded run for model inference (hence smaller `page_size` limit than
    `/results`); results longer than `max_tokens` are truncated.
    """
    search_state = manager.state.dataset_search_state
    if search_state is None:
        raise HTTPException(
            status_code=404,
            detail="No search results available. Perform a search first.",
        )

    tokenizer = loaded.tokenizer

    total_results = len(search_state.results)
    total_pages = max(1, (total_results + page_size - 1) // page_size)

    if page > total_pages and total_results > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Page {page} exceeds total pages {total_pages}",
        )

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_results = search_state.results[start_idx:end_idx]

    tokenized_results: list[TokenizedSearchResult] = []

    for result in page_results:
        token_ids = tokenizer.encode(result.text)
        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]

        if len(token_ids) == 0:
            continue

        token_strings = loaded.tokenizer.get_spans(token_ids)

        tokenized_results.append(
            TokenizedSearchResult(
                tokens=token_strings,
                next_token_probs=next_token_probs(loaded.jax_run, token_ids),
                occurrence_count=result.occurrence_count,
                metadata=result.metadata,
            )
        )

    query = search_state.metadata.query

    return TokenizedSearchPage(
        results=tokenized_results,
        query=query,
        page=page,
        page_size=page_size,
        total_results=total_results,
        total_pages=total_pages,
    )


class RandomSamplesResult(BaseModel):
    """Random samples from the dataset."""

    results: list[DatasetSearchResult]
    total_available: int
    seed: int


@router.get("/random")
@log_errors
def get_random_samples(
    loaded: DepLoadedRun,
    n_samples: Annotated[int, Query(ge=1, le=200)] = 100,
    seed: Annotated[int, Query(ge=0)] = 42,
    split: Annotated[str, Query(pattern="^(train|test)$")] = "train",
) -> RandomSamplesResult:
    """Random samples from the loaded run's training dataset.

    Reads `dataset_name` / `column_name` from the loaded run's config.
    """
    dataset_name = loaded.lm_data.dataset_name
    text_column = loaded.lm_data.column_name
    _assert_simplestories(dataset_name)

    logger.info(f"Loading dataset {dataset_name} (split={split}) for random sampling...")
    dataset = load_dataset(dataset_name, split=split)
    assert isinstance(dataset, Dataset), f"Expected Dataset, got {type(dataset)}"

    total_available = len(dataset)
    actual_samples = min(n_samples, total_available)

    # Generate random indices directly instead of shuffling entire dataset (~100x faster)
    rng = random.Random(seed)
    indices = rng.sample(range(total_available), actual_samples)
    samples = dataset.select(indices)

    metadata_columns = [c for c in dataset.column_names if c != text_column]

    results = []
    for item in samples:
        item_dict: dict[str, Any] = dict(item)
        text: str = item_dict[text_column]
        row_metadata = {
            col: str(item_dict[col]) for col in metadata_columns if item_dict.get(col) is not None
        }
        results.append(
            DatasetSearchResult(
                text=text,
                occurrence_count=0,
                metadata=row_metadata,
            )
        )

    logger.info(f"Returned {len(results)} random samples from {total_available} total rows")

    return RandomSamplesResult(
        results=results,
        total_available=total_available,
        seed=seed,
    )


class TokenizedSample(BaseModel):
    """A single tokenized sample with per-token next-token probability."""

    tokens: list[str]
    next_token_probs: list[float | None]  # Probability of next token; None for last position
    metadata: dict[str, str]


class RandomSamplesWithLossResult(BaseModel):
    """Random samples with tokenized data and next-token probabilities."""

    results: list[TokenizedSample]
    total_available: int
    seed: int


@router.get("/random_with_loss")
@log_errors
def get_random_samples_with_loss(
    loaded: DepLoadedRun,
    n_samples: Annotated[int, Query(ge=1, le=50)] = 20,
    seed: Annotated[int, Query(ge=0)] = 42,
    split: Annotated[str, Query(pattern="^(train|test)$")] = "train",
    max_tokens: Annotated[int, Query(ge=16, le=512)] = 256,
) -> RandomSamplesWithLossResult:
    """Random samples tokenized + run through the model for per-token next-token probability.

    Requires a loaded run; lower `n_samples` cap than `/random-samples` because of model
    inference; results longer than `max_tokens` are truncated.
    """
    dataset_name = loaded.lm_data.dataset_name
    text_column = loaded.lm_data.column_name
    _assert_simplestories(dataset_name)

    tokenizer = loaded.tokenizer

    logger.info(f"Loading dataset {dataset_name} (split={split}) for random sampling with loss...")
    dataset = load_dataset(dataset_name, split=split)
    assert isinstance(dataset, Dataset), f"Expected Dataset, got {type(dataset)}"

    total_available = len(dataset)
    actual_samples = min(n_samples, total_available)

    rng = random.Random(seed)
    indices = rng.sample(range(total_available), actual_samples)
    samples = dataset.select(indices)

    metadata_columns = [c for c in dataset.column_names if c != text_column]

    results: list[TokenizedSample] = []

    for item in samples:
        item_dict: dict[str, Any] = dict(item)
        text: str = item_dict[text_column]

        token_ids = tokenizer.encode(text)
        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]

        if len(token_ids) == 0:
            continue

        token_strings = loaded.tokenizer.get_spans(token_ids)

        row_metadata = {
            col: str(item_dict[col]) for col in metadata_columns if item_dict.get(col) is not None
        }

        results.append(
            TokenizedSample(
                tokens=token_strings,
                next_token_probs=next_token_probs(loaded.jax_run, token_ids),
                metadata=row_metadata,
            )
        )

    logger.info(
        f"Returned {len(results)} tokenized samples with CE loss from {total_available} total rows"
    )

    return RandomSamplesWithLossResult(
        results=results,
        total_available=total_available,
        seed=seed,
    )
