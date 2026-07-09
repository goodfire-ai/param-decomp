"""Real-run loader for the circuit builder: SavedLMRun + InterpRepo/HarvestRepo behind
the same CircuitBuilderContext interface as the mock (see mock_run.py).

Loads the PD run (e.g. p-55ea3f9b), its tokenizer, an eval-split dataloader for
j-vector averaging, and autointerp labels + harvested activation examples."""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from jaxtyping import Int
from torch import Tensor

from param_decomp.log import logger
from param_decomp_config.lm import LMDataConfig
from param_decomp_lab.app.backend.app_tokenizer import AppTokenizer
from param_decomp_lab.app.backend.mock_run import CircuitBuilderContext
from param_decomp_lab.autointerp.repo import InterpRepo
from param_decomp_lab.experiments.lm.run import SavedLMRun, build_lm_loader
from param_decomp_lab.harvest.repo import HarvestRepo
from param_decomp_lab.infra.paths import ModelPath


class HFTokenizerAdapter:
    """AppTokenizer -> the circuit builder's TokenizerProtocol."""

    def __init__(self, app_tokenizer: AppTokenizer) -> None:
        self._tok = app_tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode_tokens(self, token_ids: list[int]) -> list[str]:
        return self._tok.get_spans(token_ids)


@dataclass
class EvalSplitTokenProvider:
    """(B, T) token batches from the run's eval split — the j-vector averaging data."""

    data_cfg: LMDataConfig
    device: str

    def batches(self, batch_size: int, seq_len: int) -> Iterator[Int[Tensor, "B T"]]:
        loader = build_lm_loader(
            target_cfg=None,  # type: ignore[arg-type]  # build_lm_loader dels target_cfg
            data_cfg=self.data_cfg,
            split="eval",
            device=self.device,
            batch_size=batch_size,
            seed=0,
        )
        for batch in loader:
            assert isinstance(batch, Tensor), f"expected token tensor, got {type(batch)}"
            yield batch[:, :seq_len]


class RepoInfoProvider:
    """Autointerp labels + harvested activation examples for real runs.

    Component keys in both repos are `{concrete_module_path}:{idx}` — exactly the
    circuit builder's (site, idx)."""

    def __init__(
        self, interp: InterpRepo | None, harvest: HarvestRepo | None, tokenizer: AppTokenizer
    ) -> None:
        self._interp = interp
        self._harvest = harvest
        self._tokenizer = tokenizer

    def label(self, site: str, idx: int) -> str | None:
        if self._interp is None:
            return None
        result = self._interp.get_interpretation(f"{site}:{idx}")
        return result.label if result is not None else None

    def activating_examples(self, site: str, idx: int, limit: int) -> list[dict]:
        if self._harvest is None:
            return []
        comp = self._harvest.get_component(f"{site}:{idx}")
        if comp is None:
            return []
        out = []
        for ex in comp.activation_examples[:limit]:
            acts = next(iter(ex.activations.values()), [])
            peak = max(range(len(acts)), key=lambda i: acts[i]) if acts else 0
            out.append(
                {
                    "tokens": self._tokenizer.get_spans(ex.token_ids),
                    "active_position": peak,
                    "activation": round(acts[peak], 4) if acts else 0.0,
                }
            )
        return out


def load_run_context(
    run_ref: ModelPath,
    *,
    device: str | None = None,
    batch_size: int = 8,
    seq_len: int = 128,
) -> CircuitBuilderContext:
    """Open a saved PD run as a CircuitBuilderContext (the real counterpart of the mock)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # A bare run id resolves to the local runs dir when present (avoids the wandb path).
    from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR

    local_dir = PARAM_DECOMP_OUT_DIR / "runs" / str(run_ref)
    if local_dir.is_dir():
        run_ref = local_dir
    saved = SavedLMRun.from_path(run_ref)
    model = saved.load_model().to(device)
    model.eval()

    run_id = saved.checkpoint_path.parent.name
    app_tokenizer = AppTokenizer.from_pretrained(saved.cfg.data.tokenizer_name)

    interp = InterpRepo.open(run_id)
    if interp is None:
        logger.warning(f"no autointerp data for {run_id} — labels will be empty")
    harvest = HarvestRepo.open_most_recent(run_id)
    if harvest is None:
        logger.warning(f"no harvest data for {run_id} — activation examples will be empty")

    return CircuitBuilderContext(
        run_id=run_id,
        model=model,
        tokenizer=HFTokenizerAdapter(app_tokenizer),
        token_provider=EvalSplitTokenProvider(data_cfg=saved.cfg.data, device=device),
        info=RepoInfoProvider(interp, harvest, app_tokenizer),
        seq_len=seq_len,
        batch_size=batch_size,
    )
