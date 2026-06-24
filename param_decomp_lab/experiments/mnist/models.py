"""MNIST MLP target model for the memorization-vs-generalization PD experiment.

A plain feed-forward classifier with named `nn.Linear` modules (`fc_in`, `fc_h.*`,
`fc_out`) so VPD can decompose each weight matrix. The same architecture is used for
every label-noise / size condition; only the (possibly corrupted) labels differ.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from pydantic import PositiveInt
from torch import Tensor, nn

from param_decomp.base_config import BaseConfig
from param_decomp.schedule import ScheduleConfig
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import resolve_run_files


class MnistMLPModelConfig(BaseConfig):
    """Architecture of the MNIST MLP target.

    Modules: `fc_in` (d_input -> width), `fc_h.{0..n_hidden_layers-2}` (width -> width),
    `fc_out` (width -> n_classes), with `act_fn` between hidden layers. `n_hidden_layers`
    counts the hidden representations; with the default 2, the network is
    d_input -> width (fc_in) -> width (fc_h.0) -> n_classes (fc_out).
    """

    d_input: PositiveInt = 784
    width: PositiveInt = 2048
    n_hidden_layers: PositiveInt = 2
    n_classes: PositiveInt = 10
    act_fn_name: Literal["gelu", "relu"] = "gelu"
    bias: bool = True


class MnistTrainConfig(BaseConfig):
    """Training recipe for an MNIST MLP target. Identical across conditions except for
    `label_noise_p` (and, for the size-ladder facet, `n_train_examples`)."""

    wandb_project: str | None = None
    seed: int = 0
    mnist_model_config: MnistMLPModelConfig
    # Fraction of training labels to randomize (0.0 = clean generalizer, 1.0 = pure memorizer).
    label_noise_p: float = 0.0
    label_noise_seed: int = 0
    # Number of training examples to memorize (None = full 60k). Used by the size ladder.
    n_train_examples: int | None = None
    subsample_seed: int = 0
    normalize: bool = True
    data_dir: str
    batch_size: PositiveInt = 1024
    steps: PositiveInt = 20000
    eval_every: PositiveInt = 1000
    print_freq: PositiveInt = 200
    lr_schedule: ScheduleConfig
    weight_decay: float = 0.0


MNIST_TRAIN_CONFIG_FILENAME = "mnist_train_config.yaml"
MNIST_CHECKPOINT_FILENAME = "mnist_mlp.pth"
# Saved (subsample-index, possibly-corrupted-label) spec so the decomposition loader
# reproduces the exact memorized training set the target saw.
MNIST_MEMSET_FILENAME = "memorized_dataset.pt"


@dataclass
class MnistTargetRunInfo:
    """Run info from training an MNIST MLP target."""

    checkpoint_path: Path
    config: MnistTrainConfig
    train_indices: Int[Tensor, " n"]
    train_labels: Int[Tensor, " n"]

    @classmethod
    def from_path(cls, path: ModelPath) -> "MnistTargetRunInfo":
        files = resolve_run_files(
            path,
            config_filename=MNIST_TRAIN_CONFIG_FILENAME,
            checkpoint_filename=MNIST_CHECKPOINT_FILENAME,
            extras_from_config_path=lambda _: [MNIST_MEMSET_FILENAME],
        )
        memset = torch.load(files.extras[MNIST_MEMSET_FILENAME], weights_only=True)
        return cls(
            checkpoint_path=files.checkpoint_path,
            config=MnistTrainConfig.from_file(files.config_path),
            train_indices=memset["indices"],
            train_labels=memset["labels"],
        )


class MnistMLP(nn.Module):
    def __init__(self, config: MnistMLPModelConfig):
        super().__init__()
        self.config = config
        assert config.act_fn_name in ["gelu", "relu"]
        self.act_fn = F.gelu if config.act_fn_name == "gelu" else F.relu

        self.fc_in = nn.Linear(config.d_input, config.width, bias=config.bias)
        self.fc_h = nn.ModuleList(
            [
                nn.Linear(config.width, config.width, bias=config.bias)
                for _ in range(config.n_hidden_layers - 1)
            ]
        )
        self.fc_out = nn.Linear(config.width, config.n_classes, bias=config.bias)

    @override
    def forward(self, x: Float[Tensor, "... d_input"]) -> Float[Tensor, "... n_classes"]:
        x = self.act_fn(self.fc_in(x))
        for layer in self.fc_h:
            x = self.act_fn(layer(x))
        return self.fc_out(x)

    @classmethod
    def from_run_info(cls, run_info: MnistTargetRunInfo) -> "MnistMLP":
        model = cls(config=run_info.config.mnist_model_config)
        model.load_state_dict(
            torch.load(run_info.checkpoint_path, weights_only=True, map_location="cpu")
        )
        return model

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> "MnistMLP":
        return cls.from_run_info(MnistTargetRunInfo.from_path(path))
