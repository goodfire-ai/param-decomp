"""Train specific component U/V vectors on arbitrary losses.

Each SPD component is a rank-1 adapter: V[:, c] @ U[c, :]. This module lets you
unfreeze specific read (V column) or write (U row) vectors and optimize them,
while the rest of the model stays frozen. The forward pass uses all-ones component
masks plus a snapshotted weight delta, so it starts from the target model's behavior
and learns rank-1 perturbations.

Usage:
    em, tok = EditableModel.from_wandb("wandb:goodfire/spd/s-892f140b")

    trainer = ComponentTrainer(
        em.model,
        targets={"h.1.mlp.down_proj:798": "both", "h.1.attn.o_proj:82": "write"},
        lr=1e-4,
    )

    for batch_tokens in data:
        logits = trainer(batch_tokens)
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), batch_tokens[:, 1:].flatten())
        trainer.step(loss)

    trainer.cleanup()
"""

from functools import partial
from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from spd.editing._editing import parse_component_key
from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponents

TrainMode = Literal["read", "write", "both"]


class ComponentTrainer:
    """Trains specific component U/V vectors while the rest of the model is frozen.

    Forward pass runs through all components with ones masks + frozen weight delta,
    so the model starts from target-model behavior. Only the specified V columns
    (read vectors) and/or U rows (write vectors) receive gradients.
    """

    def __init__(
        self,
        model: ComponentModel,
        targets: dict[str, TrainMode],
        lr: float,
        weight_decay: float = 0.0,
    ):
        self.model = model
        self.model.train()

        # Snapshot weight deltas BEFORE changing requires_grad
        self._frozen_weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] = {
            k: v.detach().clone() for k, v in model.calc_weight_deltas().items()
        }

        # Freeze everything
        model.requires_grad_(False)

        # Parse targets into per-layer specs
        layer_specs: dict[str, dict[int, TrainMode]] = {}
        for key, mode in targets.items():
            layer, idx = parse_component_key(key)
            assert layer in model.components, f"Unknown layer: {layer}"
            assert idx < model.components[layer].C, (
                f"Component index {idx} >= C={model.components[layer].C} for {layer}"
            )
            layer_specs.setdefault(layer, {})[idx] = mode

        # Unfreeze relevant V/U params and register gradient masks
        self._grad_hooks: list[RemovableHandle] = []
        trainable_params: list[Tensor] = []

        for layer, specs in layer_specs.items():
            comp = model.components[layer]

            train_any_read = any(m in ("read", "both") for m in specs.values())
            train_any_write = any(m in ("write", "both") for m in specs.values())

            if train_any_read:
                comp.V.requires_grad = True
                v_mask = torch.zeros(comp.C, device=comp.V.device)
                for idx, mode in specs.items():
                    if mode in ("read", "both"):
                        v_mask[idx] = 1.0
                # Mask: [C] broadcast to [d_in, C] — zeros out gradients for non-target columns
                hook = comp.V.register_hook(lambda g, m=v_mask: g * m.unsqueeze(0))
                self._grad_hooks.append(hook)
                trainable_params.append(comp.V)

            if train_any_write:
                comp.U.requires_grad = True
                u_mask = torch.zeros(comp.C, device=comp.U.device)
                for idx, mode in specs.items():
                    if mode in ("write", "both"):
                        u_mask[idx] = 1.0
                # Mask: [C] broadcast to [C, d_out] — zeros out gradients for non-target rows
                hook = comp.U.register_hook(lambda g, m=u_mask: g * m.unsqueeze(1))
                self._grad_hooks.append(hook)
                trainable_params.append(comp.U)

        assert trainable_params, "No trainable parameters"
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Forward pass with all-ones masks and frozen weight deltas.

        Accepts the same arguments as the target model. Hooks intercept each
        decomposed layer and route through components + weight delta.
        """
        hooks = {
            module_name: partial(self._component_hook, module_name=module_name)
            for module_name in self.model.target_module_paths
        }
        with self.model._attach_forward_hooks(hooks):
            raw_out = self.model.target_model(*args, **kwargs)
        return self.model._extract_output(raw_out)

    def _component_hook(
        self,
        _module: Any,
        args: list[Any],
        kwargs: dict[Any, Any],
        _output: Any,
        module_name: str,
    ) -> Tensor:
        assert len(args) == 1 and len(kwargs) == 0
        x = args[0]
        components = self.model.components[module_name]

        batch_shape = x.shape if isinstance(components, EmbeddingComponents) else x.shape[:-1]

        weight_delta = self._frozen_weight_deltas[module_name].to(x.device)
        weight_delta_mask = torch.ones(batch_shape, device=x.device)

        return components(
            x,
            mask=None,
            weight_delta_and_mask=(weight_delta, weight_delta_mask),
        )

    def step(self, loss: Tensor) -> None:
        """Backward + optimizer step."""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def cleanup(self) -> None:
        """Remove gradient hooks and re-freeze parameters."""
        for hook in self._grad_hooks:
            hook.remove()
        self._grad_hooks.clear()
        self.model.requires_grad_(False)
        self.model.eval()
