import tempfile
from pathlib import Path
from typing import Any, override

import pytest
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformers.pytorch_utils import Conv1D as RadfordConv1D

from spd.configs import (
    Config,
    GlobalCiConfig,
    ImportanceMinimalityLossConfig,
    LayerwiseCiConfig,
    ModulePatternInfoConfig,
    ScheduleConfig,
    TMSTaskConfig,
)
from spd.identity_insertion import insert_identity_operations_
from spd.interfaces import LoadableModule, RunInfo
from spd.models.component_model import (
    ComponentModel,
    SPDRunInfo,
)
from spd.models.components import (
    ComponentsMaskInfo,
    EmbeddingComponents,
    GlobalCiFnWrapper,
    GlobalSharedMLPCiFn,
    GlobalSharedTransformerCiFn,
    LinearComponents,
    MLPCiFn,
    ParallelLinear,
    TargetLayerConfig,
    VectorMLPCiFn,
    VectorSharedMLPCiFn,
    make_mask_infos,
)
from spd.spd_types import ModelPath
from spd.utils.module_utils import ModulePathInfo, expand_module_patterns
from spd.utils.run_utils import save_file


class SimpleTestModel(LoadableModule):
    """Simple test model with Linear and Embedding layers for unit‑testing."""

    LINEAR_1_SHAPE = (10, 5)
    LINEAR_2_SHAPE = (5, 3)
    CONV1D_1_SHAPE = (3, 5)
    CONV1D_2_SHAPE = (1, 3)
    EMBEDDING_SHAPE = (100, 8)

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(*self.LINEAR_1_SHAPE, bias=True)
        self.linear2 = nn.Linear(*self.LINEAR_2_SHAPE, bias=False)
        self.conv1d1 = RadfordConv1D(*self.CONV1D_1_SHAPE)
        self.conv1d2 = RadfordConv1D(*self.CONV1D_2_SHAPE)

        self.embedding = nn.Embedding(*self.EMBEDDING_SHAPE)
        self.other_layer = nn.ReLU()  # Non‑target layer (should never be wrapped)

    @override
    def forward(self, x: Float[Tensor, "... 10"]):  # noqa: D401,E501
        x = self.linear2(self.linear1(x))
        x = self.conv1d2(self.conv1d1(x))
        return x

    @classmethod
    @override
    def from_run_info(cls, run_info: RunInfo[Any]) -> "SimpleTestModel":
        model = cls()
        model.load_state_dict(torch.load(run_info.checkpoint_path))
        return model

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "SimpleTestModel":
        model = cls()
        model.load_state_dict(torch.load(path))
        return model


def test_correct_parameters_require_grad():
    target_model = SimpleTestModel()
    target_model.requires_grad_(False)

    component_model = ComponentModel(
        target_model=target_model,
        module_path_info=[
            ModulePathInfo(module_path="linear1", C=4),
            ModulePathInfo(module_path="linear2", C=8),
            ModulePathInfo(module_path="embedding", C=6),
            ModulePathInfo(module_path="conv1d1", C=10),
            ModulePathInfo(module_path="conv1d2", C=5),
        ],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    for module_path, components in component_model.components.items():
        assert components.U.requires_grad
        assert components.V.requires_grad

        target_module = component_model.target_model.get_submodule(module_path)

        if isinstance(target_module, nn.Linear | RadfordConv1D):
            assert not target_module.weight.requires_grad
            if target_module.bias is not None:  # pyright: ignore [reportUnnecessaryComparison]
                assert not target_module.bias.requires_grad
            assert isinstance(components, LinearComponents)
            if components.bias is not None:
                assert not components.bias.requires_grad
        else:
            assert isinstance(target_module, nn.Embedding), "sanity check"
            assert isinstance(components, EmbeddingComponents)
            assert not target_module.weight.requires_grad


def test_from_run_info():
    target_model = SimpleTestModel()

    target_model.eval()
    target_model.requires_grad_(False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        base_model_dir = base_dir / "test_model"
        base_model_dir.mkdir(parents=True, exist_ok=True)
        comp_model_dir = base_dir / "comp_model"
        comp_model_dir.mkdir(parents=True, exist_ok=True)

        base_model_path = base_model_dir / "model.pth"
        save_file(target_model.state_dict(), base_model_path)

        config = Config(
            pretrained_model_class="tests.test_component_model.SimpleTestModel",
            pretrained_model_path=base_model_path,
            pretrained_model_name=None,
            module_info=[
                ModulePatternInfoConfig(module_pattern="linear1", C=4),
                ModulePatternInfoConfig(module_pattern="linear2", C=4),
                ModulePatternInfoConfig(module_pattern="embedding", C=4),
                ModulePatternInfoConfig(module_pattern="conv1d1", C=4),
                ModulePatternInfoConfig(module_pattern="conv1d2", C=4),
            ],
            identity_module_info=[ModulePatternInfoConfig(module_pattern="linear1", C=4)],
            ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
            batch_size=1,
            steps=1,
            lr_schedule=ScheduleConfig(start_val=1e-3),
            n_eval_steps=1,
            eval_batch_size=1,
            eval_freq=1,
            slow_eval_freq=1,
            loss_metric_configs=[ImportanceMinimalityLossConfig(coeff=1.0, pnorm=1.0, beta=0.5)],
            output_loss_type="mse",
            train_log_freq=1,
            n_mask_samples=1,
            task_config=TMSTaskConfig(
                task_name="tms",
                feature_probability=0.5,
                data_generation_type="exactly_one_active",
            ),
        )

        if config.identity_module_info is not None:
            insert_identity_operations_(
                target_model,
                identity_module_info=config.identity_module_info,
            )

        module_path_info = expand_module_patterns(target_model, config.all_module_info)
        cm = ComponentModel(
            target_model=target_model,
            module_path_info=module_path_info,
            ci_config=config.ci_config,
            pretrained_model_output_attr=config.pretrained_model_output_attr,
            sigmoid_type=config.sigmoid_type,
        )

        save_file(cm.state_dict(), comp_model_dir / "model.pth")
        save_file(config.model_dump(mode="json"), comp_model_dir / "final_config.yaml")

        cm_run_info = SPDRunInfo.from_path(comp_model_dir / "model.pth")
        cm_loaded = ComponentModel.from_run_info(cm_run_info)

        assert config == cm_run_info.config
        for k, v in cm_loaded.state_dict().items():
            torch.testing.assert_close(v, cm.state_dict()[k])


class TinyTarget(nn.Module):
    def __init__(
        self,
        vocab_size: int = 7,
        d_emb: int = 5,
        d_mid: int = 4,
        d_out: int = 3,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_emb)
        self.mlp = nn.Linear(d_emb, d_mid)
        self.out = nn.Linear(d_mid, d_out)

    @override
    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "..."]:
        x = self.embed(token_ids)
        x = self.mlp(x)
        x = self.out(x)
        return x


def tiny_target():
    tt = TinyTarget()
    tt.eval()
    tt.requires_grad_(False)
    return tt


BATCH_SIZE = 2


def test_patch_modules_unsupported_component_type_raises() -> None:
    model = tiny_target()
    wrong_module_path = "other_layer"

    with pytest.raises(AttributeError):
        ComponentModel._create_components(
            target_model=model,
            module_to_c={wrong_module_path: 2},
        )


def test_parallel_linear_shapes_and_forward():
    C = 3
    d_in = 4
    d_out = 5
    layer = ParallelLinear(C, d_in, d_out, nonlinearity="relu")
    x = torch.randn(BATCH_SIZE, C, d_in)
    y = layer(x)
    assert y.shape == (BATCH_SIZE, C, d_out)


@pytest.mark.parametrize("hidden_dims", [[8], [4, 3]])
def test_mlp_ci_fn_scalar_per_component(hidden_dims: list[int]):
    C = 5
    ci_fns = MLPCiFn(C=C, hidden_dims=hidden_dims)
    x = torch.randn(BATCH_SIZE, C)  # two items, C components
    y = ci_fns(x)
    assert y.shape == (BATCH_SIZE, C)


@pytest.mark.parametrize("hidden_dims", [[4], [6, 3]])
def test_vector_mlp_ci_fns(hidden_dims: list[int]):
    C = 3
    d_in = 10
    ci_fns = VectorMLPCiFn(C=C, input_dim=d_in, hidden_dims=hidden_dims)
    x = torch.randn(BATCH_SIZE, d_in)
    y = ci_fns(x)
    assert y.shape == (BATCH_SIZE, C)


@pytest.mark.parametrize("hidden_dims", [[], [7], [8, 5]])
def test_vector_shared_mlp_fn(hidden_dims: list[int]):
    C = 3
    d_in = 10
    ci_fn = VectorSharedMLPCiFn(C=C, input_dim=d_in, hidden_dims=hidden_dims)
    x = torch.randn(BATCH_SIZE, d_in)
    y = ci_fn(x)
    assert y.shape == (BATCH_SIZE, C)


def test_full_weight_delta_matches_target_behaviour():
    # GIVEN a component model
    target_model = tiny_target()

    target_module_paths = ["embed", "mlp", "out"]
    C = 4
    cm = ComponentModel(
        target_model=target_model,
        module_path_info=[ModulePathInfo(module_path=p, C=C) for p in target_module_paths],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    token_ids = torch.randint(
        low=0, high=target_model.embed.num_embeddings, size=(BATCH_SIZE,), dtype=torch.long
    )

    # WHEN we forward the component model with weight deltas and a weight delta mask of all 1s
    weight_deltas = cm.calc_weight_deltas()
    component_masks = {name: torch.ones(BATCH_SIZE, C) for name in target_module_paths}
    weight_deltas_and_masks = {
        name: (weight_deltas[name], torch.ones(BATCH_SIZE)) for name in target_module_paths
    }
    mask_infos = make_mask_infos(component_masks, weight_deltas_and_masks=weight_deltas_and_masks)
    out = cm(token_ids, mask_infos=mask_infos)

    # THEN the output matches the target model's output
    torch.testing.assert_close(out, target_model(token_ids))


def test_input_cache_captures_pre_weight_input():
    target_model = tiny_target()

    # GIVEN a component model
    target_module_paths = ["embed", "mlp"]

    cm = ComponentModel(
        target_model=target_model,
        module_path_info=[ModulePathInfo(module_path=p, C=2) for p in target_module_paths],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    # WHEN we forward the component model with input caching
    token_ids = torch.randint(
        low=0,
        high=target_model.embed.num_embeddings,
        size=(BATCH_SIZE,),
        dtype=torch.long,
    )
    out, cache = cm(token_ids, cache_type="input")

    # Output isn't altered
    torch.testing.assert_close(out, target_model(token_ids))

    # Captured inputs match the true pre-weight inputs

    assert cache["embed"].dtype == torch.long
    assert torch.equal(cache["embed"], token_ids)
    embed_out = target_model.embed(token_ids)

    assert cache["mlp"].shape == (BATCH_SIZE, target_model.mlp.in_features)
    torch.testing.assert_close(cache["mlp"], embed_out)


def test_weight_deltas():
    # GIVEN a component model
    target_model = tiny_target()
    target_module_paths = ["embed", "mlp", "out"]
    cm = ComponentModel(
        target_model=target_model,
        module_path_info=[ModulePathInfo(module_path=p, C=3) for p in target_module_paths],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    # THEN the weight deltas match the target weight
    deltas = cm.calc_weight_deltas()
    for name in target_module_paths:
        target_w = cm.target_weight(name)
        comp_w = cm.components[name].weight
        torch.testing.assert_close(target_w, comp_w + deltas[name])


def test_replacement_effects_fwd_pass():
    d_in = 10
    d_out = 20
    C = 30

    class OneLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias=False)

        @override
        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x)

    model = OneLayerModel()
    model.eval()
    model.requires_grad_(False)

    cm = ComponentModel(
        target_model=model,
        module_path_info=[ModulePathInfo(module_path="linear", C=C)],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    # WHEN we set the target model weights to be UV
    model.linear.weight.copy_(cm.components["linear"].weight)

    # AND we use all components
    input = torch.randn(BATCH_SIZE, d_in)
    use_all_components = ComponentsMaskInfo(component_mask=torch.ones(BATCH_SIZE, C))

    # THEN the model output matches the component model output
    model_out = model(input)
    cm_out_with_all_components = cm(input, mask_infos={"linear": use_all_components})
    torch.testing.assert_close(model_out, cm_out_with_all_components)

    # however, WHEN we double the values of the model weights
    model.linear.weight.mul_(2)

    # THEN the component-only output should be 1/2 the model output
    new_model_out = model(input)
    new_cm_out_with_all_components = cm(input, mask_infos={"linear": use_all_components})
    torch.testing.assert_close(new_model_out, new_cm_out_with_all_components * 2)


def test_replacing_identity():
    d = 10
    C = 20

    class IdentityLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(d, d, bias=False)
            nn.init.eye_(self.linear.weight)

        @override
        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x)

    # GIVEN a simple model that performs identity (so we can isolate the effects below)
    model = IdentityLayerModel()
    model.eval()
    model.requires_grad_(False)

    # with another prepended identity layer
    insert_identity_operations_(
        target_model=model,
        identity_module_info=[ModulePatternInfoConfig(module_pattern="linear", C=C)],
    )

    # wrapped in a component model that decomposes the prepended identity layer
    cm = ComponentModel(
        target_model=model,
        module_path_info=[ModulePathInfo(module_path="linear.pre_identity", C=C)],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    # and a random input
    input = torch.randn(BATCH_SIZE, d)

    # WHEN we forward with the model
    # THEN it should just act as the identity
    torch.testing.assert_close(model(input), input)
    torch.testing.assert_close(cm(input), input)

    # WHEN we forward with the identity components
    use_all_components = ComponentsMaskInfo(component_mask=torch.ones(BATCH_SIZE, C))

    cm_components_out = cm(input, mask_infos={"linear.pre_identity": use_all_components})

    # THEN it should modify the input
    assert not torch.allclose(cm_components_out, input)

    # BUT the original model output should be unchanged
    cm_target_out = cm(input)
    assert torch.allclose(cm_target_out, model(input))


def test_routing():
    d = 10
    C = 20

    class IdentityLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(d, d, bias=False)
            nn.init.eye_(self.linear.weight)

        @override
        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x)

    # GIVEN a simple model that performs identity (so we can isolate the effects below)
    model = IdentityLayerModel()
    model.eval()
    model.requires_grad_(False)

    # wrapped in a component model that decomposes the layer
    cm = ComponentModel(
        target_model=model,
        module_path_info=[ModulePathInfo(module_path="linear", C=C)],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    # and a random input
    input = torch.randn(BATCH_SIZE, d)

    # WHEN we forward with the model
    # THEN it should just act as the identity
    torch.testing.assert_close(model(input), input)
    torch.testing.assert_close(cm(input), input)

    # WHEN we forward with the components
    use_all_components = ComponentsMaskInfo(component_mask=torch.ones(BATCH_SIZE, C))

    cm_components_out = cm(input, mask_infos={"linear": use_all_components})

    # THEN it should modify the input
    assert not torch.allclose(cm_components_out, input)

    # but WHEN we forward with the components with routing:
    use_all_components_for_example_0 = ComponentsMaskInfo(
        component_mask=torch.ones(BATCH_SIZE, C),
        routing_mask=torch.tensor([True, False]),  # route to components only for example 0
    )

    cm_routed_out = cm(input, mask_infos={"linear": use_all_components_for_example_0})

    target_out = model(input)

    # THEN the output should be different for the first example (where it's routed to components)
    assert not torch.allclose(cm_routed_out[0], target_out[0])

    # but it should be the same for the second example (where it's not routed to components)
    assert torch.allclose(cm_routed_out[1], target_out[1])


def test_checkpoint_ci_config_mismatch_global_to_layerwise():
    """Test that loading a global CI checkpoint with layerwise config fails."""
    target_model = SimpleTestModel()
    target_model.eval()
    target_model.requires_grad_(False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        base_model_dir = base_dir / "test_model"
        base_model_dir.mkdir(parents=True, exist_ok=True)
        comp_model_dir = base_dir / "comp_model"
        comp_model_dir.mkdir(parents=True, exist_ok=True)

        base_model_path = base_model_dir / "model.pth"
        save_file(target_model.state_dict(), base_model_path)

        # Create and save a component model with GLOBAL CI
        config_global = Config(
            pretrained_model_class="tests.test_component_model.SimpleTestModel",
            pretrained_model_path=base_model_path,
            pretrained_model_name=None,
            module_info=[
                ModulePatternInfoConfig(module_pattern="linear1", C=4),
                ModulePatternInfoConfig(module_pattern="linear2", C=4),
            ],
            ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[4]),
            batch_size=1,
            steps=1,
            lr_schedule=ScheduleConfig(start_val=1e-3),
            n_eval_steps=1,
            eval_batch_size=1,
            eval_freq=1,
            slow_eval_freq=1,
            loss_metric_configs=[ImportanceMinimalityLossConfig(coeff=1.0, pnorm=1.0, beta=0.5)],
            output_loss_type="mse",
            train_log_freq=1,
            n_mask_samples=1,
            task_config=TMSTaskConfig(
                task_name="tms",
                feature_probability=0.5,
                data_generation_type="exactly_one_active",
            ),
        )

        module_path_info = expand_module_patterns(target_model, config_global.all_module_info)
        cm_global = ComponentModel(
            target_model=target_model,
            module_path_info=module_path_info,
            ci_config=config_global.ci_config,
            pretrained_model_output_attr=config_global.pretrained_model_output_attr,
            sigmoid_type=config_global.sigmoid_type,
        )

        # Save global CI checkpoint
        global_checkpoint_path = comp_model_dir / "global_model.pth"
        save_file(cm_global.state_dict(), global_checkpoint_path)
        save_file(config_global.model_dump(mode="json"), comp_model_dir / "final_config.yaml")

        # Now try to load it with LAYERWISE config - should fail
        config_layerwise = Config(
            pretrained_model_class="tests.test_component_model.SimpleTestModel",
            pretrained_model_path=base_model_path,
            pretrained_model_name=None,
            module_info=[
                ModulePatternInfoConfig(module_pattern="linear1", C=4),
                ModulePatternInfoConfig(module_pattern="linear2", C=4),
            ],
            ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
            batch_size=1,
            steps=1,
            lr_schedule=ScheduleConfig(start_val=1e-3),
            n_eval_steps=1,
            eval_batch_size=1,
            eval_freq=1,
            slow_eval_freq=1,
            loss_metric_configs=[ImportanceMinimalityLossConfig(coeff=1.0, pnorm=1.0, beta=0.5)],
            output_loss_type="mse",
            train_log_freq=1,
            n_mask_samples=1,
            task_config=TMSTaskConfig(
                task_name="tms",
                feature_probability=0.5,
                data_generation_type="exactly_one_active",
            ),
        )

        # Override the checkpoint path and config in the directory
        save_file(config_layerwise.model_dump(mode="json"), comp_model_dir / "final_config.yaml")

        cm_run_info = SPDRunInfo.from_path(global_checkpoint_path)
        # Update config to layerwise after loading run_info
        cm_run_info.config = config_layerwise

        with pytest.raises(
            AssertionError,
            match="Config specifies layerwise CI but checkpoint has no ci_fn._ci_fns keys",
        ):
            ComponentModel.from_run_info(cm_run_info)


def test_checkpoint_ci_config_mismatch_layerwise_to_global():
    """Test that loading a layerwise CI checkpoint with global config fails."""
    target_model = SimpleTestModel()
    target_model.eval()
    target_model.requires_grad_(False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        base_model_dir = base_dir / "test_model"
        base_model_dir.mkdir(parents=True, exist_ok=True)
        comp_model_dir = base_dir / "comp_model"
        comp_model_dir.mkdir(parents=True, exist_ok=True)

        base_model_path = base_model_dir / "model.pth"
        save_file(target_model.state_dict(), base_model_path)

        # Create and save a component model with LAYERWISE CI
        config_layerwise = Config(
            pretrained_model_class="tests.test_component_model.SimpleTestModel",
            pretrained_model_path=base_model_path,
            pretrained_model_name=None,
            module_info=[
                ModulePatternInfoConfig(module_pattern="linear1", C=4),
                ModulePatternInfoConfig(module_pattern="linear2", C=4),
            ],
            ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
            batch_size=1,
            steps=1,
            lr_schedule=ScheduleConfig(start_val=1e-3),
            n_eval_steps=1,
            eval_batch_size=1,
            eval_freq=1,
            slow_eval_freq=1,
            loss_metric_configs=[ImportanceMinimalityLossConfig(coeff=1.0, pnorm=1.0, beta=0.5)],
            output_loss_type="mse",
            train_log_freq=1,
            n_mask_samples=1,
            task_config=TMSTaskConfig(
                task_name="tms",
                feature_probability=0.5,
                data_generation_type="exactly_one_active",
            ),
        )

        module_path_info = expand_module_patterns(target_model, config_layerwise.all_module_info)
        cm_layerwise = ComponentModel(
            target_model=target_model,
            module_path_info=module_path_info,
            ci_config=config_layerwise.ci_config,
            pretrained_model_output_attr=config_layerwise.pretrained_model_output_attr,
            sigmoid_type=config_layerwise.sigmoid_type,
        )

        # Save layerwise CI checkpoint
        layerwise_checkpoint_path = comp_model_dir / "layerwise_model.pth"
        save_file(cm_layerwise.state_dict(), layerwise_checkpoint_path)
        save_file(config_layerwise.model_dump(mode="json"), comp_model_dir / "final_config.yaml")

        # Now try to load it with GLOBAL config - should fail
        config_global = Config(
            pretrained_model_class="tests.test_component_model.SimpleTestModel",
            pretrained_model_path=base_model_path,
            pretrained_model_name=None,
            module_info=[
                ModulePatternInfoConfig(module_pattern="linear1", C=4),
                ModulePatternInfoConfig(module_pattern="linear2", C=4),
            ],
            ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[4]),
            batch_size=1,
            steps=1,
            lr_schedule=ScheduleConfig(start_val=1e-3),
            n_eval_steps=1,
            eval_batch_size=1,
            eval_freq=1,
            slow_eval_freq=1,
            loss_metric_configs=[ImportanceMinimalityLossConfig(coeff=1.0, pnorm=1.0, beta=0.5)],
            output_loss_type="mse",
            train_log_freq=1,
            n_mask_samples=1,
            task_config=TMSTaskConfig(
                task_name="tms",
                feature_probability=0.5,
                data_generation_type="exactly_one_active",
            ),
        )

        # Override the checkpoint path and config in the directory
        save_file(config_global.model_dump(mode="json"), comp_model_dir / "final_config.yaml")

        cm_run_info = SPDRunInfo.from_path(layerwise_checkpoint_path)
        # Update config to global after loading run_info
        cm_run_info.config = config_global

        with pytest.raises(
            AssertionError,
            match="Config specifies global CI but checkpoint has no ci_fn._global_ci_fn keys",
        ):
            ComponentModel.from_run_info(cm_run_info)


# =============================================================================
# Global CI Function Tests
# =============================================================================


@pytest.mark.parametrize("hidden_dims", [[], [8], [16, 8]])
def test_global_shared_mlp_ci_fn_shapes_and_values(hidden_dims: list[int]):
    """Test GlobalSharedMLPCiFn produces correct output shapes and valid values."""
    layer_configs = {
        "layer1": (10, 5),  # (input_dim, C)
        "layer2": (20, 3),
        "layer3": (15, 7),
    }
    ci_fn = GlobalSharedMLPCiFn(layer_configs=layer_configs, hidden_dims=hidden_dims)

    inputs = {
        "layer1": torch.randn(BATCH_SIZE, 10),
        "layer2": torch.randn(BATCH_SIZE, 20),
        "layer3": torch.randn(BATCH_SIZE, 15),
    }
    outputs = ci_fn(inputs)

    # Check shapes
    assert outputs["layer1"].shape == (BATCH_SIZE, 5)
    assert outputs["layer2"].shape == (BATCH_SIZE, 3)
    assert outputs["layer3"].shape == (BATCH_SIZE, 7)

    # Check values are valid (not NaN, not Inf)
    for name, out in outputs.items():
        assert torch.isfinite(out).all(), f"Output {name} contains NaN or Inf"


def test_global_shared_mlp_ci_fn_sorted_layer_order():
    """Test that GlobalSharedMLPCiFn uses sorted layer ordering for determinism."""
    layer_configs = {
        "z_layer": (5, 2),
        "a_layer": (10, 3),
        "m_layer": (8, 4),
    }

    ci_fn = GlobalSharedMLPCiFn(layer_configs=layer_configs, hidden_dims=[16])

    # Layer order should be sorted alphabetically for deterministic concat/split
    assert ci_fn.layer_order == ["a_layer", "m_layer", "z_layer"]
    assert ci_fn.split_sizes == [3, 4, 2]  # C values in sorted order


def test_global_shared_mlp_ci_fn_different_inputs_produce_different_outputs():
    """Test that different inputs produce different CI outputs (not constant function)."""
    layer_configs = {
        "layer1": (10, 5),
        "layer2": (8, 3),
    }
    ci_fn = GlobalSharedMLPCiFn(layer_configs=layer_configs, hidden_dims=[16])

    inputs1 = {
        "layer1": torch.randn(BATCH_SIZE, 10),
        "layer2": torch.randn(BATCH_SIZE, 8),
    }
    inputs2 = {
        "layer1": torch.randn(BATCH_SIZE, 10),
        "layer2": torch.randn(BATCH_SIZE, 8),
    }

    outputs1 = ci_fn(inputs1)
    outputs2 = ci_fn(inputs2)

    # Outputs should differ for different inputs
    assert not torch.allclose(outputs1["layer1"], outputs2["layer1"])
    assert not torch.allclose(outputs1["layer2"], outputs2["layer2"])


def test_global_shared_mlp_ci_fn_gradient_flow():
    """Test that gradients flow through GlobalSharedMLPCiFn and are meaningful."""
    layer_configs = {
        "layer1": (10, 5),
        "layer2": (8, 3),
    }
    ci_fn = GlobalSharedMLPCiFn(layer_configs=layer_configs, hidden_dims=[16])

    inputs = {
        "layer1": torch.randn(BATCH_SIZE, 10, requires_grad=True),
        "layer2": torch.randn(BATCH_SIZE, 8, requires_grad=True),
    }
    outputs = ci_fn(inputs)

    loss = torch.stack([out.sum() for out in outputs.values()]).sum()
    loss.backward()

    # Check gradients exist for inputs and are meaningful
    assert inputs["layer1"].grad is not None
    assert inputs["layer2"].grad is not None
    assert torch.isfinite(inputs["layer1"].grad).all()
    assert torch.isfinite(inputs["layer2"].grad).all()
    assert inputs["layer1"].grad.abs().sum() > 0, "Input gradients should be non-zero"
    assert inputs["layer2"].grad.abs().sum() > 0, "Input gradients should be non-zero"

    # Check gradients exist for parameters and are meaningful
    for name, param in ci_fn.named_parameters():
        assert param.grad is not None, f"Param {name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"Param {name} has NaN/Inf gradient"
        assert param.grad.abs().sum() > 0, f"Param {name} has zero gradient"


def test_global_shared_mlp_ci_fn_with_seq_dim():
    """Test GlobalSharedMLPCiFn with sequence dimension produces valid outputs."""
    seq_len = 5
    layer_configs = {
        "layer1": (10, 4),
        "layer2": (8, 3),
    }
    ci_fn = GlobalSharedMLPCiFn(layer_configs=layer_configs, hidden_dims=[16])

    inputs = {
        "layer1": torch.randn(BATCH_SIZE, seq_len, 10),
        "layer2": torch.randn(BATCH_SIZE, seq_len, 8),
    }
    outputs = ci_fn(inputs)

    # Check shapes
    assert outputs["layer1"].shape == (BATCH_SIZE, seq_len, 4)
    assert outputs["layer2"].shape == (BATCH_SIZE, seq_len, 3)

    # Check values are valid
    for name, out in outputs.items():
        assert torch.isfinite(out).all(), f"Output {name} contains NaN or Inf"


def test_global_shared_mlp_ci_fn_single_component():
    """Test GlobalSharedMLPCiFn with C=1 edge case."""
    layer_configs = {
        "layer1": (10, 1),
        "layer2": (8, 1),
    }
    ci_fn = GlobalSharedMLPCiFn(layer_configs=layer_configs, hidden_dims=[4])

    inputs = {
        "layer1": torch.randn(BATCH_SIZE, 10),
        "layer2": torch.randn(BATCH_SIZE, 8),
    }
    outputs = ci_fn(inputs)

    assert outputs["layer1"].shape == (BATCH_SIZE, 1)
    assert outputs["layer2"].shape == (BATCH_SIZE, 1)
    assert torch.isfinite(outputs["layer1"]).all()
    assert torch.isfinite(outputs["layer2"]).all()


def test_global_shared_mlp_ci_fn_single_layer():
    """Test GlobalSharedMLPCiFn with single layer edge case."""
    layer_configs = {"only_layer": (10, 5)}
    ci_fn = GlobalSharedMLPCiFn(layer_configs=layer_configs, hidden_dims=[8])

    inputs = {"only_layer": torch.randn(BATCH_SIZE, 10)}
    outputs = ci_fn(inputs)

    assert outputs["only_layer"].shape == (BATCH_SIZE, 5)
    assert torch.isfinite(outputs["only_layer"]).all()


def test_global_shared_transformer_ci_fn_shapes_and_values():
    """Test GlobalSharedTransformerCiFn produces correct output shapes and valid values."""
    layer_configs = {
        "layer1": TargetLayerConfig(input_dim=10, C=5),
        "layer2": TargetLayerConfig(input_dim=20, C=3),
        "layer3": TargetLayerConfig(input_dim=15, C=7),
    }
    ci_fn = GlobalSharedTransformerCiFn(
        target_model_layer_configs=layer_configs,
        d_model=8,
        n_layers=2,
        n_heads=2,
        mlp_hidden_dims=[16],
    )

    inputs = {
        "layer1": torch.randn(BATCH_SIZE, 10),
        "layer2": torch.randn(BATCH_SIZE, 20),
        "layer3": torch.randn(BATCH_SIZE, 15),
    }
    outputs = ci_fn(inputs)

    # Check shapes
    assert outputs["layer1"].shape == (BATCH_SIZE, 5)
    assert outputs["layer2"].shape == (BATCH_SIZE, 3)
    assert outputs["layer3"].shape == (BATCH_SIZE, 7)

    # Check values are valid (not NaN, not Inf)
    for name, out in outputs.items():
        assert torch.isfinite(out).all(), f"Output {name} contains NaN or Inf"


def test_global_shared_transformer_ci_fn_with_seq_dim():
    """Test GlobalSharedTransformerCiFn with sequence dimension produces valid outputs."""
    seq_len = 5
    layer_configs = {
        "layer1": TargetLayerConfig(input_dim=10, C=4),
        "layer2": TargetLayerConfig(input_dim=8, C=3),
    }
    ci_fn = GlobalSharedTransformerCiFn(
        target_model_layer_configs=layer_configs,
        d_model=8,
        n_layers=3,
        n_heads=2,
        mlp_hidden_dims=[16],
    )

    inputs = {
        "layer1": torch.randn(BATCH_SIZE, seq_len, 10),
        "layer2": torch.randn(BATCH_SIZE, seq_len, 8),
    }
    outputs = ci_fn(inputs)

    # Check shapes
    assert outputs["layer1"].shape == (BATCH_SIZE, seq_len, 4)
    assert outputs["layer2"].shape == (BATCH_SIZE, seq_len, 3)

    # Check values are valid
    for name, out in outputs.items():
        assert torch.isfinite(out).all(), f"Output {name} contains NaN or Inf"


def test_component_model_with_global_ci():
    """Test ComponentModel instantiation and forward with global CI config."""
    target_model = tiny_target()

    target_module_paths = ["mlp", "out"]
    C = 4
    cm = ComponentModel(
        target_model=target_model,
        module_path_info=[ModulePathInfo(module_path=p, C=C) for p in target_module_paths],
        ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[16]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    assert isinstance(cm.ci_fn, GlobalCiFnWrapper)
    assert isinstance(cm.ci_fn._global_ci_fn, GlobalSharedMLPCiFn)

    # Forward pass should work and match target
    token_ids = torch.randint(
        low=0, high=target_model.embed.num_embeddings, size=(BATCH_SIZE,), dtype=torch.long
    )
    out = cm(token_ids)
    torch.testing.assert_close(out, target_model(token_ids))


def test_component_model_global_ci_calc_causal_importances():
    """Test causal importance calculation with global CI produces valid bounded outputs."""
    target_model = tiny_target()

    target_module_paths = ["mlp", "out"]
    C = 4
    cm = ComponentModel(
        target_model=target_model,
        module_path_info=[ModulePathInfo(module_path=p, C=C) for p in target_module_paths],
        ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[16]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    token_ids = torch.randint(
        low=0, high=target_model.embed.num_embeddings, size=(BATCH_SIZE,), dtype=torch.long
    )

    _, cache = cm(token_ids, cache_type="input")

    ci_outputs = cm.calc_causal_importances(
        pre_weight_acts=cache,
        sampling="continuous",
        detach_inputs=False,
    )

    for path in target_module_paths:
        # Check shapes
        assert ci_outputs.lower_leaky[path].shape == (BATCH_SIZE, C)
        assert ci_outputs.upper_leaky[path].shape == (BATCH_SIZE, C)
        assert ci_outputs.pre_sigmoid[path].shape == (BATCH_SIZE, C)

        # Check bounds (leaky sigmoids allow values slightly outside [0, 1])
        # lower_leaky: bounded to [0, 1], can be negative with small leak
        # upper_leaky: bounded to [0, inf), can exceed 1 with small leak
        assert (ci_outputs.lower_leaky[path] >= 0).all(), f"{path} lower_leaky < 0"
        assert (ci_outputs.lower_leaky[path] <= 1.0).all(), f"{path} lower_leaky > 1"
        assert (ci_outputs.upper_leaky[path] >= 0).all(), f"{path} upper_leaky < 0"
        # upper_leaky can exceed 1.0 due to leaky behavior (1 + alpha*(x-1) when x>1)

        # Check values are finite
        assert torch.isfinite(ci_outputs.pre_sigmoid[path]).all(), f"{path} pre_sigmoid has NaN/Inf"


def test_component_model_global_ci_different_inputs_different_ci():
    """Test that different inputs produce different CI values (CI is input-dependent)."""
    target_model = tiny_target()

    target_module_paths = ["mlp", "out"]
    C = 4
    cm = ComponentModel(
        target_model=target_model,
        module_path_info=[ModulePathInfo(module_path=p, C=C) for p in target_module_paths],
        ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[16]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    # Two different token inputs
    token_ids_1 = torch.tensor([0, 1], dtype=torch.long)
    token_ids_2 = torch.tensor([2, 3], dtype=torch.long)

    _, cache1 = cm(token_ids_1, cache_type="input")
    _, cache2 = cm(token_ids_2, cache_type="input")

    ci1 = cm.calc_causal_importances(cache1, sampling="continuous")
    ci2 = cm.calc_causal_importances(cache2, sampling="continuous")

    # CI values should differ for different inputs
    for path in target_module_paths:
        assert not torch.allclose(ci1.pre_sigmoid[path], ci2.pre_sigmoid[path]), (
            f"CI for {path} should differ for different inputs"
        )


def test_component_model_global_ci_binomial_sampling():
    """Test global CI with binomial sampling produces valid binary masks."""
    target_model = tiny_target()

    target_module_paths = ["mlp", "out"]
    C = 4
    cm = ComponentModel(
        target_model=target_model,
        module_path_info=[ModulePathInfo(module_path=p, C=C) for p in target_module_paths],
        ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[16]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    token_ids = torch.randint(0, target_model.embed.num_embeddings, size=(BATCH_SIZE,))
    _, cache = cm(token_ids, cache_type="input")

    ci = cm.calc_causal_importances(cache, sampling="binomial")

    for path in target_module_paths:
        assert ci.lower_leaky[path].shape == (BATCH_SIZE, C)
        assert torch.isfinite(ci.lower_leaky[path]).all()
        assert torch.isfinite(ci.upper_leaky[path]).all()


def test_component_model_global_ci_with_embeddings():
    """Test global CI with embedding layers produces valid outputs."""
    target_model = tiny_target()

    target_module_paths = ["embed", "mlp", "out"]
    C = 4
    cm = ComponentModel(
        target_model=target_model,
        module_path_info=[ModulePathInfo(module_path=p, C=C) for p in target_module_paths],
        ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[16]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    assert isinstance(cm.ci_fn, GlobalCiFnWrapper)

    token_ids = torch.randint(
        low=0, high=target_model.embed.num_embeddings, size=(BATCH_SIZE,), dtype=torch.long
    )

    _, cache = cm(token_ids, cache_type="input")

    ci_outputs = cm.calc_causal_importances(
        pre_weight_acts=cache,
        sampling="continuous",
        detach_inputs=False,
    )

    # Check all layers including embedding
    for path in target_module_paths:
        assert ci_outputs.lower_leaky[path].shape == (BATCH_SIZE, C)
        assert (ci_outputs.lower_leaky[path] >= 0).all()
        assert (ci_outputs.lower_leaky[path] <= 1.0).all()
        assert torch.isfinite(ci_outputs.pre_sigmoid[path]).all()


def test_component_model_global_ci_gradient_flow():
    """Test gradient flow through global CI - gradients are non-zero and finite."""
    target_model = tiny_target()

    target_module_paths = ["mlp", "out"]
    C = 4
    cm = ComponentModel(
        target_model=target_model,
        module_path_info=[ModulePathInfo(module_path=p, C=C) for p in target_module_paths],
        ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[16]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    token_ids = torch.randint(
        low=0, high=target_model.embed.num_embeddings, size=(BATCH_SIZE,), dtype=torch.long
    )

    _, cache = cm(token_ids, cache_type="input")

    ci_outputs = cm.calc_causal_importances(
        pre_weight_acts=cache,
        sampling="continuous",
        detach_inputs=False,
    )

    ci_loss = torch.stack([ci.sum() for ci in ci_outputs.lower_leaky.values()]).sum()
    ci_loss.backward()

    # Check that global CI function has meaningful gradients
    assert isinstance(cm.ci_fn, GlobalCiFnWrapper)
    for name, param in cm.ci_fn._global_ci_fn.named_parameters():
        assert param.grad is not None, f"Param {name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"Param {name} has NaN/Inf gradient"
        assert param.grad.abs().sum() > 0, f"Param {name} has zero gradient"


def test_component_model_global_ci_detach_inputs_blocks_gradients():
    """Test that detach_inputs=True blocks gradient flow to CI function."""
    target_model = tiny_target()

    target_module_paths = ["mlp", "out"]
    C = 4
    cm = ComponentModel(
        target_model=target_model,
        module_path_info=[ModulePathInfo(module_path=p, C=C) for p in target_module_paths],
        ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[16]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    token_ids = torch.randint(
        low=0, high=target_model.embed.num_embeddings, size=(BATCH_SIZE,), dtype=torch.long
    )

    _, cache = cm(token_ids, cache_type="input")

    # With detach_inputs=True, gradients should still flow to CI fn params
    # but from the CI loss, not from upstream
    ci_outputs = cm.calc_causal_importances(
        pre_weight_acts=cache,
        sampling="continuous",
        detach_inputs=True,  # Detach inputs
    )

    ci_loss = torch.stack([ci.sum() for ci in ci_outputs.lower_leaky.values()]).sum()
    ci_loss.backward()

    # CI function should still get gradients (from its own computation)
    assert isinstance(cm.ci_fn, GlobalCiFnWrapper)
    for param in cm.ci_fn._global_ci_fn.parameters():
        assert param.grad is not None


def test_component_model_global_ci_masking_zeros():
    """Test that zero masks actually zero out component contributions."""
    target_model = tiny_target()

    target_module_paths = ["mlp", "out"]
    C = 4
    cm = ComponentModel(
        target_model=target_model,
        module_path_info=[ModulePathInfo(module_path=p, C=C) for p in target_module_paths],
        ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[16]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    token_ids = torch.randint(
        low=0, high=target_model.embed.num_embeddings, size=(BATCH_SIZE,), dtype=torch.long
    )

    weight_deltas = cm.calc_weight_deltas()

    # All ones mask - should match target
    all_ones_masks = {name: torch.ones(BATCH_SIZE, C) for name in target_module_paths}
    weight_deltas_and_masks_ones = {
        name: (weight_deltas[name], torch.ones(BATCH_SIZE)) for name in target_module_paths
    }
    mask_infos_ones = make_mask_infos(
        all_ones_masks, weight_deltas_and_masks=weight_deltas_and_masks_ones
    )
    out_ones = cm(token_ids, mask_infos=mask_infos_ones)

    # All zeros mask - should be different from all ones
    all_zeros_masks = {name: torch.zeros(BATCH_SIZE, C) for name in target_module_paths}
    weight_deltas_and_masks_zeros = {
        name: (weight_deltas[name], torch.ones(BATCH_SIZE)) for name in target_module_paths
    }
    mask_infos_zeros = make_mask_infos(
        all_zeros_masks, weight_deltas_and_masks=weight_deltas_and_masks_zeros
    )
    out_zeros = cm(token_ids, mask_infos=mask_infos_zeros)

    # Outputs should differ
    assert not torch.allclose(out_ones, out_zeros), (
        "Zero masks should produce different output than one masks"
    )


def test_component_model_global_ci_partial_masking():
    """Test that partial masks produce outputs between fully masked extremes."""
    target_model = tiny_target()

    target_module_paths = ["mlp", "out"]
    C = 4
    cm = ComponentModel(
        target_model=target_model,
        module_path_info=[ModulePathInfo(module_path=p, C=C) for p in target_module_paths],
        ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[16]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    token_ids = torch.randint(
        low=0, high=target_model.embed.num_embeddings, size=(BATCH_SIZE,), dtype=torch.long
    )

    weight_deltas = cm.calc_weight_deltas()

    # Partial mask (0.5 for all)
    partial_masks = {name: torch.full((BATCH_SIZE, C), 0.5) for name in target_module_paths}
    weight_deltas_and_masks = {
        name: (weight_deltas[name], torch.ones(BATCH_SIZE)) for name in target_module_paths
    }
    mask_infos = make_mask_infos(partial_masks, weight_deltas_and_masks=weight_deltas_and_masks)
    out_partial = cm(token_ids, mask_infos=mask_infos)

    # Should produce valid output
    assert torch.isfinite(out_partial).all(), "Partial masking produced NaN/Inf"


def test_component_model_global_ci_weight_deltas_all_ones_matches_target():
    """Test that all-ones mask with weight deltas matches target model output."""
    target_model = tiny_target()

    target_module_paths = ["mlp", "out"]
    C = 4
    cm = ComponentModel(
        target_model=target_model,
        module_path_info=[ModulePathInfo(module_path=p, C=C) for p in target_module_paths],
        ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[16]),
        pretrained_model_output_attr=None,
        sigmoid_type="leaky_hard",
    )

    token_ids = torch.randint(
        low=0, high=target_model.embed.num_embeddings, size=(BATCH_SIZE,), dtype=torch.long
    )

    weight_deltas = cm.calc_weight_deltas()
    component_masks = {name: torch.ones(BATCH_SIZE, C) for name in target_module_paths}
    weight_deltas_and_masks = {
        name: (weight_deltas[name], torch.ones(BATCH_SIZE)) for name in target_module_paths
    }
    mask_infos = make_mask_infos(component_masks, weight_deltas_and_masks=weight_deltas_and_masks)
    out = cm(token_ids, mask_infos=mask_infos)

    torch.testing.assert_close(out, target_model(token_ids))


def test_global_ci_save_and_load():
    """Test saving and loading a model with global CI preserves functionality."""
    target_model = SimpleTestModel()
    target_model.eval()
    target_model.requires_grad_(False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        base_model_dir = base_dir / "test_model"
        base_model_dir.mkdir(parents=True, exist_ok=True)
        comp_model_dir = base_dir / "comp_model"
        comp_model_dir.mkdir(parents=True, exist_ok=True)

        base_model_path = base_model_dir / "model.pth"
        save_file(target_model.state_dict(), base_model_path)

        config = Config(
            pretrained_model_class="tests.test_component_model.SimpleTestModel",
            pretrained_model_path=base_model_path,
            pretrained_model_name=None,
            module_info=[
                ModulePatternInfoConfig(module_pattern="linear1", C=4),
                ModulePatternInfoConfig(module_pattern="linear2", C=4),
            ],
            ci_config=GlobalCiConfig(fn_type="global_shared_mlp", hidden_dims=[8]),
            batch_size=1,
            steps=1,
            lr_schedule=ScheduleConfig(start_val=1e-3),
            n_eval_steps=1,
            eval_batch_size=1,
            eval_freq=1,
            slow_eval_freq=1,
            loss_metric_configs=[ImportanceMinimalityLossConfig(coeff=1.0, pnorm=1.0, beta=0.5)],
            output_loss_type="mse",
            train_log_freq=1,
            n_mask_samples=1,
            task_config=TMSTaskConfig(
                task_name="tms",
                feature_probability=0.5,
                data_generation_type="exactly_one_active",
            ),
        )

        module_path_info = expand_module_patterns(target_model, config.all_module_info)
        cm = ComponentModel(
            target_model=target_model,
            module_path_info=module_path_info,
            ci_config=config.ci_config,
            pretrained_model_output_attr=config.pretrained_model_output_attr,
            sigmoid_type=config.sigmoid_type,
        )

        assert isinstance(cm.ci_fn, GlobalCiFnWrapper)

        save_file(cm.state_dict(), comp_model_dir / "model.pth")
        save_file(config.model_dump(mode="json"), comp_model_dir / "final_config.yaml")

        # Load and verify
        cm_run_info = SPDRunInfo.from_path(comp_model_dir / "model.pth")
        cm_loaded = ComponentModel.from_run_info(cm_run_info)

        assert isinstance(cm_loaded.ci_fn, GlobalCiFnWrapper)

        # Verify state dict matches
        for k, v in cm_loaded.state_dict().items():
            torch.testing.assert_close(v, cm.state_dict()[k])

        # Verify global CI function weights specifically
        global_ci_fn = cm.ci_fn._global_ci_fn
        global_ci_fn_loaded = cm_loaded.ci_fn._global_ci_fn
        assert isinstance(global_ci_fn, GlobalSharedMLPCiFn)
        assert isinstance(global_ci_fn_loaded, GlobalSharedMLPCiFn)
        assert global_ci_fn_loaded.layer_order == global_ci_fn.layer_order
        for p1, p2 in zip(global_ci_fn.parameters(), global_ci_fn_loaded.parameters(), strict=True):
            torch.testing.assert_close(p1, p2)

        # Verify global CI function produces same outputs
        test_acts = {
            name: torch.randn(BATCH_SIZE, global_ci_fn.layer_configs[name][0])
            for name in global_ci_fn.layer_order
        }
        outputs_orig = global_ci_fn(test_acts)
        outputs_loaded = global_ci_fn_loaded(test_acts)
        for name in global_ci_fn.layer_order:
            torch.testing.assert_close(outputs_orig[name], outputs_loaded[name])
