"""API endpoint tests for spd.app.backend.server.

These tests bypass slow operations (W&B model loading, large data loaders) by:
1. Manually constructing app state with a fresh randomly-initialized model
2. Using small data splits and short sequences
3. Using an in-memory SQLite database
"""

import json
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.database import PromptAttrDB
from spd.app.backend.routers import graphs as graphs_router
from spd.app.backend.routers import intervention as intervention_router
from spd.app.backend.routers import runs as runs_router
from spd.app.backend.server import app
from spd.app.backend.state import RunState, StateManager
from spd.configs import (
    Config,
    LayerwiseCiConfig,
    LMTaskConfig,
    ModulePatternInfoConfig,
    ScheduleConfig,
)
from spd.models.component_model import ComponentModel
from spd.pretrain.models.gpt2_simple import GPT2Simple, GPT2SimpleConfig
from spd.topology import TransformerTopology, get_sources_by_target
from spd.utils.module_utils import expand_module_patterns

DEVICE = "cpu"


@pytest.fixture
def app_with_state():
    """Set up app state with a fresh randomly-initialized model.

    This fixture:
    1. Creates an in-memory SQLite database
    2. Creates a fake "run" in the database
    3. Creates a fresh GPT2Simple model (randomly initialized, 1 layer)
    4. Wraps it in a fresh ComponentModel (randomly initialized)
    5. Constructs a test Config with small data split and short sequences
    6. Computes sources_by_target mapping
    7. Creates RunState and sets it on StateManager (with hardcoded token_strings, no real tokenizer)
    8. Returns FastAPI's TestClient
    """
    # Reset StateManager singleton for clean test state
    StateManager.reset()

    # Patch DEVICE in all router modules to use CPU for tests
    with (
        mock.patch.object(graphs_router, "DEVICE", DEVICE),
        mock.patch.object(intervention_router, "DEVICE", DEVICE),
        mock.patch.object(runs_router, "DEVICE", DEVICE),
    ):
        db = PromptAttrDB(db_path=Path(":memory:"), check_same_thread=False)
        db.init_schema()

        run_id = db.create_run("wandb:test/test/runs/testrun1")
        run = db.get_run(run_id)
        assert run is not None

        model_config = GPT2SimpleConfig(
            model_type="GPT2Simple",
            block_size=16,
            vocab_size=4019,  # Match tokenizer
            n_layer=1,
            n_head=2,
            n_embd=32,
            flash_attention=False,
        )
        target_model = GPT2Simple(model_config)
        target_model.eval()
        target_model.requires_grad_(False)

        target_module_patterns = [
            "h.*.mlp.c_fc",
            "h.*.mlp.down_proj",
            "h.*.attn.q_proj",
            "h.*.attn.k_proj",
            "h.*.attn.v_proj",
            "h.*.attn.o_proj",
        ]
        C = 8

        config = Config(
            n_mask_samples=1,
            ci_config=LayerwiseCiConfig(fn_type="shared_mlp", hidden_dims=[16]),
            sampling="continuous",
            sigmoid_type="leaky_hard",
            module_info=[
                ModulePatternInfoConfig(module_pattern=p, C=C) for p in target_module_patterns
            ],
            pretrained_model_class="spd.pretrain.models.gpt2_simple.GPT2Simple",
            pretrained_model_output_attr="idx_0",
            tokenizer_name="SimpleStories/test-SimpleStories-gpt2-1.25M",
            output_loss_type="kl",
            lr_schedule=ScheduleConfig(start_val=1e-3),
            steps=1,
            batch_size=1,
            eval_batch_size=1,
            n_eval_steps=1,
            eval_freq=1,
            slow_eval_freq=1,
            train_log_freq=1,
            task_config=LMTaskConfig(
                task_name="lm",
                max_seq_len=3,  # Short sequences
                dataset_name="SimpleStories/SimpleStories",
                column_name="story",
                train_data_split="test[:20]",  # Only 20 samples
                eval_data_split="test[:20]",
            ),
        )
        module_path_info = expand_module_patterns(target_model, config.module_info)
        model = ComponentModel(
            target_model=target_model,
            module_path_info=module_path_info,
            ci_config=config.ci_config,
            pretrained_model_output_attr=config.pretrained_model_output_attr,
            sigmoid_type=config.sigmoid_type,
        )
        model.eval()
        topology = TransformerTopology(model.target_model)
        sources_by_target = get_sources_by_target(
            model=model, topology=topology, device=DEVICE, sampling=config.sampling
        )

        from transformers import AutoTokenizer
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        hf_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        assert isinstance(hf_tokenizer, PreTrainedTokenizerBase)
        tokenizer = AppTokenizer(hf_tokenizer)

        run_state = RunState(
            run=run,
            model=model,
            topology=topology,
            context_length=1,
            tokenizer=tokenizer,
            sources_by_target=sources_by_target,
            config=config,
            harvest=None,
            interp=None,
            attributions=None,
            graph_interp=None,
        )

        manager = StateManager.get()
        manager.initialize(db)
        manager.run_state = run_state

        yield TestClient(app)

        manager.close()
        StateManager.reset()


@pytest.fixture
def app_with_prompt(app_with_state: TestClient) -> tuple[TestClient, int]:
    """Extends app_with_state with a pre-created prompt for graph tests.

    Returns:
        Tuple of (TestClient, prompt_id)
    """
    manager = StateManager.get()
    assert manager.run_state is not None
    prompt_id = manager.db.add_custom_prompt(
        run_id=manager.run_state.run.id,
        token_ids=[0, 2, 1],
        context_length=manager.run_state.context_length,
    )
    return app_with_state, prompt_id


# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------


def test_health_check(app_with_state: TestClient):
    """Test that health endpoint returns ok status."""
    response = app_with_state.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# -----------------------------------------------------------------------------
# Run Management
# -----------------------------------------------------------------------------


def test_get_status(app_with_state: TestClient):
    """Test getting current server status with loaded run."""
    response = app_with_state.get("/api/status")
    assert response.status_code == 200
    status = response.json()
    assert status["wandb_path"] == "wandb:test/test/runs/testrun1"
    assert "config_yaml" in status


# -----------------------------------------------------------------------------
# Compute
# -----------------------------------------------------------------------------


def test_compute_graph(app_with_prompt: tuple[TestClient, int]):
    """Test computing attribution graph for a prompt."""
    client, prompt_id = app_with_prompt
    response = client.post(
        "/api/graphs",
        params={"prompt_id": prompt_id, "normalize": "none", "ci_threshold": 0.0},
    )
    assert response.status_code == 200

    # Parse SSE stream
    lines = response.text.strip().split("\n")
    events = [line for line in lines if line.startswith("data:")]
    assert len(events) >= 1

    # Final event should be complete with graph data
    final_data = json.loads(events[-1].replace("data: ", ""))
    assert final_data["type"] == "complete"

    data = final_data["data"]
    assert "edges" in data
    assert "tokens" in data
    assert "outputProbs" in data


def test_run_and_save_intervention_without_text(app_with_prompt: tuple[TestClient, int]):
    """Run-and-save intervention should use graph-linked prompt tokens (no text in request)."""
    client, prompt_id = app_with_prompt

    graph_response = client.post(
        "/api/graphs",
        params={"prompt_id": prompt_id, "normalize": "none", "ci_threshold": 0.0},
    )
    assert graph_response.status_code == 200
    events = [line for line in graph_response.text.strip().split("\n") if line.startswith("data:")]
    final_data = json.loads(events[-1].replace("data: ", ""))
    graph_data = final_data["data"]
    graph_id = graph_data["id"]

    selected_nodes = [
        key
        for key, ci in graph_data["nodeCiVals"].items()
        if not key.startswith("embed:") and not key.startswith("output:") and ci > 0
    ]
    assert len(selected_nodes) > 0

    request = {
        "graph_id": graph_id,
        "selected_nodes": selected_nodes[: min(5, len(selected_nodes))],
        "top_k": 5,
        "adv_pgd": {"n_steps": 1, "step_size": 1.0},
    }
    response = client.post("/api/intervention/run", json=request)
    assert response.status_code == 200
    body = response.json()
    assert body["selected_nodes"] == request["selected_nodes"]
    result = body["result"]
    assert len(result["input_tokens"]) > 0
    assert len(result["ci"]) > 0
    assert len(result["stochastic"]) > 0
    assert len(result["adversarial"]) > 0
    assert result["target_sans"] is None
    assert "ci_loss" in result
    assert "stochastic_loss" in result
    assert "adversarial_loss" in result
    assert result["target_sans_loss"] is None


# -----------------------------------------------------------------------------
# Streaming: Prompt Generation
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------


def test_get_prompts_initially_empty(app_with_state: TestClient):
    """Test that prompts list is initially empty."""
    response = app_with_state.get("/api/prompts")
    assert response.status_code == 200
    prompts = response.json()
    assert len(prompts) == 0


def test_get_prompts_after_adding(app_with_state: TestClient):
    """Test getting prompts after adding via database."""
    manager = StateManager.get()
    assert manager.run_state is not None
    manager.db.add_custom_prompt(
        run_id=manager.run_state.run.id,
        token_ids=[0, 2, 1],
        context_length=manager.run_state.context_length,
    )
    manager.db.add_custom_prompt(
        run_id=manager.run_state.run.id,
        token_ids=[1, 3, 2],
        context_length=manager.run_state.context_length,
    )

    response = app_with_state.get("/api/prompts")
    assert response.status_code == 200
    prompts = response.json()
    assert len(prompts) == 2


# -----------------------------------------------------------------------------
# Activation Contexts
# -----------------------------------------------------------------------------


def test_activation_contexts_not_found_initially(app_with_state: TestClient):
    """Test that activation contexts return 404 when not generated."""
    response = app_with_state.get("/api/activation_contexts/summary")
    assert response.status_code == 404


# -----------------------------------------------------------------------------
# Optimized Compute (Streaming)
# -----------------------------------------------------------------------------


@pytest.mark.slow
def test_compute_optimized_stream(app_with_prompt: tuple[TestClient, int]):
    """Test streaming optimized attribution computation."""
    client, prompt_id = app_with_prompt
    response = client.post(
        "/api/graphs/optimized/stream",
        params={
            "prompt_id": prompt_id,
            "label_token": 2,
            "imp_min_coeff": 0.01,
            "loss_type": "ce",
            "loss_coeff": 1.0,
            "loss_position": 2,
            "steps": 5,  # Very few steps for testing
            "pnorm": 0.5,
            "beta": 0.5,
            "normalize": "none",
            "ci_threshold": 0.0,
            "output_prob_threshold": 0.01,
            "mask_type": "stochastic",
        },
    )
    assert response.status_code == 200

    events = [line for line in response.text.strip().split("\n") if line.startswith("data:")]
    assert len(events) >= 1
