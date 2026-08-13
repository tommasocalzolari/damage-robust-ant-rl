"""Tests for the PPO training command."""

import argparse

import pytest
import torch

from damage_robust_ant.train import (
    _checkpoint_settings,
    _damage_configuration,
    _ppo_configuration,
    _prepare_output_dir,
    make_ppo,
    make_training_env,
    parse_args,
)


def test_training_argument_defaults(tmp_path) -> None:
    """The command defaults match the fixed main experiment."""
    args = parse_args(
        ["--condition", "nominal", "--output-dir", str(tmp_path / "run")]
    )
    assert args.seed == 0
    assert args.timesteps == 1_000_000
    assert args.num_envs == 4
    assert args.learning_rate == 3e-4
    assert args.clip_range == 0.2


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--timesteps", "0"),
        ("--num-envs", "0"),
        ("--learning-rate", "0"),
        ("--clip-range", "0"),
    ],
)
def test_training_rejects_nonpositive_values(tmp_path, option, value) -> None:
    """Counts and PPO scalar arguments must be positive."""
    arguments = [
        "--condition",
        "robust",
        "--output-dir",
        str(tmp_path / "run"),
        option,
        value,
    ]
    with pytest.raises(SystemExit):
        parse_args(arguments)


@pytest.mark.parametrize(
    ("condition", "expected_mode"),
    [("nominal", "nominal"), ("robust", "random")],
)
def test_training_environment_uses_requested_damage_mode(
    tmp_path,
    condition,
    expected_mode,
) -> None:
    """Only the environment damage mode differs between conditions."""
    env = make_training_env(condition, 2, 4, tmp_path / condition)
    try:
        assert env.num_envs == 2
        assert env.get_attr("mode") == [expected_mode, expected_mode]
        assert env.action_space.shape == (8,)
        assert env.observation_space.shape == (105,)
    finally:
        env.close()


def test_ppo_uses_fixed_configuration(tmp_path) -> None:
    """PPO uses the required networks and explicit shared settings."""
    env = make_training_env("nominal", 1, 0, tmp_path / "monitor")
    try:
        model = make_ppo(env, 3e-4, 0.2, 0, tmp_path / "tensorboard")
        policy_layers = [
            layer.out_features
            for layer in model.policy.mlp_extractor.policy_net
            if isinstance(layer, torch.nn.Linear)
        ]
        value_layers = [
            layer.out_features
            for layer in model.policy.mlp_extractor.value_net
            if isinstance(layer, torch.nn.Linear)
        ]
        configuration = _ppo_configuration(model, 3e-4, 0.2, 0)

        assert policy_layers == [256, 256]
        assert value_layers == [256, 256]
        assert model.lr_schedule(1.0) == 3e-4
        assert model.clip_range(1.0) == 0.2
        assert configuration["n_steps"] == 2_048
        assert configuration["batch_size"] == 64
        assert configuration["n_epochs"] == 10
        assert configuration["gamma"] == 0.99
        assert configuration["gae_lambda"] == 0.95
        assert configuration["rollout_buffer_class"] == "RolloutBuffer"
    finally:
        env.close()


def test_checkpoint_frequency_uses_total_environment_steps() -> None:
    """Callback calls are converted from total vectorized environment steps."""
    assert _checkpoint_settings(10_000, 4) == {
        "requested_interval_environment_steps": 10_000,
        "callback_frequency": 2_500,
        "effective_interval_environment_steps": 10_000,
    }
    assert _checkpoint_settings(10_000, 3)[
        "effective_interval_environment_steps"
    ] == 10_002
    assert _checkpoint_settings(1_000_000, 4)[
        "effective_interval_environment_steps"
    ] == 100_000


def test_output_directory_cannot_be_reused(tmp_path) -> None:
    """A completed or partial run cannot be silently overwritten."""
    output_dir = tmp_path / "run"
    paths = _prepare_output_dir(output_dir)
    assert paths["checkpoints"].is_dir()
    assert paths["monitor"].is_dir()
    assert paths["tensorboard"].is_dir()
    with pytest.raises(FileExistsError):
        _prepare_output_dir(output_dir)


def test_damage_metadata_describes_both_conditions() -> None:
    """Recorded damage settings distinguish nominal and robust training."""
    nominal = _damage_configuration("nominal")
    robust = _damage_configuration("robust")
    assert nominal["healthy_probability"] == 1.0
    assert nominal["alpha_range"] == [1.0, 1.0]
    assert robust["healthy_probability"] == 0.25
    assert robust["leg_selection"] == "uniform"
    assert robust["alpha_range"] == [0.25, 1.0]
