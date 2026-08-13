"""Tests for the PPO training command."""

import argparse
import subprocess
from pathlib import Path

import pytest
import torch

from damage_robust_ant.train import (
    TrainingProgressCallback,
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
    assert args.run_index == 1
    assert args.total_runs == 1


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


def test_training_rejects_run_index_after_total(tmp_path) -> None:
    """The human-readable run counter must describe a valid position."""
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--condition",
                "nominal",
                "--output-dir",
                str(tmp_path / "run"),
                "--run-index",
                "2",
                "--total-runs",
                "1",
            ]
        )


def test_training_progress_callback_reports_remaining_work(
    monkeypatch,
    capsys,
) -> None:
    """The five-minute update reports steps, runs and a rough ETA."""
    times = iter((100.0, 399.9, 400.0))
    monkeypatch.setattr("damage_robust_ant.train.time.perf_counter", lambda: next(times))
    callback = TrainingProgressCallback(
        requested_steps=1_000_000,
        run_index=2,
        total_runs=6,
    )
    callback._on_training_start()
    assert "about every five minutes" in capsys.readouterr().out
    callback.num_timesteps = 250_000

    assert callback._on_step()
    assert capsys.readouterr().out == ""
    assert callback._on_step()
    output = capsys.readouterr().out
    assert "Training 2/6" in output
    assert "250,000/1,000,000 steps (25.0%)" in output
    assert "4 full runs remain afterward" in output
    assert "rough training ETA 1h 35m" in output


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


def test_main_experiment_launcher_has_fixed_command_matrix(tmp_path) -> None:
    """The launcher plans every frozen training and evaluation command once."""
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["bash", str(repository / "run_main_experiment.sh"), "--dry-run"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.splitlines()
    training_lines = [line for line in lines if line.startswith("DRY RUN TRAINING:")]
    evaluation_lines = [
        line for line in lines if line.startswith("DRY RUN EVALUATION:")
    ]

    assert "Live training summaries are printed every 5 minutes." in lines
    assert len(training_lines) == 6
    assert all("--timesteps 1000000" in line for line in training_lines)
    assert all("--num-envs 4" in line for line in training_lines)
    assert all("--learning-rate 0.0003" in line for line in training_lines)
    assert all("--clip-range 0.2" in line for line in training_lines)
    assert sum("--run-index" in line for line in training_lines) == 6
    assert all("--total-runs 6" in line for line in training_lines)
    for condition in ("nominal", "robust"):
        for seed in range(3):
            expected = f"--condition {condition} --seed {seed} "
            assert sum(expected in line for line in training_lines) == 1

    assert len(evaluation_lines) == 54
    assert "--append" not in evaluation_lines[0]
    assert sum("--append" in line for line in evaluation_lines) == 53
    assert all("--evaluation-seed 100" in line for line in evaluation_lines)
    assert all("--episodes 10" in line for line in evaluation_lines)
    for condition in ("nominal", "robust"):
        for seed in range(3):
            expected = (
                f"--training-condition {condition} --training-seed {seed} "
            )
            assert sum(expected in line for line in evaluation_lines) == 9
    assert sum(
        "--damage-leg healthy --alpha 1.0" in line for line in evaluation_lines
    ) == 6
    for alpha in ("0.5", "0.0"):
        for leg in ("front_left", "front_right", "back_left", "back_right"):
            expected = f"--damage-leg {leg} --alpha {alpha}"
            assert sum(expected in line for line in evaluation_lines) == 6
