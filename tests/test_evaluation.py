"""Tests for controlled policy evaluation."""

import argparse
import csv
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import damage_robust_ant.evaluate as evaluation
from damage_robust_ant.evaluate import (
    CSV_COLUMNS,
    _evaluate_episodes,
    _make_evaluation_env,
    parse_args,
    run_evaluation,
)


class FakeModel:
    """Small deterministic policy stand-in with inspectable state."""

    def __init__(self, action: np.ndarray | None = None) -> None:
        self.policy = torch.nn.Linear(1, 1)
        self.num_timesteps = 17
        self.action = (
            np.linspace(0.1, 0.8, 8, dtype=np.float32)
            if action is None
            else action
        )
        self.deterministic_arguments = []

    def predict(self, observation, deterministic=False):
        self.deterministic_arguments.append(deterministic)
        return self.action.copy(), None

    def learn(self, *args, **kwargs):
        raise AssertionError("evaluation must not call learn")


class TrackingEnv:
    """Two-step environment that records seeds, damage, and closure."""

    def __init__(
        self,
        damage_leg: str,
        alpha: float,
        *,
        truncated: bool = False,
    ) -> None:
        self.requested_leg = damage_leg
        self.damage_leg = None if damage_leg == "healthy" else damage_leg
        self.damage_alpha = alpha
        self.unwrapped = SimpleNamespace(dt=0.5)
        self.policy_action = None
        self.applied_action = None
        self.truncated = truncated
        self.reset_seeds = []
        self.closed = False

    def _info(self, x_position: float) -> dict[str, object]:
        return {
            "x_position": x_position,
            "damage_leg": self.damage_leg,
            "damage_alpha": self.damage_alpha,
        }

    def reset(self, *, seed: int):
        self.reset_seeds.append(seed)
        self.step_count = 0
        self.initial_x = seed / 1_000
        return np.array([0.0]), self._info(self.initial_x)

    def step(self, action: np.ndarray):
        self.step_count += 1
        self.policy_action = np.asarray(action).copy()
        self.applied_action = self.policy_action.copy()
        if self.damage_leg is not None:
            indices = list(evaluation.LEG_ACTION_INDICES[self.damage_leg])
            self.applied_action[indices] *= self.damage_alpha

        finished = self.step_count == 2
        terminated = finished and not self.truncated
        truncated = finished and self.truncated
        x_position = self.initial_x + (1.0 if self.step_count == 1 else 3.0)
        reward = 1.5 if self.step_count == 1 else 2.5
        return (
            np.array([float(self.step_count)]),
            reward,
            terminated,
            truncated,
            self._info(x_position),
        )

    def close(self) -> None:
        self.closed = True


def make_args(tmp_path, **overrides) -> argparse.Namespace:
    values = {
        "model": tmp_path / "model.zip",
        "training_condition": "nominal",
        "training_seed": 0,
        "damage_leg": "front_left",
        "alpha": 0.5,
        "evaluation_seed": 100,
        "episodes": 2,
        "output_csv": tmp_path / "evaluation.csv",
        "append": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_evaluation_argument_defaults_and_validation(tmp_path) -> None:
    """The CLI exposes controlled conditions and safe defaults."""
    model_path = tmp_path / "model.zip"
    model_path.touch()
    args = parse_args(
        [
            "--model",
            str(model_path),
            "--training-condition",
            "robust",
            "--training-seed",
            "2",
            "--damage-leg",
            "healthy",
            "--alpha",
            "1",
            "--output-csv",
            str(tmp_path / "results.csv"),
        ]
    )
    assert args.evaluation_seed == 0
    assert args.episodes == 10
    assert args.append is False

    invalid_arguments = [
        ["--damage-leg", "healthy", "--alpha", "0.5"],
        ["--damage-leg", "front_left", "--alpha", "1.1"],
        ["--damage-leg", "front_left", "--alpha", "nan"],
        ["--damage-leg", "front_left", "--alpha", "0.5", "--episodes", "0"],
    ]
    common = [
        "--model",
        str(model_path),
        "--training-condition",
        "nominal",
        "--training-seed",
        "0",
        "--output-csv",
        str(tmp_path / "results.csv"),
    ]
    for condition_arguments in invalid_arguments:
        with pytest.raises(SystemExit):
            parse_args([*common, *condition_arguments])


def test_episode_metrics_seeds_commands_and_policy_are_controlled(tmp_path) -> None:
    """Evaluation produces the required metrics without modifying the policy."""
    args = make_args(tmp_path)
    model = FakeModel()
    env = TrackingEnv(args.damage_leg, args.alpha)
    original_state = {
        name: value.clone() for name, value in model.policy.state_dict().items()
    }

    rows = _evaluate_episodes(model, env, args)

    assert env.reset_seeds == [100, 101]
    assert model.deterministic_arguments == [True, True, True, True]
    assert model.num_timesteps == 17
    for name, value in model.policy.state_dict().items():
        assert torch.equal(value, original_state[name])

    for episode, row in enumerate(rows):
        assert list(row) == CSV_COLUMNS
        assert row["evaluation_seed"] == 100 + episode
        assert row["episode_return"] == 4.0
        assert row["episode_length"] == 2
        assert row["terminated_before_time_limit"] is True
        assert row["forward_distance"] == pytest.approx(3.0)
        assert row["mean_forward_speed"] == pytest.approx(3.0)
        for index, raw_value in enumerate(model.action):
            raw_key = f"mean_abs_raw_command_actuator_{index}"
            applied_key = f"mean_abs_applied_command_actuator_{index}"
            assert row[raw_key] == pytest.approx(raw_value)
            expected_applied = raw_value * (0.5 if index in (2, 3) else 1.0)
            assert row[applied_key] == pytest.approx(expected_applied)


def test_time_limit_is_not_recorded_as_early_termination(tmp_path) -> None:
    """A truncation at the time limit is distinct from unhealthy termination."""
    args = make_args(tmp_path, episodes=1)
    row = _evaluate_episodes(
        FakeModel(),
        TrackingEnv(args.damage_leg, args.alpha, truncated=True),
        args,
    )[0]
    assert row["terminated_before_time_limit"] is False


def test_run_writes_exact_schema_appends_and_closes(monkeypatch, tmp_path) -> None:
    """Successful runs write complete rows and always close their environments."""
    first_args = make_args(tmp_path, episodes=1)
    second_args = make_args(
        tmp_path,
        training_condition="robust",
        damage_leg="back_right",
        alpha=0.25,
        episodes=1,
        append=True,
    )
    environments = []

    def make_env(damage_leg, alpha):
        env = TrackingEnv(damage_leg, alpha)
        environments.append(env)
        return env

    load_model = Mock(side_effect=(FakeModel(), FakeModel()))
    monkeypatch.setattr(evaluation, "PPO", SimpleNamespace(load=load_model))
    monkeypatch.setattr(evaluation, "_make_evaluation_env", make_env)

    run_evaluation(first_args)
    run_evaluation(second_args)

    assert all(env.closed for env in environments)
    with first_args.output_csv.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    assert reader.fieldnames == CSV_COLUMNS
    assert len(rows) == 2
    assert {row["policy_training_condition"] for row in rows} == {
        "nominal",
        "robust",
    }


def test_existing_output_is_not_overwritten(monkeypatch, tmp_path) -> None:
    """An existing CSV requires explicit append permission."""
    args = make_args(tmp_path)
    args.output_csv.write_text("existing data\n")
    load_model = Mock()
    monkeypatch.setattr(evaluation, "PPO", SimpleNamespace(load=load_model))

    with pytest.raises(FileExistsError):
        run_evaluation(args)

    assert args.output_csv.read_text() == "existing data\n"
    load_model.assert_not_called()


def test_environment_closes_when_evaluation_fails(monkeypatch, tmp_path) -> None:
    """A failed numerical check does not leak the MuJoCo environment."""
    args = make_args(tmp_path, episodes=1)
    model = FakeModel(np.full(8, np.nan, dtype=np.float32))
    env = TrackingEnv(args.damage_leg, args.alpha)
    monkeypatch.setattr(evaluation, "PPO", SimpleNamespace(load=Mock(return_value=model)))
    monkeypatch.setattr(evaluation, "_make_evaluation_env", Mock(return_value=env))

    with pytest.raises(RuntimeError, match="non-finite policy action"):
        run_evaluation(args)

    assert env.closed
    assert not args.output_csv.exists()


def test_real_ant_episode_produces_finite_results(tmp_path) -> None:
    """The metric loop works with Ant-v5's real information and time limit."""
    args = make_args(tmp_path, episodes=1)
    model = FakeModel(np.zeros(8, dtype=np.float32))
    env = _make_evaluation_env(args.damage_leg, args.alpha)
    try:
        row = _evaluate_episodes(model, env, args)[0]
    finally:
        env.close()

    assert 1 <= row["episode_length"] <= 1_000
    assert all(
        np.isfinite(value)
        for name, value in row.items()
        if name not in {
            "policy_training_condition",
            "damage_leg",
            "terminated_before_time_limit",
        }
    )
