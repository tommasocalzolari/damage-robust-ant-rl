"""Tests for the staged overnight locomotion command."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import damage_robust_ant.gait_train as gait
from damage_robust_ant.damage import LEG_ACTION_INDICES
from damage_robust_ant.evaluate import CSV_COLUMNS
from damage_robust_ant.gait_train import (
    POLICIES,
    LongTrainingProgressCallback,
    _evaluate_checkpoint,
    _rollout_target,
    _selection_decision,
    _summarize_validation,
    parse_args,
)


def _validation_frame(condition: str) -> pd.DataFrame:
    seed = POLICIES[condition]["seed"]
    rows = []
    for leg, alpha in [("healthy", 1.0), *[(leg, 0.5) for leg in LEG_ACTION_INDICES]]:
        for episode_index, evaluation_seed in enumerate(range(200, 210)):
            positive = episode_index != 0
            rows.append(
                {
                    "policy_training_condition": condition,
                    "policy_training_seed": seed,
                    "evaluation_seed": evaluation_seed,
                    "damage_leg": leg,
                    "damage_alpha": alpha,
                    "episode_return": 10.0,
                    "episode_length": 1_000,
                    "terminated_before_time_limit": episode_index == 0,
                    "forward_distance": 1.0 if positive else -0.1,
                    "mean_forward_speed": 0.5,
                    **{
                        f"mean_abs_raw_command_actuator_{index}": 0.25
                        for index in range(8)
                    },
                    **{
                        f"mean_abs_applied_command_actuator_{index}": 0.25
                        for index in range(8)
                    },
                }
            )
    frame = pd.DataFrame(rows)
    assert list(frame.columns) == CSV_COLUMNS
    return frame


def _checkpoint_record(
    requested_steps: int,
    *,
    passes: bool,
    distance: float,
    early_rate: float = 0.1,
    speed: float = 0.5,
    healthy_distance: float = 1.0,
) -> dict[str, object]:
    return {
        "requested_checkpoint_steps": requested_steps,
        "actual_checkpoint_steps": requested_steps,
        "target_condition": "moderate_damage",
        "target_criteria_met": passes,
        "moderate_damage_mean_forward_distance": distance,
        "moderate_damage_early_termination_rate": early_rate,
        "moderate_damage_mean_forward_speed": speed,
        "healthy_mean_forward_distance": healthy_distance,
    }


def test_gait_arguments_and_frozen_policy_choices(tmp_path) -> None:
    """The small CLI exposes only the frozen condition and output choices."""
    args = parse_args(
        [
            "--condition",
            "robust",
            "--output-dir",
            str(tmp_path / "run"),
            "--policy-index",
            "2",
            "--total-policies",
            "2",
        ]
    )
    assert args.condition == "robust"
    assert POLICIES["nominal"]["seed"] == 1
    assert POLICIES["nominal"]["target"] == "healthy"
    assert POLICIES["robust"]["seed"] == 2
    assert POLICIES["robust"]["target"] == "moderate_damage"

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--condition",
                "nominal",
                "--output-dir",
                str(tmp_path / "other"),
                "--policy-index",
                "3",
                "--total-policies",
                "2",
            ]
        )


def test_rollout_targets_respect_boundaries_and_hard_cap() -> None:
    """Requested stages finish rollouts without crossing five million steps."""
    rollout_size = 8_192
    assert _rollout_target(250_000, rollout_size) == 253_952
    assert _rollout_target(1_000_000, rollout_size) == 1_007_616
    assert _rollout_target(3_000_000, rollout_size) == 3_006_464
    assert _rollout_target(5_000_000, rollout_size) == 4_997_120
    targets = [
        _rollout_target(requested, rollout_size)
        for requested in range(250_000, 5_000_001, 250_000)
    ]
    assert targets == sorted(set(targets))
    assert targets[-1] <= 5_000_000


def test_checkpoint_evaluation_uses_fixed_conditions_and_seeds(
    monkeypatch,
    tmp_path,
) -> None:
    """Validation runs healthy plus all four moderately damaged legs."""
    calls = []

    def capture(command, check):
        calls.append((command, check))

    monkeypatch.setattr("damage_robust_ant.gait_train.subprocess.run", capture)
    checkpoint = tmp_path / "model.zip"
    checkpoint.touch()
    output = _evaluate_checkpoint(checkpoint, "robust", 2, 1_007_616, tmp_path)

    assert output == tmp_path / "checkpoint_1007616_episodes.csv"
    assert len(calls) == 5
    assert all(check is True for _, check in calls)
    assert sum("--append" in command for command, _ in calls) == 4
    expected = [("healthy", "1.0"), *[(leg, "0.5") for leg in LEG_ACTION_INDICES]]
    for (command, _), (leg, alpha) in zip(calls, expected):
        assert command[1:3] == ["-m", "damage_robust_ant.evaluate"]
        assert command[command.index("--damage-leg") + 1] == leg
        assert command[command.index("--alpha") + 1] == alpha
        assert command[command.index("--evaluation-seed") + 1] == "200"
        assert command[command.index("--episodes") + 1] == "10"


def test_validation_summary_accepts_boundary_criteria_and_records_legs(
    tmp_path,
) -> None:
    """The robust pooled thresholds are inclusive and per-leg metrics remain visible."""
    frame = _validation_frame("robust")
    path = tmp_path / "validation.csv"
    frame.to_csv(path, index=False)

    record = _summarize_validation(path, "robust", 1_000_000, 1_007_616)

    assert record["moderate_damage_early_termination_rate"] == pytest.approx(0.1)
    assert record["moderate_damage_mean_forward_speed"] == pytest.approx(0.5)
    assert record["moderate_damage_positive_distance_rate"] == pytest.approx(0.9)
    assert record["target_criteria_met"] is True
    for leg in LEG_ACTION_INDICES:
        assert f"moderate_damage_{leg}_mean_forward_distance" in record


@pytest.mark.parametrize("failure", ["early", "speed", "positive"])
def test_validation_summary_rejects_metrics_outside_threshold(
    tmp_path,
    failure,
) -> None:
    """A candidate fails when any frozen locomotion threshold is missed."""
    frame = _validation_frame("robust")
    moderate = frame["damage_leg"] != "healthy"
    if failure == "early":
        affected = frame[moderate].index[:5]
        frame.loc[moderate, "terminated_before_time_limit"] = False
        frame.loc[affected, "terminated_before_time_limit"] = True
    elif failure == "speed":
        frame.loc[moderate, "mean_forward_speed"] = 0.499
    else:
        affected = frame[moderate].index[:5]
        frame.loc[affected, "forward_distance"] = -0.1
    path = tmp_path / f"{failure}.csv"
    frame.to_csv(path, index=False)

    record = _summarize_validation(path, "robust", 1_000_000, 1_007_616)

    assert record["target_criteria_met"] is False


def test_validation_summary_rejects_wrong_matrix(tmp_path) -> None:
    """Missing, duplicate, unexpected or wrongly seeded cases cannot be selected."""
    frame = _validation_frame("nominal")
    frame.loc[frame.index[10:20], "damage_alpha"] = 0.4
    path = tmp_path / "wrong.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        _summarize_validation(path, "nominal", 1_000_000, 1_007_616)


@pytest.mark.parametrize(
    "problem",
    ["wrong_seed", "duplicate", "nan", "invalid_boolean", "invalid_length"],
)
def test_validation_summary_rejects_invalid_rows(tmp_path, problem) -> None:
    """Invalid rows cannot influence checkpoint selection."""
    frame = _validation_frame("robust")
    if problem == "wrong_seed":
        frame.loc[0, "evaluation_seed"] = 999
    elif problem == "duplicate":
        frame.loc[1] = frame.loc[0]
    elif problem == "nan":
        frame.loc[0, "episode_return"] = float("nan")
    elif problem == "invalid_boolean":
        frame["terminated_before_time_limit"] = -1
    else:
        frame.loc[0, "episode_length"] = 0
    path = tmp_path / f"{problem}.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="incomplete or invalid|incomplete seed set"):
        _summarize_validation(path, "robust", 1_000_000, 1_007_616)


def test_validation_summary_rejects_nonfinite_auxiliary_metric(tmp_path) -> None:
    """Return and command metrics receive the same finite-value check."""
    frame = _validation_frame("robust")
    frame.loc[0, "mean_abs_raw_command_actuator_7"] = float("inf")
    path = tmp_path / "nonfinite.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        _summarize_validation(path, "robust", 1_000_000, 1_007_616)


def test_checkpoint_selection_waits_for_complete_training_stages() -> None:
    """Passing checkpoints are considered only after complete 3M/4M/5M stages."""
    first = _checkpoint_record(1_000_000, passes=True, distance=2.0)
    better = _checkpoint_record(
        2_500_000,
        passes=True,
        distance=3.0,
        early_rate=0.05,
    )
    records = [first, better]

    assert _selection_decision(records, 2_500_000) == (None, False)
    assert _selection_decision(records, 3_000_000) == (better, True)
    assert _selection_decision(records, 3_500_000) == (None, False)

    failures = [
        _checkpoint_record(3_000_000, passes=False, distance=1.0),
        _checkpoint_record(4_000_000, passes=False, distance=4.0),
        _checkpoint_record(5_000_000, passes=False, distance=2.0),
    ]
    assert _selection_decision(failures[:2], 4_000_000) == (None, False)
    assert _selection_decision(failures, 5_000_000) == (failures[1], False)


def test_checkpoint_selection_uses_frozen_tie_breakers() -> None:
    """Checkpoint ranking follows the documented fixed comparison order."""
    base = _checkpoint_record(1_000_000, passes=True, distance=3.0)
    later = _checkpoint_record(1_500_000, passes=True, distance=3.0)
    lower_early = _checkpoint_record(
        2_000_000,
        passes=True,
        distance=3.0,
        early_rate=0.05,
    )
    faster = _checkpoint_record(
        2_500_000,
        passes=True,
        distance=3.0,
        early_rate=0.05,
        speed=0.6,
    )
    better_healthy = _checkpoint_record(
        3_000_000,
        passes=True,
        distance=3.0,
        early_rate=0.05,
        speed=0.6,
        healthy_distance=2.0,
    )
    records = [base, later, lower_early, faster, better_healthy]
    assert _selection_decision(records, 3_000_000) == (better_healthy, True)

    identical = [base, later]
    assert _selection_decision(identical, 3_000_000) == (base, True)


def test_progress_timer_is_not_reset_between_training_chunks(
    monkeypatch,
    capsys,
) -> None:
    """Short chunks do not postpone the five-minute update indefinitely."""
    times = iter((100.0, 400.0))
    monkeypatch.setattr(
        "damage_robust_ant.gait_train.time.perf_counter",
        lambda: next(times),
    )
    callback = LongTrainingProgressCallback("nominal", 1, 1, 2, 100.0)
    callback._on_training_start()
    callback._on_training_start()
    callback.num_timesteps = 1_000_000

    assert callback._on_step()
    output = capsys.readouterr().out
    assert "Policy 1/2 (nominal seed 1)" in output
    assert "1,000,000/5,000,000 maximum steps" in output


def test_long_training_keeps_learning_until_first_decision(
    monkeypatch,
    tmp_path,
) -> None:
    """An early passing model is saved only after the complete initial stage."""
    reset_arguments = []
    validation_steps = []
    saved_steps = {}

    class FakeLogger:
        def dump(self, step):
            pass

        def close(self):
            pass

    class FakeEnv:
        num_envs = 1

        def reset(self):
            return np.array([[0.0]])

        def close(self):
            pass

    class FakeModel:
        n_steps = 1
        num_timesteps = 0
        logger = FakeLogger()
        _logger = logger

        def learn(self, total_timesteps, callback, **kwargs):
            reset_arguments.append(kwargs["reset_num_timesteps"])
            self.num_timesteps += total_timesteps

        def save(self, path):
            Path(path).write_bytes(b"model")
            saved_steps[str(path)] = self.num_timesteps

    fake_model = FakeModel()
    fake_env = FakeEnv()
    monkeypatch.setattr(gait, "CHECKPOINT_INTERVAL_STEPS", 1)
    monkeypatch.setattr(gait, "VALIDATION_INTERVAL_STEPS", 1)
    monkeypatch.setattr(gait, "FIRST_VALIDATION_STEPS", 1)
    monkeypatch.setattr(gait, "DECISION_STEPS", {3})
    monkeypatch.setattr(gait, "MAXIMUM_TRAINING_STEPS", 3)
    selection_results = tmp_path / "main.csv"
    selection_results.write_bytes(b"main results")
    monkeypatch.setattr(gait, "MAIN_RESULTS_RELATIVE_PATH", selection_results)
    monkeypatch.setattr(gait, "MAIN_RESULTS_SHA256", gait._sha256(selection_results))
    monkeypatch.setattr(gait, "_git_provenance", lambda: ("0" * 40, True))
    monkeypatch.setattr(gait, "_package_versions", lambda: {})
    monkeypatch.setattr(gait, "_damage_configuration", lambda condition: {})
    monkeypatch.setattr(
        gait,
        "_ppo_configuration",
        lambda *args: {"reset_num_timesteps": True},
    )
    monkeypatch.setattr(gait, "_assert_finite_model", lambda model: None)
    monkeypatch.setattr(gait, "make_training_env", lambda *args: fake_env)
    monkeypatch.setattr(gait, "make_ppo", lambda *args: fake_model)

    def fake_evaluate(checkpoint, condition, seed, actual_steps, validation_dir):
        validation_steps.append(actual_steps)
        return validation_dir / f"{actual_steps}.csv"

    def fake_summary(path, condition, requested_steps, actual_steps):
        return {
            "condition": condition,
            "seed": 1,
            "requested_checkpoint_steps": requested_steps,
            "actual_checkpoint_steps": actual_steps,
            "target_condition": "healthy",
            "target_criteria_met": True,
            "healthy_mean_forward_distance": 1.0,
            "healthy_early_termination_rate": 0.0,
            "healthy_mean_forward_speed": 0.5,
            "healthy_positive_distance_rate": 1.0,
            "moderate_damage_mean_forward_distance": 0.5,
        }

    monkeypatch.setattr(gait, "_evaluate_checkpoint", fake_evaluate)
    monkeypatch.setattr(gait, "_summarize_validation", fake_summary)
    reloaded = SimpleNamespace(
        num_timesteps=1,
        predict=lambda observation, deterministic: (np.zeros((1, 8)), None),
    )
    monkeypatch.setattr(
        gait,
        "PPO",
        SimpleNamespace(load=lambda path, env, device: reloaded),
    )

    output = tmp_path / "run"
    result = gait.run_long_training(
        SimpleNamespace(
            condition="nominal",
            output_dir=output,
            policy_index=1,
            total_policies=2,
        )
    )

    assert result == output / "selected_model.zip"
    assert reset_arguments == [True, False, False]
    assert validation_steps == [1, 2, 3]
    assert len(saved_steps) == 3
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["total_search_environment_steps"] == 3
    assert metadata["selected_model_environment_steps"] == 1
    assert metadata["criteria_met"] is True


def test_overnight_launcher_has_two_frozen_fresh_runs(tmp_path) -> None:
    """The human-facing launcher plans exactly the two selected policies."""
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["bash", str(repository / "run_overnight_gait_training.sh"), "--dry-run"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.splitlines()
    commands = [line for line in lines if line.startswith("DRY RUN TRAINING:")]

    assert len(commands) == 2
    assert "Both policies start as new networks" in completed.stdout
    assert "condition=nominal selected_seed=1" in completed.stdout
    assert "condition=robust selected_seed=2" in completed.stdout
    assert "--condition nominal" in commands[0]
    assert "--output-dir artifacts/overnight/nominal_seed_1" in commands[0]
    assert "--condition robust" in commands[1]
    assert "--output-dir artifacts/overnight/robust_seed_2" in commands[1]
    assert not (tmp_path / "artifacts").exists()
