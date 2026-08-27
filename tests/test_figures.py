"""Tests for final figure data preparation."""

import numpy as np
import pandas as pd
import pytest

from damage_robust_ant.damage import LEG_ACTION_INDICES
from damage_robust_ant.evaluate import CSV_COLUMNS
from damage_robust_ant.figures import (
    FINAL_EVALUATION_SEEDS,
    FINAL_TRAINING_SEEDS,
    LEG_ORDER,
    _command_magnitudes_by_seed,
    _evaluation_by_seed,
    _evaluation_summary,
    _holdout_comparison,
    _load_evaluation,
    _validate_matrix,
    parse_args,
)


def _evaluation_row(
    condition: str,
    training_seed: int,
    evaluation_seed: int,
    damage_leg: str,
    alpha: float,
) -> dict[str, object]:
    forward_distance = float(training_seed + alpha)
    lateral_distance = 0.5
    horizontal_distance = float(np.hypot(forward_distance, lateral_distance))
    episode_length = 1000
    raw = np.linspace(0.1, 0.8, 8)
    applied = raw.copy()
    if damage_leg != "healthy":
        applied[list(LEG_ACTION_INDICES[damage_leg])] *= alpha
    row: dict[str, object] = {
        "policy_training_condition": condition,
        "policy_training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "damage_leg": damage_leg,
        "damage_alpha": alpha,
        "episode_return": 100.0 + training_seed,
        "episode_length": episode_length,
        "terminated_before_time_limit": False,
        "forward_distance": forward_distance,
        "lateral_distance": lateral_distance,
        "horizontal_distance": horizontal_distance,
        "mean_forward_speed": forward_distance / (episode_length * 0.05),
        "mean_horizontal_speed": horizontal_distance / (episode_length * 0.05),
    }
    row.update(
        {
            f"mean_abs_raw_command_actuator_{index}": value
            for index, value in enumerate(raw)
        }
    )
    row.update(
        {
            f"mean_abs_applied_command_actuator_{index}": value
            for index, value in enumerate(applied)
        }
    )
    return row


def _complete_evaluation() -> pd.DataFrame:
    cases = [("healthy", 1.0)] + [
        (leg, alpha) for leg in LEG_ORDER for alpha in (0.5, 0.0)
    ]
    rows = [
        _evaluation_row(condition, training_seed, evaluation_seed, leg, alpha)
        for condition in ("nominal", "robust")
        for training_seed in FINAL_TRAINING_SEEDS
        for leg, alpha in cases
        for evaluation_seed in FINAL_EVALUATION_SEEDS
    ]
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def test_figure_cli_defaults_and_dpi_validation() -> None:
    args = parse_args([])
    assert args.final_results.name == "final_episode_results.csv"
    assert args.figures_dir.name == "figures"
    assert args.processed_dir.name == "processed"
    assert args.dpi == 220

    with pytest.raises(SystemExit):
        parse_args(["--dpi", "60"])


def test_final_matrix_validation_and_seed_level_aggregation(tmp_path) -> None:
    frame = _complete_evaluation()
    path = tmp_path / "evaluation.csv"
    frame.to_csv(path, index=False)

    loaded = _load_evaluation(path)
    _validate_matrix(
        loaded,
        conditions=("nominal", "robust"),
        training_seeds=FINAL_TRAINING_SEEDS,
        evaluation_seeds=FINAL_EVALUATION_SEEDS,
    )
    by_seed = _evaluation_by_seed(loaded)

    assert len(by_seed) == 18
    assert set(by_seed["remaining_strength"]) == {0.0, 0.5, 1.0}
    assert set(by_seed["episodes_averaged"]) == {10, 40}
    nominal_seed_five = by_seed[
        (by_seed["policy_training_condition"] == "nominal")
        & (by_seed["policy_training_seed"] == 5)
    ]
    assert nominal_seed_five.loc[
        nominal_seed_five["remaining_strength"] == 0.5, "forward_distance"
    ].item() == pytest.approx(5.5)
    summary = _evaluation_summary(by_seed)
    assert len(summary) == 6
    assert set(summary["training_seed_count"]) == {3}
    assert "mean_forward_speed_mean" in summary
    assert "mean_forward_speed_std" in summary


def test_command_summary_uses_leg_mapping_and_damage_scaling() -> None:
    commands = _command_magnitudes_by_seed(_complete_evaluation())
    assert len(commands) == 48
    selected = commands[
        (commands["policy_training_condition"] == "robust")
        & (commands["policy_training_seed"] == 6)
        & (commands["command_leg"] == "front_left")
    ].set_index("command_stage")["mean_abs_command_proxy"]
    assert selected["raw"] == pytest.approx((0.3 + 0.4) / 2)
    assert selected["applied"] == pytest.approx(selected["raw"] * 0.5)

    back_left = commands[
        (commands["policy_training_condition"] == "robust")
        & (commands["policy_training_seed"] == 6)
        & (commands["command_leg"] == "back_left")
    ].set_index("command_stage")["mean_abs_command_proxy"]
    assert back_left["applied"] == pytest.approx(back_left["raw"])


def test_holdout_comparison_uses_paired_episode_differences() -> None:
    preserved = pd.DataFrame(
        [
            _evaluation_row("robust", 6, seed, "healthy", 1.0)
            for seed in (500, 501, 502)
        ],
        columns=CSV_COLUMNS,
    )
    selected = preserved.copy()
    selected["horizontal_distance"] += [1.0, 2.0, 3.0]

    comparison = _holdout_comparison(preserved, selected)

    assert comparison["episodes"].item() == 3
    assert comparison["mean_paired_horizontal_difference"].item() == pytest.approx(
        2.0
    )
    assert comparison["sd_paired_horizontal_difference"].item() == pytest.approx(
        1.0
    )


def test_evaluation_loader_rejects_incorrect_applied_commands(tmp_path) -> None:
    frame = _complete_evaluation()
    damaged_index = frame.index[frame["damage_leg"] == "front_left"][0]
    frame.loc[damaged_index, "mean_abs_applied_command_actuator_2"] = 99.0
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="applied commands"):
        _load_evaluation(path)
