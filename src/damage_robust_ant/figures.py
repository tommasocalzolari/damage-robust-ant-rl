"""Generate the final report figures from completed experiment outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from damage_robust_ant.damage import LEG_ACTION_INDICES
from damage_robust_ant.evaluate import CSV_COLUMNS


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FINAL_TRAINING_SEEDS = (5, 6, 7)
FINAL_EVALUATION_SEEDS = tuple(range(300, 310))
HOLDOUT_EVALUATION_SEEDS = tuple(range(500, 510))
LEG_ORDER = ("front_left", "front_right", "back_left", "back_right")
LEG_LABELS = {
    "front_left": "Front left",
    "front_right": "Front right",
    "back_left": "Back left",
    "back_right": "Back right",
}
LEG_SHORT_LABELS = {
    "front_left": "FL",
    "front_right": "FR",
    "back_left": "BL",
    "back_right": "BR",
}
CONDITION_LABELS = {"nominal": "Nominal", "robust": "Robust"}
CONDITION_COLORS = {"nominal": "#0072B2", "robust": "#D55E00"}
DESIGN_COLORS = {"legacy": "#777777", "stabilized": "#009E73"}
SEVERITY_LABELS = {
    1.0: "Healthy",
    0.5: "Moderate damage",
    0.0: "Complete failure",
}
EVALUATION_METRICS = (
    "episode_return",
    "forward_distance",
    "horizontal_distance",
    "mean_forward_speed",
    "mean_horizontal_speed",
    "episode_length",
    "terminated_before_time_limit",
)
TRAINING_TAGS = (
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/approx_kl",
    "train/clip_fraction",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-results",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "final_episode_results.csv",
    )
    parser.add_argument(
        "--sensitivity-results",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "sensitivity_summary.csv",
    )
    parser.add_argument(
        "--final-configuration",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "final_configuration.json",
    )
    parser.add_argument(
        "--selected-results",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results"
            / "final_selected_episode_results.csv"
        ),
    )
    parser.add_argument(
        "--preserved-results",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results"
            / "preserved_robust_seed_6_holdout_episode_results.csv"
        ),
    )
    parser.add_argument(
        "--final-artifacts",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "final",
    )
    parser.add_argument(
        "--legacy-artifacts",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "main",
        help="original unstable PPO runs used only for the development diagnostic",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPOSITORY_ROOT / "figures",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "processed",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--skip-development-diagnostic",
        action="store_true",
        help="skip the legacy-versus-stabilized PPO diagnostic",
    )
    args = parser.parse_args(argv)
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    return args


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.0,
            "savefig.bbox": "tight",
        }
    )


# Validate source data before any aggregation or plotting.
def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required file does not exist: {path}")


def _coerce_boolean(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"{name} contains values that are not booleans")
    return mapped.astype(bool)


def _load_evaluation(path: Path) -> pd.DataFrame:
    _require_file(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != CSV_COLUMNS:
        raise ValueError(f"unexpected evaluation schema: {path}")
    if frame.empty or frame.isna().any().any():
        raise ValueError(f"evaluation is empty or contains missing values: {path}")

    frame = frame.copy()
    frame["terminated_before_time_limit"] = _coerce_boolean(
        frame["terminated_before_time_limit"],
        "terminated_before_time_limit",
    )
    nonnumeric = {
        "policy_training_condition",
        "damage_leg",
        "terminated_before_time_limit",
    }
    numeric_columns = [column for column in frame if column not in nonnumeric]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"evaluation contains nonfinite numeric values: {path}")
    frame[numeric_columns] = numeric

    expected_horizontal = np.hypot(
        frame["forward_distance"], frame["lateral_distance"]
    )
    if not np.allclose(
        frame["horizontal_distance"], expected_horizontal, rtol=1e-9, atol=1e-9
    ):
        raise ValueError(f"horizontal distance is inconsistent in {path}")
    expected_forward_speed = frame["forward_distance"] / (
        frame["episode_length"] * 0.05
    )
    expected_horizontal_speed = frame["horizontal_distance"] / (
        frame["episode_length"] * 0.05
    )
    if not np.allclose(
        frame["mean_forward_speed"], expected_forward_speed, rtol=1e-9, atol=1e-9
    ) or not np.allclose(
        frame["mean_horizontal_speed"],
        expected_horizontal_speed,
        rtol=1e-9,
        atol=1e-9,
    ):
        raise ValueError(f"reported speed is inconsistent in {path}")
    _validate_command_scaling(frame, path)
    return frame


def _validate_command_scaling(frame: pd.DataFrame, path: Path) -> None:
    raw_columns = [f"mean_abs_raw_command_actuator_{index}" for index in range(8)]
    applied_columns = [
        f"mean_abs_applied_command_actuator_{index}" for index in range(8)
    ]
    raw = frame[raw_columns].to_numpy(dtype=float)
    applied = frame[applied_columns].to_numpy(dtype=float)
    expected = raw.copy()
    for row_index, (leg, alpha) in enumerate(
        zip(frame["damage_leg"], frame["damage_alpha"])
    ):
        if leg == "healthy":
            if not math.isclose(float(alpha), 1.0):
                raise ValueError(f"healthy row has alpha != 1 in {path}")
            continue
        if leg not in LEG_ACTION_INDICES:
            raise ValueError(f"unknown damaged leg {leg!r} in {path}")
        expected[row_index, list(LEG_ACTION_INDICES[leg])] *= float(alpha)
    if not np.allclose(applied, expected, rtol=1e-9, atol=1e-9):
        raise ValueError(f"applied commands are inconsistent with damage in {path}")


def _expected_cases() -> set[tuple[str, float]]:
    return {("healthy", 1.0)} | {
        (leg, alpha) for leg in LEG_ORDER for alpha in (0.5, 0.0)
    }


def _validate_matrix(
    frame: pd.DataFrame,
    *,
    conditions: Iterable[str],
    training_seeds: Iterable[int],
    evaluation_seeds: Iterable[int],
) -> None:
    expected_conditions = set(conditions)
    expected_training_seeds = set(training_seeds)
    expected_evaluation_seeds = set(evaluation_seeds)
    if set(frame["policy_training_condition"]) != expected_conditions:
        raise ValueError("evaluation has unexpected training conditions")
    if set(frame["policy_training_seed"]) != expected_training_seeds:
        raise ValueError("evaluation has unexpected training seeds")
    if set(zip(frame["damage_leg"], frame["damage_alpha"])) != _expected_cases():
        raise ValueError("evaluation does not contain the exact nine damage cases")

    keys = [
        "policy_training_condition",
        "policy_training_seed",
        "evaluation_seed",
        "damage_leg",
        "damage_alpha",
    ]
    if frame.duplicated(keys).any():
        raise ValueError("evaluation contains duplicate policy/case/episode rows")
    expected_groups = len(expected_conditions) * len(expected_training_seeds) * 9
    groups = frame.groupby(
        [
            "policy_training_condition",
            "policy_training_seed",
            "damage_leg",
            "damage_alpha",
        ]
    )
    if len(groups) != expected_groups:
        raise ValueError("evaluation matrix is incomplete")
    for _, group in groups:
        if set(group["evaluation_seed"]) != expected_evaluation_seeds:
            raise ValueError("evaluation groups do not share the expected episode seeds")


# Aggregate episodes within each trained policy before comparing seeds.
def _evaluation_by_seed(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["remaining_strength"] = work["damage_alpha"]
    work["severity"] = work["remaining_strength"].map(SEVERITY_LABELS)
    grouped = (
        work.groupby(
            [
                "policy_training_condition",
                "policy_training_seed",
                "remaining_strength",
                "severity",
            ],
            as_index=False,
            observed=True,
        )[list(EVALUATION_METRICS)]
        .mean()
        .rename(
            columns={
                "terminated_before_time_limit": "early_termination_rate",
            }
        )
    )
    grouped["episodes_averaged"] = grouped["remaining_strength"].map(
        {1.0: 10, 0.5: 40, 0.0: 40}
    )
    return grouped.sort_values(
        ["policy_training_condition", "policy_training_seed", "remaining_strength"]
    ).reset_index(drop=True)


def _leg_performance_by_seed(frame: pd.DataFrame) -> pd.DataFrame:
    damaged = frame[frame["damage_leg"] != "healthy"]
    grouped = (
        damaged.groupby(
            [
                "policy_training_condition",
                "policy_training_seed",
                "damage_alpha",
                "damage_leg",
            ],
            as_index=False,
        )[list(EVALUATION_METRICS)]
        .mean()
        .rename(
            columns={
                "terminated_before_time_limit": "early_termination_rate",
            }
        )
    )
    grouped["episodes_averaged"] = 10
    return grouped


def _evaluation_summary(by_seed: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        *EVALUATION_METRICS[:-1],
        "early_termination_rate",
    ]
    summary = by_seed.groupby(
        ["policy_training_condition", "remaining_strength", "severity"],
        as_index=False,
        observed=True,
    )[metric_columns].agg(["mean", "std"])
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary = summary.rename(
        columns={
            "policy_training_condition_": "policy_training_condition",
            "remaining_strength_": "remaining_strength",
            "severity_": "severity",
        }
    )
    summary["training_seed_count"] = 3
    return summary.sort_values(
        ["policy_training_condition", "remaining_strength"]
    ).reset_index(drop=True)


def _command_magnitudes_by_seed(frame: pd.DataFrame) -> pd.DataFrame:
    representative = frame[
        (frame["damage_leg"] == "front_left")
        & np.isclose(frame["damage_alpha"], 0.5)
    ]
    records: list[dict[str, object]] = []
    for _, row in representative.iterrows():
        for command_leg in LEG_ORDER:
            indices = LEG_ACTION_INDICES[command_leg]
            for stage in ("raw", "applied"):
                values = [
                    float(row[f"mean_abs_{stage}_command_actuator_{index}"])
                    for index in indices
                ]
                records.append(
                    {
                        "policy_training_condition": row[
                            "policy_training_condition"
                        ],
                        "policy_training_seed": int(row["policy_training_seed"]),
                        "evaluation_seed": int(row["evaluation_seed"]),
                        "command_leg": command_leg,
                        "command_stage": stage,
                        "mean_abs_command_proxy": float(np.mean(values)),
                    }
                )
    per_episode = pd.DataFrame.from_records(records)
    return (
        per_episode.groupby(
            [
                "policy_training_condition",
                "policy_training_seed",
                "command_leg",
                "command_stage",
            ],
            as_index=False,
        )["mean_abs_command_proxy"]
        .mean()
        .sort_values(
            [
                "policy_training_condition",
                "policy_training_seed",
                "command_leg",
                "command_stage",
            ]
        )
        .reset_index(drop=True)
    )


# TensorBoard curves remain separate from deterministic evaluation results.
def _event_scalars(run_dir: Path, tags: Iterable[str]) -> pd.DataFrame:
    event_files = sorted(run_dir.glob("tensorboard/**/events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"no TensorBoard event file below {run_dir}")
    values: dict[str, dict[int, float]] = {tag: {} for tag in tags}
    for event_file in event_files:
        accumulator = EventAccumulator(
            str(event_file), size_guidance={"scalars": 0}
        )
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        missing = set(tags) - available
        if missing:
            raise ValueError(f"{event_file} is missing scalar tags: {sorted(missing)}")
        for tag in tags:
            for event in accumulator.Scalars(tag):
                values[tag][int(event.step)] = float(event.value)
    records = [
        {"tag": tag, "environment_steps": step, "value": value}
        for tag, step_values in values.items()
        for step, value in sorted(step_values.items())
    ]
    frame = pd.DataFrame.from_records(records)
    if frame.empty or not np.isfinite(frame["value"]).all():
        raise ValueError(f"invalid TensorBoard scalar data below {run_dir}")
    return frame


def _final_training_scalars(final_artifacts: Path) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for condition in ("nominal", "robust"):
        for seed in FINAL_TRAINING_SEEDS:
            run_dir = final_artifacts / f"{condition}_seed_{seed}"
            frame = _event_scalars(run_dir, TRAINING_TAGS)
            frame.insert(0, "policy_training_seed", seed)
            frame.insert(0, "policy_training_condition", condition)
            records.append(frame)
    return pd.concat(records, ignore_index=True)


def _legacy_nominal_scalars(legacy_artifacts: Path) -> pd.DataFrame:
    tags = ("rollout/ep_len_mean", "train/approx_kl", "train/clip_fraction")
    records: list[pd.DataFrame] = []
    for seed in (0, 1, 2):
        run_dir = legacy_artifacts / f"nominal_seed_{seed}"
        frame = _event_scalars(run_dir, tags)
        frame.insert(0, "policy_training_seed", seed)
        frame.insert(0, "design", "legacy")
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(temporary, format=path.suffix.lstrip("."), dpi=dpi)
    plt.close(figure)
    temporary.replace(path)


# Plotting functions consume only validated, processed tables.
def _seed_statistics(
    frame: pd.DataFrame, value_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = frame.pivot(
        index="environment_steps", columns="policy_training_seed", values=value_column
    ).sort_index()
    pivot = pivot.dropna()
    if pivot.shape[1] != 3:
        raise ValueError("expected exactly three complete training-seed curves")
    return (
        pivot.index.to_numpy(dtype=float),
        pivot.mean(axis=1).to_numpy(dtype=float),
        pivot.std(axis=1, ddof=1).to_numpy(dtype=float),
    )


def _plot_training_returns(
    scalars: pd.DataFrame, output_path: Path, dpi: int
) -> None:
    reward = scalars[scalars["tag"] == "rollout/ep_rew_mean"]
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for condition in ("nominal", "robust"):
        subset = reward[reward["policy_training_condition"] == condition]
        color = CONDITION_COLORS[condition]
        for _, seed_curve in subset.groupby("policy_training_seed"):
            axis.plot(
                seed_curve["environment_steps"] / 1e6,
                seed_curve["value"],
                color=color,
                alpha=0.24,
                linewidth=1.0,
            )
        steps, mean, standard_deviation = _seed_statistics(subset, "value")
        axis.plot(
            steps / 1e6,
            mean,
            color=color,
            label=CONDITION_LABELS[condition],
            linewidth=2.4,
        )
        axis.fill_between(
            steps / 1e6,
            mean - standard_deviation,
            mean + standard_deviation,
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    axis.set_title("Training return across the six final runs")
    axis.set_xlabel("Environment steps (millions)")
    axis.set_ylabel("Episode return")
    axis.set_xlim(left=0)
    axis.legend(frameon=False, ncol=2)
    figure.text(
        0.5,
        0.015,
        "Thin lines: individual seeds. Bold line and band: mean ± 1 SD across three training seeds. "
        "Each seed curve is SB3's trailing mean over 100 completed episodes; robust returns include randomized damage.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.075, 1, 1))
    _save_figure(figure, output_path, dpi)


def _plot_strength_metric(
    axis: plt.Axes,
    by_seed: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    *,
    scale: float = 1.0,
) -> None:
    offsets = {"nominal": -0.018, "robust": 0.018}
    for condition in ("nominal", "robust"):
        subset = by_seed[by_seed["policy_training_condition"] == condition]
        color = CONDITION_COLORS[condition]
        for _, seed_values in subset.groupby("policy_training_seed"):
            seed_values = seed_values.sort_values("remaining_strength")
            axis.plot(
                seed_values["remaining_strength"] + offsets[condition],
                seed_values[metric] * scale,
                color=color,
                alpha=0.23,
                linewidth=1.0,
                marker="o",
                markersize=3,
            )
        summary = subset.groupby("remaining_strength")[metric].mean().sort_index()
        axis.plot(
            summary.index + offsets[condition],
            summary * scale,
            color=color,
            marker="o",
            markersize=5,
            label=CONDITION_LABELS[condition],
        )
    axis.set_title(title)
    axis.set_xlabel(r"Remaining actuator strength $\alpha$")
    axis.set_ylabel(ylabel)
    axis.set_xticks([0.0, 0.5, 1.0], ["0\nFailure", "0.5\nModerate", "1\nHealthy"])


def _plot_evaluation_overview(
    by_seed: pd.DataFrame, output_path: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.9), sharex=True)
    _plot_strength_metric(
        axes[0],
        by_seed,
        "horizontal_distance",
        "Distance travelled",
        "Mean horizontal distance (m)",
    )
    _plot_strength_metric(
        axes[1],
        by_seed,
        "early_termination_rate",
        "Falls before the time limit",
        "Early termination rate (%)",
        scale=100.0,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
    )
    figure.suptitle("Final policy evaluation versus damage severity", y=0.995)
    figure.text(
        0.5,
        0.01,
        "Faint paths show all three training-seed means; bold points show their arithmetic mean. "
        "Damaged points pool four legs (40 episodes per seed); healthy points use 10 episodes per seed.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.82))
    _save_figure(figure, output_path, dpi)


def _plot_leg_effects(
    by_seed: pd.DataFrame, output_path: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), sharey=True)
    x_positions = np.arange(len(LEG_ORDER), dtype=float)
    condition_offsets = {"nominal": -0.12, "robust": 0.12}
    seed_jitter = {-1: -0.035, 0: 0.0, 1: 0.035}
    for axis, alpha in zip(axes, (0.5, 0.0)):
        damage = by_seed[np.isclose(by_seed["damage_alpha"], alpha)]
        for condition in ("nominal", "robust"):
            condition_data = damage[
                damage["policy_training_condition"] == condition
            ]
            means: list[float] = []
            for leg_index, leg in enumerate(LEG_ORDER):
                leg_data = condition_data[condition_data["damage_leg"] == leg]
                values = leg_data["horizontal_distance"].to_numpy(dtype=float)
                means.append(float(values.mean()))
                for offset_index, value in zip((-1, 0, 1), values):
                    axis.scatter(
                        leg_index
                        + condition_offsets[condition]
                        + seed_jitter[offset_index],
                        value,
                        color=CONDITION_COLORS[condition],
                        alpha=0.38,
                        s=16,
                        zorder=3,
                    )
            axis.scatter(
                x_positions + condition_offsets[condition],
                means,
                color=CONDITION_COLORS[condition],
                marker="o",
                s=50,
                label=CONDITION_LABELS[condition],
                zorder=4,
            )
        axis.set_title(
            "Moderate damage ($\\alpha=0.5$)"
            if alpha == 0.5
            else "Complete failure ($\\alpha=0$)"
        )
        axis.set_xticks(x_positions, [LEG_LABELS[leg] for leg in LEG_ORDER])
        axis.tick_params(axis="x", rotation=18)
        axis.set_xlabel("Damaged leg")
    axes[0].set_ylabel("Mean horizontal distance (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    figure.suptitle("Leg-specific damage response", y=0.995)
    figure.text(
        0.5,
        0.01,
        "Small points show all three training-seed means over 10 evaluation episodes. "
        "Large points are their arithmetic mean. All damaged episodes reached the time limit (0% early termination).",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.065, 1, 0.84))
    _save_figure(figure, output_path, dpi)


def _load_sensitivity(
    summary_path: Path, configuration_path: Path
) -> pd.DataFrame:
    _require_file(summary_path)
    _require_file(configuration_path)
    frame = pd.read_csv(summary_path)
    required = {
        "configuration",
        "learning_rate",
        "clip_range",
        "moderate_mean_forward_distance",
        "moderate_mean_forward_speed",
        "moderate_early_termination_rate",
        "healthy_mean_forward_distance",
    }
    if set(frame.columns).issuperset(required) is False or len(frame) != 4:
        raise ValueError("sensitivity summary is incomplete")
    numeric = frame[list(required - {"configuration"})].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("sensitivity summary contains nonfinite values")
    expected_grid = {(1e-4, 0.1), (1e-4, 0.2), (3e-4, 0.1), (3e-4, 0.2)}
    actual_grid = set(zip(frame["learning_rate"], frame["clip_range"]))
    if actual_grid != expected_grid:
        raise ValueError("sensitivity summary does not contain the four-cell grid")
    configuration = json.loads(configuration_path.read_text())
    selected = configuration["selected_configuration"]["configuration"]
    if selected not in set(frame["configuration"]):
        raise ValueError("frozen selected configuration is absent from sensitivity data")
    frame = frame.copy()
    frame["selected"] = frame["configuration"] == selected
    return frame.sort_values(["learning_rate", "clip_range"]).reset_index(drop=True)


def _plot_sensitivity(
    sensitivity: pd.DataFrame, output_path: Path, dpi: int
) -> None:
    learning_rates = [1e-4, 3e-4]
    clip_ranges = [0.1, 0.2]
    matrix = np.empty((2, 2), dtype=float)
    records: dict[tuple[float, float], pd.Series] = {}
    for row_index, learning_rate in enumerate(learning_rates):
        for column_index, clip_range in enumerate(clip_ranges):
            row = sensitivity[
                np.isclose(sensitivity["learning_rate"], learning_rate)
                & np.isclose(sensitivity["clip_range"], clip_range)
            ].iloc[0]
            records[(learning_rate, clip_range)] = row
            matrix[row_index, column_index] = row[
                "moderate_mean_forward_distance"
            ]

    figure, axis = plt.subplots(figsize=(6.4, 4.9))
    image = axis.imshow(matrix, cmap="viridis", aspect="auto")
    midpoint = float((matrix.min() + matrix.max()) / 2)
    for row_index, learning_rate in enumerate(learning_rates):
        for column_index, clip_range in enumerate(clip_ranges):
            row = records[(learning_rate, clip_range)]
            selected = bool(row["selected"])
            annotation = (
                ("★ " if selected else "")
                + f"{row['moderate_mean_forward_distance']:.1f} m\n"
                + f"{row['moderate_mean_forward_speed']:.2f} m/s\n"
                + f"{100 * row['moderate_early_termination_rate']:.1f}% early"
            )
            axis.text(
                column_index,
                row_index,
                annotation,
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row_index, column_index] < midpoint
                    else "black"
                ),
                fontsize=10,
                fontweight="bold" if selected else "normal",
            )
    axis.set_xticks(range(2), ["0.1", "0.2"])
    axis.set_yticks(range(2), [r"$1\times10^{-4}$", r"$3\times10^{-4}$"])
    axis.set_xlabel("PPO clip range")
    axis.set_ylabel("Learning rate")
    axis.set_title("Exploratory PPO sensitivity (robust seed 6)")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.84)
    colorbar.set_label("Mean forward distance at $\\alpha=0.5$ (m)")
    figure.text(
        0.5,
        0.01,
        "Each cell continued the same preserved 5M-step policy for 1M additional steps. "
        "Annotations: distance, speed and early termination; ★ marks the frozen selection. One training seed only.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.065, 1, 1))
    _save_figure(figure, output_path, dpi)


def _plot_command_redistribution(
    command_data: pd.DataFrame, output_path: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.1, 4.8), sharey=True)
    x_positions = np.arange(len(LEG_ORDER), dtype=float)
    stage_offsets = {"raw": -0.18, "applied": 0.18}
    stage_colors = {"raw": "#A6CEE3", "applied": "#1F78B4"}
    stage_labels = {"raw": "Raw policy command", "applied": "Applied command"}
    for axis, condition in zip(axes, ("nominal", "robust")):
        subset = command_data[
            command_data["policy_training_condition"] == condition
        ]
        axis.axvspan(-0.48, 0.48, color="#D55E00", alpha=0.08, zorder=0)
        for stage in ("raw", "applied"):
            stage_data = subset[subset["command_stage"] == stage]
            means: list[float] = []
            deviations: list[float] = []
            for leg in LEG_ORDER:
                values = stage_data[stage_data["command_leg"] == leg][
                    "mean_abs_command_proxy"
                ]
                means.append(float(values.mean()))
                deviations.append(float(values.std(ddof=1)))
            axis.bar(
                x_positions + stage_offsets[stage],
                means,
                width=0.34,
                color=stage_colors[stage],
                edgecolor="#333333",
                linewidth=0.5,
                yerr=deviations,
                capsize=3,
                label=stage_labels[stage],
            )
        axis.set_title(f"{CONDITION_LABELS[condition]} policy")
        axis.set_xticks(x_positions, [LEG_LABELS[leg] for leg in LEG_ORDER])
        axis.tick_params(axis="x", rotation=18)
        axis.set_xlabel("Commanded leg")
    axes[0].set_ylabel("Mean absolute command magnitude (proxy)")
    axes[0].legend(frameon=False, loc="upper right")
    figure.suptitle(
        "Actuator command redistribution with front-left damage ($\\alpha=0.5$)",
        y=0.985,
    )
    figure.text(
        0.5,
        0.01,
        "Bars show mean ± 1 SD across three training seeds after averaging the two actuators per leg. "
        "The shaded group is damaged. Commands are control-magnitude proxies, not physical energy.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.065, 1, 0.93))
    _save_figure(figure, output_path, dpi)


def _holdout_comparison(
    preserved: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    keys = [
        "policy_training_condition",
        "policy_training_seed",
        "evaluation_seed",
        "damage_leg",
        "damage_alpha",
    ]
    if set(map(tuple, preserved[keys].to_numpy())) != set(
        map(tuple, selected[keys].to_numpy())
    ):
        raise ValueError("preserved and selected policies do not share evaluation rows")
    merged = selected.merge(preserved, on=keys, suffixes=("_selected", "_preserved"))
    records: list[dict[str, object]] = []
    for (leg, alpha), group in merged.groupby(["damage_leg", "damage_alpha"]):
        selected_distance = group["horizontal_distance_selected"]
        preserved_distance = group["horizontal_distance_preserved"]
        difference = selected_distance - preserved_distance
        records.append(
            {
                "damage_leg": leg,
                "damage_alpha": float(alpha),
                "episodes": len(group),
                "selected_mean_horizontal_distance": selected_distance.mean(),
                "selected_sd_horizontal_distance": selected_distance.std(ddof=1),
                "preserved_mean_horizontal_distance": preserved_distance.mean(),
                "preserved_sd_horizontal_distance": preserved_distance.std(ddof=1),
                "mean_paired_horizontal_difference": difference.mean(),
                "sd_paired_horizontal_difference": difference.std(ddof=1),
                "selected_early_termination_rate": group[
                    "terminated_before_time_limit_selected"
                ].mean(),
                "preserved_early_termination_rate": group[
                    "terminated_before_time_limit_preserved"
                ].mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def _case_order() -> list[tuple[str, float]]:
    return [("healthy", 1.0)] + [(leg, 0.5) for leg in LEG_ORDER] + [
        (leg, 0.0) for leg in LEG_ORDER
    ]


def _case_label(leg: str, alpha: float) -> str:
    if leg == "healthy":
        return "Healthy"
    return f"{LEG_SHORT_LABELS[leg]}\n$\\alpha={alpha:g}$"


def _plot_final_policy_comparison(
    comparison: pd.DataFrame, output_path: Path, dpi: int
) -> None:
    ordered = comparison.set_index(["damage_leg", "damage_alpha"]).loc[
        _case_order()
    ].reset_index()
    x_positions = np.arange(len(ordered), dtype=float)
    labels = [
        _case_label(str(row.damage_leg), float(row.damage_alpha))
        for row in ordered.itertuples()
    ]
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.6, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )
    model_specs = (
        (
            "preserved",
            "Original robust seed 6 (5M)",
            "#777777",
            "s",
            -0.12,
        ),
        (
            "selected",
            "Refined seed 6 (+1M, LR $10^{-4}$, clip 0.1)",
            "#009E73",
            "o",
            0.12,
        ),
    )
    for prefix, label, color, marker, offset in model_specs:
        axes[0].errorbar(
            x_positions + offset,
            ordered[f"{prefix}_mean_horizontal_distance"],
            yerr=ordered[f"{prefix}_sd_horizontal_distance"],
            color=color,
            marker=marker,
            linestyle="none",
            capsize=3,
            markersize=6,
            label=label,
        )
    differences = ordered["mean_paired_horizontal_difference"].to_numpy()
    difference_colors = np.where(differences >= 0, "#009E73", "#D55E00")
    axes[1].bar(
        x_positions,
        differences,
        yerr=ordered["sd_paired_horizontal_difference"],
        color=difference_colors,
        edgecolor="#333333",
        linewidth=0.5,
        capsize=3,
    )
    axes[1].axhline(0, color="#333333", linewidth=1.0)
    axes[0].set_ylabel("Mean horizontal distance (m)")
    axes[0].set_title("Held-out performance on common evaluation seeds")
    axes[0].legend(frameon=False, ncol=2, loc="upper center")
    axes[1].set_ylabel("Refined − original (m)")
    axes[1].set_xlabel("Evaluation condition")
    axes[1].set_xticks(x_positions, labels)
    axes[1].set_title("Paired change for the refined policy")
    figure.suptitle("Final robust policy refinement", y=0.992)
    figure.text(
        0.5,
        0.008,
        "Mean ± 1 SD across 10 held-out evaluation episodes (seeds 500–509). "
        "This is a one-training-seed post-hoc comparison; it is separate from the six-policy experiment.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.96))
    _save_figure(figure, output_path, dpi)


def _plot_design_curve(
    axis: plt.Axes,
    frame: pd.DataFrame,
    tag: str,
    design: str,
    *,
    log_scale: bool = False,
) -> None:
    subset = frame[(frame["tag"] == tag) & (frame["design"] == design)]
    color = DESIGN_COLORS[design]
    for _, seed_curve in subset.groupby("policy_training_seed"):
        axis.plot(
            seed_curve["environment_steps"] / 1e6,
            seed_curve["value"],
            color=color,
            alpha=0.2,
            linewidth=0.9,
        )
    steps, mean, standard_deviation = _seed_statistics(subset, "value")
    lower = mean - standard_deviation
    if log_scale:
        lower = np.maximum(lower, 1e-5)
    axis.plot(
        steps / 1e6,
        mean,
        color=color,
        label="Original PPO" if design == "legacy" else "Stabilized PPO",
    )
    axis.fill_between(
        steps / 1e6,
        lower,
        mean + standard_deviation,
        color=color,
        alpha=0.15,
        linewidth=0,
    )


def _plot_ppo_stabilization(
    diagnostic: pd.DataFrame, output_path: Path, dpi: int
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.35))
    plots = (
        ("rollout/ep_len_mean", "Episode duration", "Episode length (steps)"),
        ("train/approx_kl", "PPO update size", "Approximate KL divergence"),
        ("train/clip_fraction", "Clipped policy updates", "Clip fraction"),
    )
    for axis, (tag, title, ylabel) in zip(axes, plots):
        for design in ("legacy", "stabilized"):
            _plot_design_curve(
                axis,
                diagnostic,
                tag,
                design,
                log_scale=tag == "train/approx_kl",
            )
        axis.set_title(title)
        axis.set_xlabel("Environment steps (millions)")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0, 1.01)
    axes[1].set_yscale("log")
    axes[1].axhline(0.02, color="#333333", linestyle="--", linewidth=1.0)
    axes[1].text(0.03, 0.022, "target KL = 0.02", fontsize=8, va="bottom")
    axes[2].set_ylim(0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
    )
    figure.suptitle("Why the PPO training setup was revised (nominal runs)", y=0.995)
    figure.text(
        0.5,
        0.008,
        "Thin curves are individual runs; bold curves and bands are mean ± 1 SD across three seeds. "
        "Development diagnostic only: seeds, reward setting and stabilization elements differ, so it does not isolate one causal change.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.065, 1, 0.82))
    _save_figure(figure, output_path, dpi)


def _figure_metadata(include_diagnostic: bool) -> dict[str, object]:
    figures: dict[str, object] = {
        "final_training_returns.png": {
            "source": "final TensorBoard rollout/ep_rew_mean scalars",
            "smoothing": "SB3 trailing mean over 100 completed episodes",
            "uncertainty": "sample standard deviation across three training seeds",
            "caveat": "robust returns are collected under randomized damage",
        },
        "final_evaluation_overview.png": {
            "source": "results/final_episode_results.csv",
            "aggregation": "episodes within policy seed, damaged legs within severity, then training seeds",
            "uncertainty": "all three training-seed means are shown directly; bold points are their arithmetic mean",
            "metrics": "horizontal distance and early termination rate",
        },
        "final_leg_effects.png": {
            "source": "results/final_episode_results.csv",
            "aggregation": "10 evaluation episodes within each policy seed and damaged leg",
            "uncertainty": "all three training-seed means are shown directly; large points are their arithmetic mean",
            "metric": "horizontal distance",
        },
        "final_sensitivity.png": {
            "source": "results/sensitivity_summary.csv",
            "aggregation": "one selected training seed; four legs pooled at alpha=0.5",
            "uncertainty": "not shown because the study used one training seed",
        },
        "final_command_redistribution.png": {
            "source": "results/final_episode_results.csv",
            "aggregation": "two actuator magnitudes per leg, episodes within policy seed, then training seeds",
            "uncertainty": "sample standard deviation across three training-seed means",
            "units": "mean absolute normalized command magnitude proxy, not physical energy",
        },
        "final_policy_refinement.png": {
            "source": "matched held-out evaluation CSVs for original and refined robust seed 6",
            "aggregation": "10 paired evaluation seeds per damage case",
            "uncertainty": "sample standard deviation across evaluation episodes, not training seeds",
        },
    }
    if include_diagnostic:
        figures["ppo_stabilization_diagnostic.png"] = {
            "source": "legacy and final nominal TensorBoard scalars",
            "aggregation": "mean across three runs for each training design",
            "uncertainty": "sample standard deviation across three runs",
            "caveat": "development diagnostic; multiple configuration elements and seeds differ",
        }
    return {"schema_version": 1, "figures": figures}


# One command rebuilds every figure and its processed source table.
def generate_figures(args: argparse.Namespace) -> list[Path]:
    _configure_style()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    final_evaluation = _load_evaluation(args.final_results)
    _validate_matrix(
        final_evaluation,
        conditions=("nominal", "robust"),
        training_seeds=FINAL_TRAINING_SEEDS,
        evaluation_seeds=FINAL_EVALUATION_SEEDS,
    )
    evaluation_by_seed = _evaluation_by_seed(final_evaluation)
    evaluation_summary = _evaluation_summary(evaluation_by_seed)
    leg_performance = _leg_performance_by_seed(final_evaluation)
    command_magnitudes = _command_magnitudes_by_seed(final_evaluation)
    final_training = _final_training_scalars(args.final_artifacts)
    sensitivity = _load_sensitivity(
        args.sensitivity_results, args.final_configuration
    )

    selected = _load_evaluation(args.selected_results)
    preserved = _load_evaluation(args.preserved_results)
    for frame in (selected, preserved):
        _validate_matrix(
            frame,
            conditions=("robust",),
            training_seeds=(6,),
            evaluation_seeds=HOLDOUT_EVALUATION_SEEDS,
        )
    holdout = _holdout_comparison(preserved, selected)

    processed_outputs = {
        "final_training_scalars.csv": final_training,
        "final_evaluation_by_seed.csv": evaluation_by_seed,
        "final_evaluation_summary.csv": evaluation_summary,
        "final_leg_performance_by_seed.csv": leg_performance,
        "final_command_magnitudes_by_seed.csv": command_magnitudes,
        "final_sensitivity_figure_data.csv": sensitivity,
        "final_policy_holdout_comparison.csv": holdout,
    }
    for filename, frame in processed_outputs.items():
        _atomic_csv(frame, args.processed_dir / filename)

    figure_paths = [
        args.figures_dir / "final_training_returns.png",
        args.figures_dir / "final_evaluation_overview.png",
        args.figures_dir / "final_leg_effects.png",
        args.figures_dir / "final_sensitivity.png",
        args.figures_dir / "final_command_redistribution.png",
        args.figures_dir / "final_policy_refinement.png",
    ]
    _plot_training_returns(final_training, figure_paths[0], args.dpi)
    _plot_evaluation_overview(evaluation_by_seed, figure_paths[1], args.dpi)
    _plot_leg_effects(leg_performance, figure_paths[2], args.dpi)
    _plot_sensitivity(sensitivity, figure_paths[3], args.dpi)
    _plot_command_redistribution(command_magnitudes, figure_paths[4], args.dpi)
    _plot_final_policy_comparison(holdout, figure_paths[5], args.dpi)

    include_diagnostic = not args.skip_development_diagnostic
    additional_processed_paths: list[Path] = []
    if include_diagnostic:
        legacy = _legacy_nominal_scalars(args.legacy_artifacts)
        stabilized = final_training[
            (final_training["policy_training_condition"] == "nominal")
            & (final_training["environment_steps"] <= 1_007_616)
            & final_training["tag"].isin(
                ("rollout/ep_len_mean", "train/approx_kl", "train/clip_fraction")
            )
        ].copy()
        stabilized.insert(0, "design", "stabilized")
        stabilized = stabilized.drop(columns="policy_training_condition")
        diagnostic = pd.concat([legacy, stabilized], ignore_index=True)
        diagnostic_data_path = args.processed_dir / "ppo_stabilization_curves.csv"
        _atomic_csv(diagnostic, diagnostic_data_path)
        additional_processed_paths.append(diagnostic_data_path)
        diagnostic_path = args.figures_dir / "ppo_stabilization_diagnostic.png"
        _plot_ppo_stabilization(diagnostic, diagnostic_path, args.dpi)
        figure_paths.append(diagnostic_path)

    metadata_path = args.processed_dir / "figure_metadata.json"
    _atomic_json(_figure_metadata(include_diagnostic), metadata_path)
    return [
        *figure_paths,
        *[args.processed_dir / name for name in processed_outputs],
        *additional_processed_paths,
        metadata_path,
    ]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = generate_figures(args)
    print("Generated final report figures and processed data:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
