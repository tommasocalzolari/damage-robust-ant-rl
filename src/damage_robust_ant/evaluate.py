"""Evaluate a saved Ant PPO policy under a controlled damage condition."""

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from damage_robust_ant.damage import AntDamageWrapper, LEG_ACTION_INDICES


ACTUATOR_COUNT = 8
HEALTHY_LEG = "healthy"
CSV_COLUMNS = [
    "policy_training_condition",
    "policy_training_seed",
    "evaluation_seed",
    "damage_leg",
    "damage_alpha",
    "episode_return",
    "episode_length",
    "terminated_before_time_limit",
    "forward_distance",
    "mean_forward_speed",
    *[
        f"mean_abs_raw_command_actuator_{index}"
        for index in range(ACTUATOR_COUNT)
    ],
    *[
        f"mean_abs_applied_command_actuator_{index}"
        for index in range(ACTUATOR_COUNT)
    ],
]


def _positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed_value


def _nonnegative_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed_value


def _unit_float(value: str) -> float:
    parsed_value = float(value)
    if not 0.0 <= parsed_value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed_value


def _existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse evaluation command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=_existing_file, required=True)
    parser.add_argument(
        "--training-condition",
        choices=("nominal", "robust"),
        required=True,
    )
    parser.add_argument("--training-seed", type=_nonnegative_int, required=True)
    parser.add_argument(
        "--damage-leg",
        choices=(HEALTHY_LEG, *LEG_ACTION_INDICES),
        required=True,
    )
    parser.add_argument("--alpha", type=_unit_float, required=True)
    parser.add_argument("--evaluation-seed", type=_nonnegative_int, default=0)
    parser.add_argument("--episodes", type=_positive_int, default=10)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--append",
        action="store_true",
        help="append to an existing CSV with the same schema",
    )
    args = parser.parse_args(argv)
    if args.damage_leg == HEALTHY_LEG and args.alpha != 1.0:
        parser.error("--damage-leg healthy requires --alpha 1.0")
    return args


def _make_evaluation_env(damage_leg: str, alpha: float) -> AntDamageWrapper:
    base_env = gym.make("Ant-v5")
    try:
        if damage_leg == HEALTHY_LEG:
            return AntDamageWrapper(base_env, mode="nominal")
        return AntDamageWrapper(
            base_env,
            mode="fixed",
            fixed_leg=damage_leg,
            fixed_alpha=alpha,
        )
    except Exception:
        base_env.close()
        raise


def _finite_float(value: Any, name: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value):
        raise RuntimeError(f"non-finite {name}")
    return parsed_value


def _finite_array(
    value: Any,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if expected_shape is not None and array.shape != expected_shape:
        raise RuntimeError(
            f"unexpected {name} shape: {array.shape}, expected {expected_shape}"
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise RuntimeError(f"non-finite {name}")
    return array


def _expected_damage_leg(damage_leg: str) -> str | None:
    return None if damage_leg == HEALTHY_LEG else damage_leg


def _check_damage_info(
    info: dict[str, Any],
    damage_leg: str,
    alpha: float,
) -> None:
    expected_leg = _expected_damage_leg(damage_leg)
    if info.get("damage_leg") != expected_leg:
        raise RuntimeError("environment reported an unexpected damage leg")
    reported_alpha = _finite_float(info.get("damage_alpha"), "damage alpha")
    if not math.isclose(reported_alpha, alpha, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("environment reported an unexpected damage alpha")


def _check_applied_action(
    policy_action: np.ndarray,
    applied_action: np.ndarray,
    damage_leg: str,
    alpha: float,
) -> None:
    expected_action = policy_action.copy()
    if damage_leg != HEALTHY_LEG:
        indices = list(LEG_ACTION_INDICES[damage_leg])
        expected_action[indices] *= alpha
    if not np.allclose(applied_action, expected_action, rtol=1e-7, atol=1e-8):
        raise RuntimeError("applied action does not match the requested damage")


def _policy_snapshot(model: PPO) -> tuple[int, dict[str, torch.Tensor]]:
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.policy.state_dict().items()
    }
    return int(model.num_timesteps), state


def _assert_policy_unchanged(
    model: PPO,
    initial_timesteps: int,
    initial_state: dict[str, torch.Tensor],
) -> None:
    if model.num_timesteps != initial_timesteps:
        raise RuntimeError("evaluation changed the model timestep count")
    current_state = model.policy.state_dict()
    if current_state.keys() != initial_state.keys():
        raise RuntimeError("evaluation changed the policy state")
    for name, original_value in initial_state.items():
        if not torch.equal(current_state[name].detach().cpu(), original_value):
            raise RuntimeError(f"evaluation changed policy tensor: {name}")


def _evaluate_episodes(
    model: PPO,
    env: AntDamageWrapper,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    initial_timesteps, initial_state = _policy_snapshot(model)
    time_step = _finite_float(env.unwrapped.dt, "environment time step")
    if time_step <= 0.0:
        raise RuntimeError("environment time step must be positive")
    rows = []

    for episode_index in range(args.episodes):
        episode_seed = args.evaluation_seed + episode_index
        observation, reset_info = env.reset(seed=episode_seed)
        _finite_array(observation, "reset observation")
        _check_damage_info(reset_info, args.damage_leg, args.alpha)
        initial_x = _finite_float(reset_info.get("x_position"), "initial x position")

        episode_return = 0.0
        episode_length = 0
        raw_command_sum = np.zeros(ACTUATOR_COUNT, dtype=np.float64)
        applied_command_sum = np.zeros(ACTUATOR_COUNT, dtype=np.float64)

        while True:
            action, _ = model.predict(observation, deterministic=True)
            action = _finite_array(action, "policy action", (ACTUATOR_COUNT,))
            observation, reward, terminated, truncated, step_info = env.step(action)
            _finite_array(observation, "step observation")
            _check_damage_info(step_info, args.damage_leg, args.alpha)

            policy_action = _finite_array(
                env.policy_action,
                "raw policy command",
                (ACTUATOR_COUNT,),
            )
            applied_action = _finite_array(
                env.applied_action,
                "applied command",
                (ACTUATOR_COUNT,),
            )
            if not np.array_equal(policy_action, action):
                raise RuntimeError("wrapper did not retain the raw policy command")
            _check_applied_action(
                policy_action,
                applied_action,
                args.damage_leg,
                args.alpha,
            )

            episode_return += _finite_float(reward, "reward")
            episode_length += 1
            raw_command_sum += np.abs(policy_action)
            applied_command_sum += np.abs(applied_action)

            if terminated or truncated:
                final_x = _finite_float(
                    step_info.get("x_position"),
                    "final x position",
                )
                break

        forward_distance = final_x - initial_x
        mean_forward_speed = forward_distance / (episode_length * time_step)
        _finite_float(episode_return, "episode return")
        _finite_float(forward_distance, "forward distance")
        _finite_float(mean_forward_speed, "mean forward speed")

        mean_raw_commands = raw_command_sum / episode_length
        mean_applied_commands = applied_command_sum / episode_length
        row: dict[str, object] = {
            "policy_training_condition": args.training_condition,
            "policy_training_seed": args.training_seed,
            "evaluation_seed": episode_seed,
            "damage_leg": args.damage_leg,
            "damage_alpha": args.alpha,
            "episode_return": episode_return,
            "episode_length": episode_length,
            "terminated_before_time_limit": bool(terminated and not truncated),
            "forward_distance": forward_distance,
            "mean_forward_speed": mean_forward_speed,
        }
        row.update(
            {
                f"mean_abs_raw_command_actuator_{index}": value
                for index, value in enumerate(mean_raw_commands)
            }
        )
        row.update(
            {
                f"mean_abs_applied_command_actuator_{index}": value
                for index, value in enumerate(mean_applied_commands)
            }
        )
        rows.append(row)

    _assert_policy_unchanged(model, initial_timesteps, initial_state)
    return rows


def _row_key(row: dict[str, object]) -> tuple[str, int, int, str, float]:
    return (
        str(row["policy_training_condition"]),
        int(row["policy_training_seed"]),
        int(row["evaluation_seed"]),
        str(row["damage_leg"]),
        float(row["damage_alpha"]),
    )


def _load_existing_rows(output_csv: Path, append: bool) -> list[dict[str, object]]:
    if not output_csv.exists():
        return []
    if not append:
        raise FileExistsError(f"output CSV already exists: {output_csv}")

    with output_csv.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError("existing CSV has an incompatible schema")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("existing CSV contains malformed rows")
    return rows


def _check_duplicate_rows(rows: list[dict[str, object]]) -> None:
    keys = [_row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("CSV would contain duplicate evaluation rows")


def _write_results(
    output_csv: Path,
    existing_rows: list[dict[str, object]],
    new_rows: list[dict[str, object]],
    append: bool,
) -> None:
    rows = [*existing_rows, *new_rows]
    _check_duplicate_rows(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output_csv.with_name(f"{output_csv.name}.tmp")
    with temporary_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if output_csv.exists() and not append:
        temporary_csv.unlink()
        raise FileExistsError(f"output CSV already exists: {output_csv}")
    temporary_csv.replace(output_csv)


def _print_summary(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    mean_return = float(np.mean([row["episode_return"] for row in rows]))
    mean_distance = float(np.mean([row["forward_distance"] for row in rows]))
    mean_speed = float(np.mean([row["mean_forward_speed"] for row in rows]))
    mean_length = float(np.mean([row["episode_length"] for row in rows]))
    early_rate = float(
        np.mean([row["terminated_before_time_limit"] for row in rows])
    )
    print(
        f"Evaluated {args.training_condition} policy seed={args.training_seed} "
        f"damage_leg={args.damage_leg} alpha={args.alpha:.3f} "
        f"episodes={args.episodes}"
    )
    print(
        f"mean_return={mean_return:.3f} mean_distance={mean_distance:.3f} "
        f"mean_speed={mean_speed:.3f} mean_length={mean_length:.1f} "
        f"early_termination_rate={early_rate:.3f}"
    )
    print("Model parameters unchanged: yes")
    print(f"CSV: {args.output_csv}")


def run_evaluation(args: argparse.Namespace) -> Path:
    """Evaluate one model and write one row per episode."""
    existing_rows = _load_existing_rows(args.output_csv, args.append)
    model = PPO.load(args.model, device="cpu")
    env: AntDamageWrapper | None = None
    try:
        env = _make_evaluation_env(args.damage_leg, args.alpha)
        rows = _evaluate_episodes(model, env, args)
    finally:
        if env is not None:
            env.close()

    _write_results(args.output_csv, existing_rows, rows, args.append)
    _print_summary(args, rows)
    return args.output_csv


def main() -> None:
    """Run controlled evaluation from the command line."""
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
