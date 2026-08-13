"""Run staged long training for one selected Ant policy."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from damage_robust_ant.damage import LEG_ACTION_INDICES
from damage_robust_ant.evaluate import CSV_COLUMNS
from damage_robust_ant.train import (
    _assert_finite_model,
    _damage_configuration,
    _git_provenance,
    _package_versions,
    _ppo_configuration,
    make_ppo,
    make_training_env,
)


POLICIES = {
    "nominal": {
        "seed": 1,
        "target": "healthy",
        "selection_rule": "lowest healthy mean forward distance in the main run",
        "selection_score": 0.259630176186546,
    },
    "robust": {
        "seed": 2,
        "target": "moderate_damage",
        "selection_rule": (
            "highest alpha=0.5 mean forward distance across all four legs "
            "in the main run"
        ),
        "selection_score": 4.319067590614017,
    },
}
CHECKPOINT_INTERVAL_STEPS = 250_000
VALIDATION_INTERVAL_STEPS = 500_000
FIRST_VALIDATION_STEPS = 1_000_000
DECISION_STEPS = {3_000_000, 4_000_000, 5_000_000}
MAXIMUM_TRAINING_STEPS = 5_000_000
VALIDATION_SEED = 200
VALIDATION_EPISODES = 10
MAX_EARLY_TERMINATION_RATE = 0.10
MIN_MEAN_FORWARD_SPEED = 0.50
MIN_POSITIVE_DISTANCE_RATE = 0.90
PROGRESS_INTERVAL_SECONDS = 300.0
MAIN_RESULTS_RELATIVE_PATH = Path("results/main_episode_results.csv")
MAIN_RESULTS_SHA256 = "f011949b67050ee96854ab80ea9305bb14e072d993a88850b36488debf6f8508"


def _positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed_value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments for one selected long-training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(POLICIES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-index", type=_positive_int, default=1)
    parser.add_argument("--total-policies", type=_positive_int, default=1)
    args = parser.parse_args(argv)
    if args.policy_index > args.total_policies:
        parser.error("--policy-index cannot exceed --total-policies")
    return args


def _format_duration(seconds: float) -> str:
    total_minutes = max(round(seconds / 60), 0)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


class LongTrainingProgressCallback(BaseCallback):
    """Print cumulative overnight progress every five minutes."""

    def __init__(
        self,
        condition: str,
        seed: int,
        policy_index: int,
        total_policies: int,
        policy_started_at: float,
        interval_seconds: float = PROGRESS_INTERVAL_SECONDS,
    ) -> None:
        super().__init__()
        self.condition = condition
        self.seed = seed
        self.policy_index = policy_index
        self.total_policies = total_policies
        self.policy_started_at = policy_started_at
        self.interval_seconds = interval_seconds
        self.next_update_at: float | None = None

    def _on_training_start(self) -> None:
        if self.next_update_at is None:
            self.next_update_at = time.perf_counter() + self.interval_seconds

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if self.next_update_at is None or now < self.next_update_at:
            return True

        current_steps = min(self.num_timesteps, MAXIMUM_TRAINING_STEPS)
        elapsed = now - self.policy_started_at
        steps_per_second = current_steps / elapsed
        maximum_steps_left = MAXIMUM_TRAINING_STEPS - current_steps
        maximum_steps_left += (
            self.total_policies - self.policy_index
        ) * MAXIMUM_TRAINING_STEPS
        print(
            f"[Overnight progress] Policy {self.policy_index}/"
            f"{self.total_policies} ({self.condition} seed {self.seed}): "
            f"{current_steps:,}/{MAXIMUM_TRAINING_STEPS:,} maximum steps; "
            f"rough maximum-budget ETA "
            f"{_format_duration(maximum_steps_left / steps_per_second)}."
        )
        self.next_update_at = now + self.interval_seconds
        return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rollout_target(requested_steps: int, rollout_size: int) -> int:
    """Finish a rollout without crossing the hard five-million-step cap."""
    rounded_up = math.ceil(requested_steps / rollout_size) * rollout_size
    final_allowed = (MAXIMUM_TRAINING_STEPS // rollout_size) * rollout_size
    return min(rounded_up, final_allowed)


def _prepare_output(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "run": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "monitor": output_dir / "monitor",
        "tensorboard": output_dir / "tensorboard",
        "validation": output_dir / "validation",
        "summary": output_dir / "validation_summary.csv",
        "model": output_dir / "selected_model.zip",
        "metadata": output_dir / "metadata.json",
    }
    for name in ("checkpoints", "monitor", "tensorboard", "validation"):
        paths[name].mkdir()
    return paths


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _write_summary(path: Path, records: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    pd.DataFrame(records).to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _evaluate_checkpoint(
    checkpoint: Path,
    condition: str,
    seed: int,
    actual_steps: int,
    validation_dir: Path,
) -> Path:
    """Evaluate in subprocesses so loading cannot change the trainer RNG."""
    output_csv = validation_dir / f"checkpoint_{actual_steps}_episodes.csv"
    conditions = [("healthy", 1.0), *[(leg, 0.5) for leg in LEG_ACTION_INDICES]]
    for index, (leg, alpha) in enumerate(conditions):
        command = [
            sys.executable,
            "-m",
            "damage_robust_ant.evaluate",
            "--model",
            str(checkpoint),
            "--training-condition",
            condition,
            "--training-seed",
            str(seed),
            "--damage-leg",
            leg,
            "--alpha",
            str(alpha),
            "--evaluation-seed",
            str(VALIDATION_SEED),
            "--episodes",
            str(VALIDATION_EPISODES),
            "--output-csv",
            str(output_csv),
        ]
        if index:
            command.append("--append")
        subprocess.run(command, check=True)
    return output_csv


def _locomotion_metrics(rows: pd.DataFrame) -> dict[str, object]:
    values = rows[
        ["forward_distance", "mean_forward_speed", "episode_length"]
    ].to_numpy()
    mean_distance = float(rows["forward_distance"].mean())
    mean_speed = float(rows["mean_forward_speed"].mean())
    early_rate = float(rows["terminated_before_time_limit"].mean())
    positive_rate = float((rows["forward_distance"] > 0.0).mean())
    return {
        "mean_forward_distance": mean_distance,
        "mean_forward_speed": mean_speed,
        "early_termination_rate": early_rate,
        "positive_distance_rate": positive_rate,
        "criteria_met": bool(
            np.isfinite(values).all()
            and early_rate <= MAX_EARLY_TERMINATION_RATE
            and mean_speed >= MIN_MEAN_FORWARD_SPEED
            and positive_rate >= MIN_POSITIVE_DISTANCE_RATE
        ),
    }


def _summarize_validation(
    evaluation_csv: Path,
    condition: str,
    requested_steps: int,
    actual_steps: int,
) -> dict[str, object]:
    frame = pd.read_csv(evaluation_csv)
    policy = POLICIES[condition]
    seed = int(policy["seed"])
    keys = [
        "policy_training_condition",
        "policy_training_seed",
        "evaluation_seed",
        "damage_leg",
        "damage_alpha",
    ]
    nonnumeric = {
        "policy_training_condition",
        "damage_leg",
        "terminated_before_time_limit",
    }
    numeric = [column for column in CSV_COLUMNS if column not in nonnumeric]
    expected_cases = {("healthy", 1.0)} | {
        (leg, 0.5) for leg in LEG_ACTION_INDICES
    }
    actual_cases = set(
        zip(frame.get("damage_leg", []), frame.get("damage_alpha", []))
    )
    numeric_values = frame.reindex(columns=numeric).apply(
        pd.to_numeric,
        errors="coerce",
    )
    episode_lengths = numeric_values["episode_length"]
    if (
        len(frame) != 50
        or list(frame.columns) != CSV_COLUMNS
        or frame.duplicated(keys).any()
        or frame.isna().any().any()
        or actual_cases != expected_cases
        or not np.isfinite(numeric_values.to_numpy()).all()
        or not pd.api.types.is_bool_dtype(
            frame["terminated_before_time_limit"]
        )
        or (episode_lengths < 1).any()
        or not np.equal(episode_lengths, np.floor(episode_lengths)).all()
        or set(frame["policy_training_condition"]) != {condition}
        or set(frame["policy_training_seed"]) != {seed}
    ):
        raise RuntimeError("checkpoint validation is incomplete or invalid")

    expected_seeds = set(range(VALIDATION_SEED, VALIDATION_SEED + VALIDATION_EPISODES))
    for _, rows in frame.groupby(["damage_leg", "damage_alpha"]):
        complete_seed_set = set(rows["evaluation_seed"]) == expected_seeds
        if len(rows) != VALIDATION_EPISODES or not complete_seed_set:
            raise RuntimeError("checkpoint validation used an incomplete seed set")

    healthy = frame[
        (frame["damage_leg"] == "healthy") & (frame["damage_alpha"] == 1.0)
    ]
    moderate = frame[
        (frame["damage_leg"].isin(LEG_ACTION_INDICES))
        & (frame["damage_alpha"] == 0.5)
    ]
    metrics = {
        "healthy": _locomotion_metrics(healthy),
        "moderate_damage": _locomotion_metrics(moderate),
    }
    metrics.update(
        {
            f"moderate_damage_{leg}": _locomotion_metrics(
                moderate[moderate["damage_leg"] == leg]
            )
            for leg in LEG_ACTION_INDICES
        }
    )
    target = str(policy["target"])
    record: dict[str, object] = {
        "condition": condition,
        "seed": seed,
        "requested_checkpoint_steps": requested_steps,
        "actual_checkpoint_steps": actual_steps,
        "target_condition": target,
        "target_criteria_met": metrics[target]["criteria_met"],
    }
    for prefix, condition_metrics in metrics.items():
        for name, value in condition_metrics.items():
            record[f"{prefix}_{name}"] = value
    return record


def _score(record: dict[str, object]) -> tuple[float, float, float, float, int]:
    target = str(record["target_condition"])
    opposite = "moderate_damage" if target == "healthy" else "healthy"
    return (
        float(record[f"{target}_mean_forward_distance"]),
        -float(record[f"{target}_early_termination_rate"]),
        float(record[f"{target}_mean_forward_speed"]),
        float(record[f"{opposite}_mean_forward_distance"]),
        -int(record["actual_checkpoint_steps"]),
    )


def _selection_decision(
    records: list[dict[str, object]],
    requested_steps: int,
) -> tuple[dict[str, object] | None, bool]:
    """Choose only after a complete 3M, 4M, or 5M training stage."""
    if requested_steps not in DECISION_STEPS:
        return None, False
    passing = [record for record in records if record["target_criteria_met"]]
    if passing:
        return max(passing, key=_score), True
    if requested_steps == MAXIMUM_TRAINING_STEPS:
        return max(records, key=_score), False
    return None, False


def run_long_training(args: argparse.Namespace) -> Path:
    """Train a fresh policy and select it using fixed validation criteria."""
    policy = POLICIES[args.condition]
    seed = int(policy["seed"])
    git_commit, tracked_worktree_clean = _git_provenance()
    if not tracked_worktree_clean:
        raise RuntimeError("tracked files must be clean before long training")
    repository = Path(__file__).resolve().parents[2]
    main_results = repository / MAIN_RESULTS_RELATIVE_PATH
    if not main_results.is_file() or _sha256(main_results) != MAIN_RESULTS_SHA256:
        raise RuntimeError("main evaluation results do not match the selected seeds")

    paths = _prepare_output(args.output_dir)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "purpose": "post_hoc_locomotion_follow_up",
        "included_in_main_comparison": False,
        "status": "training",
        "condition": args.condition,
        "seed": seed,
        "initialization": "new_network",
        "main_model_reused": False,
        "selection": policy,
        "selection_source": {
            "path": str(MAIN_RESULTS_RELATIVE_PATH),
            "sha256": MAIN_RESULTS_SHA256,
        },
        "git_commit": git_commit,
        "tracked_worktree_clean": tracked_worktree_clean,
        "package_versions": _package_versions(),
        "schedule": {
            "checkpoint_interval_requested_steps": CHECKPOINT_INTERVAL_STEPS,
            "validation_interval_requested_steps": VALIDATION_INTERVAL_STEPS,
            "first_validation_requested_steps": FIRST_VALIDATION_STEPS,
            "decision_requested_steps": sorted(DECISION_STEPS),
            "hard_maximum_environment_steps": MAXIMUM_TRAINING_STEPS,
            "first_chunk_resets_timesteps": True,
            "later_chunks_reset_timesteps": False,
        },
        "validation": {
            "deterministic": True,
            "episode_seeds": list(
                range(VALIDATION_SEED, VALIDATION_SEED + VALIDATION_EPISODES)
            ),
            "target_condition": policy["target"],
            "criteria": {
                "maximum_early_termination_rate": MAX_EARLY_TERMINATION_RATE,
                "minimum_mean_forward_speed": MIN_MEAN_FORWARD_SPEED,
                "minimum_positive_distance_rate": MIN_POSITIVE_DISTANCE_RATE,
            },
        },
    }
    _write_json(paths["metadata"], metadata)

    env = None
    model = None
    records: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    training_seconds = 0.0
    started_at = time.perf_counter()
    try:
        env = make_training_env(args.condition, 4, seed, paths["monitor"])
        model = make_ppo(env, 3e-4, 0.2, seed, paths["tensorboard"])
        configuration = _ppo_configuration(model, 3e-4, 0.2, seed)
        configuration.pop("reset_num_timesteps")
        metadata["ppo_configuration"] = configuration
        metadata["damage_configuration"] = _damage_configuration(args.condition)
        rollout_size = model.n_steps * env.num_envs
        metadata["rollout_size_environment_steps"] = rollout_size
        progress = LongTrainingProgressCallback(
            args.condition,
            seed,
            args.policy_index,
            args.total_policies,
            started_at,
        )
        print(
            "[Overnight progress] A status line will appear about every five minutes."
        )

        selected_record = None
        criteria_met = False
        for requested_steps in range(
            CHECKPOINT_INTERVAL_STEPS,
            MAXIMUM_TRAINING_STEPS + 1,
            CHECKPOINT_INTERVAL_STEPS,
        ):
            target_steps = _rollout_target(requested_steps, rollout_size)
            chunk_started_at = time.perf_counter()
            model.learn(
                total_timesteps=target_steps - model.num_timesteps,
                callback=progress,
                log_interval=1,
                tb_log_name="PPO",
                reset_num_timesteps=model.num_timesteps == 0,
            )
            training_seconds += time.perf_counter() - chunk_started_at
            model.logger.dump(model.num_timesteps)
            model.logger.close()
            if model.num_timesteps != target_steps:
                raise RuntimeError(
                    "training stopped away from the expected rollout boundary"
                )
            _assert_finite_model(model)

            checkpoint = paths["checkpoints"] / f"ppo_{model.num_timesteps}_steps.zip"
            model.save(checkpoint)
            checkpoints.append(
                {
                    "requested_steps": requested_steps,
                    "actual_steps": model.num_timesteps,
                    "path": str(checkpoint.relative_to(paths["run"])),
                    "sha256": _sha256(checkpoint),
                }
            )
            metadata.update(
                {
                    "checkpoints": checkpoints,
                    "total_search_environment_steps": model.num_timesteps,
                    "elapsed_training_seconds": training_seconds,
                    "training_fps": model.num_timesteps / training_seconds,
                }
            )
            _write_json(paths["metadata"], metadata)

            if (
                requested_steps >= FIRST_VALIDATION_STEPS
                and requested_steps % VALIDATION_INTERVAL_STEPS == 0
            ):
                print(
                    f"[Validation] {args.condition} seed {seed} at "
                    f"{model.num_timesteps:,} steps"
                )
                evaluation_csv = _evaluate_checkpoint(
                    checkpoint,
                    args.condition,
                    seed,
                    model.num_timesteps,
                    paths["validation"],
                )
                record = _summarize_validation(
                    evaluation_csv,
                    args.condition,
                    requested_steps,
                    model.num_timesteps,
                )
                record["checkpoint"] = checkpoints[-1]["path"]
                record["episode_results"] = str(
                    evaluation_csv.relative_to(paths["run"])
                )
                records.append(record)
                _write_summary(paths["summary"], records)

                target = str(policy["target"])
                print(
                    f"[Validation] target={target} "
                    f"distance={record[f'{target}_mean_forward_distance']:.3f} "
                    f"speed={record[f'{target}_mean_forward_speed']:.3f} "
                    f"early_rate={record[f'{target}_early_termination_rate']:.3f} "
                    f"positive_rate={record[f'{target}_positive_distance_rate']:.3f} "
                    f"criteria_met={record['target_criteria_met']}"
                )
                selected_record, criteria_met = _selection_decision(
                    records,
                    requested_steps,
                )

            metadata["validation_results"] = records
            _write_json(paths["metadata"], metadata)
            if selected_record is not None:
                break

        if selected_record is None:
            raise RuntimeError("training ended without selecting a checkpoint")

        selected_checkpoint = paths["run"] / str(selected_record["checkpoint"])
        shutil.copy2(selected_checkpoint, paths["model"])
        reloaded = PPO.load(paths["model"], env=env, device="cpu")
        _assert_finite_model(reloaded)
        if reloaded.num_timesteps != selected_record["actual_checkpoint_steps"]:
            raise RuntimeError("selected model has an unexpected timestep count")
        observation = env.reset()
        action, _ = reloaded.predict(observation, deterministic=True)
        if not np.isfinite(action).all():
            raise RuntimeError("selected model produced a non-finite action")

        metadata.update(
            {
                "status": (
                    "locomotion_criteria_met"
                    if criteria_met
                    else "maximum_reached_without_meeting_criteria"
                ),
                "criteria_met": criteria_met,
                "selected_checkpoint": selected_record,
                "selected_model_environment_steps": selected_record[
                    "actual_checkpoint_steps"
                ],
                "selected_model": {
                    "path": paths["model"].name,
                    "sha256": _sha256(paths["model"]),
                },
                "elapsed_total_seconds": time.perf_counter() - started_at,
                "selected_model_reloaded": True,
                "finite_selected_prediction": True,
            }
        )
        _write_json(paths["metadata"], metadata)
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        metadata["elapsed_total_seconds"] = time.perf_counter() - started_at
        _write_json(paths["metadata"], metadata)
        raise
    finally:
        if model is not None and hasattr(model, "_logger"):
            model.logger.close()
        if env is not None:
            env.close()

    print(
        f"Long training finished: condition={args.condition} seed={seed} "
        f"criteria_met={metadata['criteria_met']} "
        f"selected_steps={metadata['selected_model_environment_steps']}"
    )
    print(f"Selected model: {paths['model']}")
    return paths["model"]


def main() -> None:
    """Run one selected long-training policy from the command line."""
    run_long_training(parse_args())


if __name__ == "__main__":
    main()
