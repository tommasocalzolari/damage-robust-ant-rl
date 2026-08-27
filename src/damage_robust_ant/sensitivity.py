"""Run the four-configuration PPO sensitivity study.

The study starts each variant from a copy of the preserved robust seed-6
checkpoint.  The checkpoint itself is never modified; each variant receives
one additional, rollout-aligned million environment steps and is evaluated
under the same five conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan, VecEnv, VecNormalize

from damage_robust_ant.damage import LEG_ACTION_INDICES, AntDamageWrapper, make_ant_env
from damage_robust_ant.evaluate import CSV_COLUMNS
from damage_robust_ant.train import (
    _assert_finite_model,
    _damage_configuration,
    _git_provenance,
    _normalization_configuration,
    _ppo_configuration,
)


SOURCE_DEFAULT = Path("artifacts/final/robust_seed_6/final_model.zip")
NORMALIZER_DEFAULT = Path("artifacts/final/robust_seed_6/vecnormalize.pkl")
STEPS_PER_VARIANT = 1_000_000
NUM_ENVS = 4
ROLLOUT_SIZE = 2_048 * NUM_ENVS
EVALUATION_SEED = 400
EVALUATION_EPISODES = 10
CONFIGURATIONS = (
    (1e-4, 0.1),
    (1e-4, 0.2),
    (3e-4, 0.1),
    (3e-4, 0.2),
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=_existing_file, default=SOURCE_DEFAULT)
    parser.add_argument("--normalizer", type=_existing_file, default=NORMALIZER_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timesteps", type=_positive_int, default=STEPS_PER_VARIANT)
    parser.add_argument("--evaluation-seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--episodes", type=_positive_int, default=EVALUATION_EPISODES)
    parser.add_argument(
        "--configuration",
        choices=(
            "lr_1em04_clip_0.1",
            "lr_1em04_clip_0.2",
            "lr_3em04_clip_0.1",
            "lr_3em04_clip_0.2",
        ),
        help="run one variant only; useful for manually resuming the grid",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="select from an existing complete sensitivity_summary.csv",
    )
    args = parser.parse_args(argv)
    if args.finalize and args.configuration is not None:
        parser.error("--finalize cannot be combined with --configuration")
    if args.timesteps < ROLLOUT_SIZE:
        parser.error(f"--timesteps must be at least one rollout ({ROLLOUT_SIZE})")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(learning_rate: float, clip_range: float) -> str:
    return f"lr_{learning_rate:.0e}_clip_{clip_range:g}".replace("-", "m")


CONFIGURATION_LABELS = {
    _slug(learning_rate, clip_range): (learning_rate, clip_range)
    for learning_rate, clip_range in CONFIGURATIONS
}


# Train and evaluate one learning-rate/clip-range variant at a time.
def _make_training_env(
    normalizer_path: Path,
    seed: int,
    monitor_dir: Path,
) -> VecEnv:
    def make_env() -> gym.Env:
        return AntDamageWrapper(make_ant_env(), mode="random")

    raw_env = make_vec_env(
        make_env,
        n_envs=NUM_ENVS,
        seed=seed,
        monitor_dir=str(monitor_dir),
    )
    normalized_env = VecNormalize.load(str(normalizer_path), raw_env)
    normalized_env.training = True
    normalized_env.norm_reward = True
    return VecCheckNan(normalized_env, raise_exception=True, check_inf=True)


def _set_hyperparameters(model: PPO, learning_rate: float, clip_range: float) -> None:
    """Replace only the two sensitivity parameters on a loaded PPO model."""
    model.learning_rate = learning_rate
    model.lr_schedule = lambda _progress: learning_rate
    for parameter_group in model.policy.optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    model.clip_range = lambda _progress: clip_range


class ProgressCallback(BaseCallback):
    """Print one human-readable update while a variant is training."""

    def __init__(self, label: str, start_steps: int, additional_steps: int) -> None:
        super().__init__()
        self.label = label
        self.start_steps = start_steps
        self.additional_steps = additional_steps
        self.started_at = 0.0
        self.next_update = 0.0

    def _on_training_start(self) -> None:
        self.started_at = time.perf_counter()
        self.next_update = self.started_at + 300.0

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if now < self.next_update:
            return True
        completed = max(self.num_timesteps - self.start_steps, 0)
        elapsed = max(now - self.started_at, 1e-9)
        rate = completed / elapsed
        remaining = max(self.additional_steps - completed, 0)
        eta = remaining / rate if rate else math.inf
        print(
            f"[Sensitivity progress] {self.label}: "
            f"{completed:,}/{self.additional_steps:,} additional steps; "
            f"ETA {eta / 60:.1f} min"
        )
        self.next_update = now + 300.0
        return True


def _run_evaluation(
    model_path: Path,
    normalizer_path: Path,
    output_csv: Path,
    evaluation_seed: int,
    episodes: int,
) -> None:
    conditions = [("healthy", 1.0), *[(leg, 0.5) for leg in LEG_ACTION_INDICES]]
    for index, (leg, alpha) in enumerate(conditions):
        command = [
            sys.executable,
            "-m",
            "damage_robust_ant.evaluate",
            "--model",
            str(model_path),
            "--normalizer",
            str(normalizer_path),
            "--training-condition",
            "robust",
            "--training-seed",
            "6",
            "--damage-leg",
            leg,
            "--alpha",
            str(alpha),
            "--evaluation-seed",
            str(evaluation_seed),
            "--episodes",
            str(episodes),
            "--output-csv",
            str(output_csv),
        ]
        if index:
            command.append("--append")
        subprocess.run(command, check=True)


def _validate_evaluation(path: Path, episodes: int, evaluation_seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected_cases = {("healthy", 1.0)} | {
        (leg, 0.5) for leg in LEG_ACTION_INDICES
    }
    actual_cases = set(zip(frame["damage_leg"], frame["damage_alpha"]))
    numeric_columns = [
        column
        for column in CSV_COLUMNS
        if column not in {"policy_training_condition", "damage_leg", "terminated_before_time_limit"}
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    keys = [
        "policy_training_condition",
        "policy_training_seed",
        "evaluation_seed",
        "damage_leg",
        "damage_alpha",
    ]
    if (
        list(frame.columns) != CSV_COLUMNS
        or len(frame) != 5 * episodes
        or frame.duplicated(keys).any()
        or frame.isna().any().any()
        or actual_cases != expected_cases
        or not np.isfinite(numeric.to_numpy()).all()
        or set(frame["policy_training_condition"]) != {"robust"}
        or set(frame["policy_training_seed"]) != {6}
    ):
        raise RuntimeError(f"invalid sensitivity evaluation: {path}")
    expected_seeds = set(range(evaluation_seed, evaluation_seed + episodes))
    for _, group in frame.groupby(["damage_leg", "damage_alpha"]):
        if set(group["evaluation_seed"]) != expected_seeds or len(group) != episodes:
            raise RuntimeError(f"incomplete sensitivity evaluation: {path}")
    return frame


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    moderate = frame[(frame["damage_leg"] != "healthy") & (frame["damage_alpha"] == 0.5)]
    healthy = frame[(frame["damage_leg"] == "healthy") & (frame["damage_alpha"] == 1.0)]
    return {
        "moderate_mean_forward_distance": float(moderate["forward_distance"].mean()),
        "moderate_mean_forward_speed": float(moderate["mean_forward_speed"].mean()),
        "moderate_early_termination_rate": float(moderate["terminated_before_time_limit"].mean()),
        "healthy_mean_forward_distance": float(healthy["forward_distance"].mean()),
        "healthy_mean_forward_speed": float(healthy["mean_forward_speed"].mean()),
    }


def _ranking_key(record: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return (
        record["moderate_mean_forward_distance"],
        -record["moderate_early_termination_rate"],
        record["moderate_mean_forward_speed"],
        record["healthy_mean_forward_distance"],
        -record["learning_rate"],
        -record["clip_range"],
    )


# Record enough provenance to compare all four variants later.
def _package_versions() -> dict[str, str]:
    distributions = {
        "project": "damage-robust-ant",
        "gymnasium": "gymnasium",
        "mujoco": "mujoco",
        "stable_baselines3": "stable-baselines3",
        "torch": "torch",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "tensorboard": "tensorboard",
    }
    values = {name: version(distribution) for name, distribution in distributions.items()}
    values["python"] = platform.python_version()
    return values


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _load_summary_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    required = {
        "configuration",
        "learning_rate",
        "clip_range",
        "moderate_mean_forward_distance",
        "moderate_mean_forward_speed",
        "moderate_early_termination_rate",
        "healthy_mean_forward_distance",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(f"existing sensitivity summary is incomplete: {path}")
    records = frame.to_dict(orient="records")
    for record in records:
        label = str(record["configuration"])
        if label not in CONFIGURATION_LABELS:
            raise RuntimeError(f"unknown sensitivity configuration in {path}: {label}")
        for field in (
            "learning_rate",
            "clip_range",
            "moderate_mean_forward_distance",
            "moderate_mean_forward_speed",
            "moderate_early_termination_rate",
            "healthy_mean_forward_distance",
        ):
            record[field] = float(record[field])
    if len({str(record["configuration"]) for record in records}) != len(records):
        raise RuntimeError(f"existing sensitivity summary contains duplicate variants: {path}")
    return records


def _freeze_configuration(
    output_dir: Path,
    source_model: Path,
    source_metadata: dict[str, Any],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    git_commit: str,
    clean: bool,
) -> Path:
    if len(records) != len(CONFIGURATIONS):
        raise RuntimeError(
            f"need all four sensitivity variants before finalizing (found {len(records)})"
        )
    expected_labels = set(CONFIGURATION_LABELS)
    if {str(record["configuration"]) for record in records} != expected_labels:
        raise RuntimeError("sensitivity summary does not contain the complete four-cell grid")
    selected = max(records, key=_ranking_key)
    summary_path = output_dir / "sensitivity_summary.csv"
    final_config_path = Path(__file__).resolve().parents[2] / "results" / "final_configuration.json"
    if final_config_path.exists():
        raise FileExistsError(
            f"final configuration is already frozen: {final_config_path}"
        )
    summary = {
        "schema_version": 1,
        "study": "one-seed exploratory PPO sensitivity study",
        "source_model": str(source_model),
        "source_model_sha256": _sha256(source_model),
        "source_condition": "robust",
        "source_seed": 6,
        "initialization": "loaded_from_preserved_robust_seed_6_checkpoint",
        "training_steps_per_configuration": args.timesteps,
        "actual_additional_steps_per_configuration": max(
            int(record["actual_additional_steps"]) for record in records
        ),
        "num_envs": NUM_ENVS,
        "rollout_size_environment_steps": ROLLOUT_SIZE,
        "configurations": records,
        "selected_configuration": {
            "configuration": selected["configuration"],
            "learning_rate": selected["learning_rate"],
            "clip_range": selected["clip_range"],
            "selection_metrics": {
                name: selected[name]
                for name in (
                    "moderate_mean_forward_distance",
                    "moderate_early_termination_rate",
                    "moderate_mean_forward_speed",
                    "healthy_mean_forward_distance",
                )
            },
        },
        "selection_rule": [
            "higher mean forward distance across four alpha=0.5 legs",
            "lower early termination rate",
            "higher mean forward speed",
            "higher healthy forward distance",
            "lower learning rate",
            "lower clip range",
        ],
        "validation_seeds": list(
            range(args.evaluation_seed, args.evaluation_seed + args.episodes)
        ),
        "held_out_evaluation_seeds": list(range(500, 510)),
        "evaluation_conditions": [
            {"damage_leg": "healthy", "alpha": 1.0},
            *[{"damage_leg": leg, "alpha": 0.5} for leg in LEG_ACTION_INDICES],
        ],
        "ppo_base_configuration": {
            "policy": "MlpPolicy",
            "actor_critic_architecture": [256, 256],
            "activation": "Tanh",
            "num_envs": NUM_ENVS,
            "target_kl": source_metadata["ppo_configuration"]["target_kl"],
            "normalization": _normalization_configuration(),
        },
        "damage_distribution": _damage_configuration("robust"),
        "package_versions": _package_versions(),
        "git_commit": git_commit,
        "tracked_worktree_clean": clean,
        "selection_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sensitivity_summary": str(
            summary_path.relative_to(Path(__file__).resolve().parents[2])
        ),
    }
    _write_json(final_config_path, summary)
    print(f"Selected configuration: {selected['configuration']}")
    print(f"Frozen in: {final_config_path}")
    return final_config_path


def run_study(args: argparse.Namespace) -> Path:
    source_model = args.source_model.resolve()
    source_normalizer = args.normalizer.resolve()
    output_dir = args.output_dir.resolve()
    source_metadata_path = source_model.with_name("metadata.json")
    if not source_metadata_path.is_file():
        raise FileNotFoundError(f"source metadata does not exist: {source_metadata_path}")
    source_metadata = json.loads(source_metadata_path.read_text())
    if source_metadata.get("condition") != "robust" or source_metadata.get("seed") != 6:
        raise RuntimeError("source model must be the preserved robust seed-6 policy")
    if args.finalize:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"sensitivity output does not exist: {output_dir}")
        records = _load_summary_records(output_dir / "sensitivity_summary.csv")
        git_commit, clean = _git_provenance()
        return _freeze_configuration(
            output_dir,
            source_model,
            source_metadata,
            records,
            args,
            git_commit,
            clean,
        )
    # Match the successful final runs and avoid oversubscribing the CPU.
    torch.set_num_threads(1)
    if args.configuration is None:
        if output_dir.exists():
            raise FileExistsError(f"sensitivity output already exists: {output_dir}")
        output_dir.mkdir(parents=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sensitivity_summary.csv"
    final_config_path = Path(__file__).resolve().parents[2] / "results" / "final_configuration.json"
    git_commit, clean = _git_provenance()
    records: list[dict[str, Any]] = _load_summary_records(summary_path)
    configurations = (
        [CONFIGURATION_LABELS[args.configuration]]
        if args.configuration is not None
        else list(CONFIGURATIONS)
    )
    started_at = time.perf_counter()
    try:
        for index, (learning_rate, clip_range) in enumerate(configurations, start=1):
            label = _slug(learning_rate, clip_range)
            run_dir = output_dir / label
            if run_dir.exists():
                raise FileExistsError(f"sensitivity variant output already exists: {run_dir}")
            run_dir.mkdir()
            for name in ("checkpoints", "monitor", "tensorboard"):
                (run_dir / name).mkdir()
            evaluation_dir = run_dir / "evaluation"
            evaluation_dir.mkdir()
            print(
                f"[Sensitivity {index}/4] {label}: loading robust seed-6 model "
                f"and training {args.timesteps:,} additional steps"
            )
            env = _make_training_env(source_normalizer, 6, run_dir / "monitor")
            model: PPO | None = None
            try:
                model = PPO.load(
                    source_model,
                    env=env,
                    device="cpu",
                    force_reset=True,
                    tensorboard_log=str(run_dir / "tensorboard"),
                )
                source_steps = int(model.num_timesteps)
                _set_hyperparameters(model, learning_rate, clip_range)
                target_additional = math.ceil(args.timesteps / ROLLOUT_SIZE) * ROLLOUT_SIZE
                callback = CheckpointCallback(
                    save_freq=max(100_000 // NUM_ENVS, 1),
                    save_path=str(run_dir / "checkpoints"),
                    name_prefix="ppo",
                    save_vecnormalize=True,
                )
                progress = ProgressCallback(label, source_steps, target_additional)
                train_started = time.perf_counter()
                model.learn(
                    total_timesteps=target_additional,
                    callback=[callback, progress],
                    log_interval=1,
                    tb_log_name="PPO",
                    reset_num_timesteps=False,
                )
                elapsed = time.perf_counter() - train_started
                actual_additional = int(model.num_timesteps) - source_steps
                if actual_additional != target_additional:
                    raise RuntimeError("sensitivity training stopped off rollout boundary")
                model.logger.dump(model.num_timesteps)
                model.logger.close()
                _assert_finite_model(model)
                model_path = run_dir / "final_model.zip"
                normalizer_path = run_dir / "vecnormalize.pkl"
                model.save(model_path)
                normalized = model.get_vec_normalize_env()
                if normalized is None:
                    raise RuntimeError("sensitivity environment has no normalization state")
                normalized.save(str(normalizer_path))
            finally:
                if model is not None and hasattr(model, "_logger"):
                    model.logger.close()
                env.close()

            evaluation_csv = evaluation_dir / "episode_results.csv"
            _run_evaluation(
                model_path,
                normalizer_path,
                evaluation_csv,
                args.evaluation_seed,
                args.episodes,
            )
            evaluated = _validate_evaluation(evaluation_csv, args.episodes, args.evaluation_seed)
            metrics = _metrics(evaluated)
            record: dict[str, Any] = {
                "configuration": label,
                "learning_rate": learning_rate,
                "clip_range": clip_range,
                "source_model": str(source_model),
                "source_model_sha256": _sha256(source_model),
                "source_environment_steps": source_steps,
                "requested_additional_steps": args.timesteps,
                "actual_additional_steps": actual_additional,
                "cumulative_environment_steps": source_steps + actual_additional,
                "elapsed_training_seconds": elapsed,
                "training_fps": actual_additional / elapsed,
                "evaluation_csv": str(evaluation_csv.relative_to(output_dir)),
                **metrics,
            }
            records.append(record)
            pd.DataFrame(records).to_csv(summary_path, index=False)
            _write_json(
                run_dir / "metadata.json",
                {
                    "schema_version": 1,
                    "study": "one-seed exploratory PPO sensitivity study",
                    "configuration": label,
                    "learning_rate": learning_rate,
                    "clip_range": clip_range,
                    "source_model": str(source_model),
                    "source_model_sha256": record["source_model_sha256"],
                    "source_seed": 6,
                    "source_environment_steps": source_steps,
                    "requested_additional_steps": args.timesteps,
                    "actual_additional_steps": actual_additional,
                    "cumulative_environment_steps": source_steps + actual_additional,
                    "num_envs": NUM_ENVS,
                    "rollout_size_environment_steps": ROLLOUT_SIZE,
                    "evaluation_seeds": list(
                        range(args.evaluation_seed, args.evaluation_seed + args.episodes)
                    ),
                    "evaluation_conditions": [
                        {"damage_leg": "healthy", "alpha": 1.0},
                        *[
                            {"damage_leg": leg, "alpha": 0.5}
                            for leg in LEG_ACTION_INDICES
                        ],
                    ],
                    "ppo_base_configuration": source_metadata["ppo_configuration"],
                    "normalization_configuration": _normalization_configuration(),
                    "damage_configuration": _damage_configuration("robust"),
                    "metrics": metrics,
                    "outputs": {
                        "model": model_path.name,
                        "normalizer": normalizer_path.name,
                        "evaluation": str(evaluation_csv.relative_to(run_dir)),
                    },
                    "package_versions": _package_versions(),
                    "git_commit": git_commit,
                    "tracked_worktree_clean": clean,
                },
            )
            print(
                f"[Sensitivity {index}/4] {label}: "
                f"moderate distance={metrics['moderate_mean_forward_distance']:.3f} m, "
                f"early={metrics['moderate_early_termination_rate']:.3f}"
            )

        if args.configuration is not None:
            print(
                f"Completed {args.configuration}. "
                "Run the remaining variants with --configuration, then use --finalize."
            )
            return summary_path

        return _freeze_configuration(
            output_dir,
            source_model,
            source_metadata,
            records,
            args,
            git_commit,
            clean,
        )
    except BaseException:
        # Keep the partial output directory for inspection; never silently reuse it.
        raise


def main() -> None:
    run_study(parse_args())


if __name__ == "__main__":
    main()
