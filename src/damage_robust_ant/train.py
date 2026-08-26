"""Train a nominal or damage-robust PPO policy for Ant-v5."""

import argparse
import json
import math
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import VecCheckNan, VecEnv, VecNormalize

from damage_robust_ant.damage import (
    ANT_ENV_ID,
    ANT_HEALTHY_REWARD,
    AntDamageWrapper,
    LEG_ACTION_INDICES,
    make_ant_env,
)


PROGRESS_INTERVAL_SECONDS = 300.0
TARGET_KL = 0.02
NORMALIZATION_CLIP = 10.0
FINAL_LEARNING_RATE = 1e-4
PPO_EPOCHS = 10
TORCH_THREADS = 1


def _format_duration(seconds: float) -> str:
    total_minutes = max(round(seconds / 60), 0)
    hours, minutes = divmod(total_minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


class TrainingProgressCallback(BaseCallback):
    """Print a plain-language progress summary at a fixed time interval."""

    def __init__(
        self,
        requested_steps: int,
        run_index: int,
        total_runs: int,
        interval_seconds: float = PROGRESS_INTERVAL_SECONDS,
    ) -> None:
        super().__init__()
        self.requested_steps = requested_steps
        self.run_index = run_index
        self.total_runs = total_runs
        self.interval_seconds = interval_seconds
        self.started_at = 0.0
        self.next_update_at = 0.0

    def _on_training_start(self) -> None:
        print(
            "[Progress] Plain-language training summaries will appear "
            "about every five minutes."
        )
        self.started_at = time.perf_counter()
        self.next_update_at = self.started_at + self.interval_seconds

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if now < self.next_update_at:
            return True

        current_steps = min(self.num_timesteps, self.requested_steps)
        elapsed_seconds = now - self.started_at
        remaining_steps = (
            self.requested_steps - current_steps
            + (self.total_runs - self.run_index) * self.requested_steps
        )
        steps_per_second = current_steps / elapsed_seconds
        eta = _format_duration(remaining_steps / steps_per_second)
        percent = 100 * current_steps / self.requested_steps
        runs_afterward = self.total_runs - self.run_index
        print(
            f"[Progress] Training {self.run_index}/{self.total_runs}: "
            f"{current_steps:,}/{self.requested_steps:,} steps "
            f"({percent:.1f}%); {runs_afterward} full runs remain afterward; "
            f"rough training ETA {eta}."
        )
        self.next_update_at = now + self.interval_seconds
        return True


def _positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed_value


def _positive_float(value: str) -> float:
    parsed_value = float(value)
    if parsed_value <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed_value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse training command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("nominal", "robust"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=_positive_int, default=1_000_000)
    parser.add_argument("--num-envs", type=_positive_int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=_positive_float, default=3e-4)
    parser.add_argument("--clip-range", type=_positive_float, default=0.2)
    parser.add_argument(
        "--run-index",
        type=_positive_int,
        default=1,
        help="current run number for progress reporting",
    )
    parser.add_argument(
        "--total-runs",
        type=_positive_int,
        default=1,
        help="total run count for progress reporting",
    )
    args = parser.parse_args(argv)
    if args.run_index > args.total_runs:
        parser.error("--run-index cannot exceed --total-runs")
    return args


def make_training_env(
    condition: str,
    num_envs: int,
    seed: int,
    monitor_dir: Path,
    *,
    normalize: bool = False,
) -> VecEnv:
    """Create monitored vectorized Ant environments for one condition."""
    damage_mode = "nominal" if condition == "nominal" else "random"

    def make_env() -> gym.Env:
        return AntDamageWrapper(make_ant_env(), mode=damage_mode)

    vector_env = make_vec_env(
        make_env,
        n_envs=num_envs,
        seed=seed,
        monitor_dir=str(monitor_dir),
    )
    if normalize:
        vector_env = VecNormalize(
            vector_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=NORMALIZATION_CLIP,
            clip_reward=NORMALIZATION_CLIP,
            gamma=0.99,
        )
    return VecCheckNan(vector_env, raise_exception=True, check_inf=True)


def load_normalized_env(
    normalizer_path: Path,
    condition: str,
    num_envs: int,
    seed: int,
) -> VecEnv:
    """Load frozen normalization statistics around fresh Ant environments."""
    damage_mode = "nominal" if condition == "nominal" else "random"

    def make_env() -> gym.Env:
        return AntDamageWrapper(make_ant_env(), mode=damage_mode)

    vector_env = make_vec_env(make_env, n_envs=num_envs, seed=seed)
    normalized_env = VecNormalize.load(str(normalizer_path), vector_env)
    normalized_env.training = False
    normalized_env.norm_reward = False
    return VecCheckNan(normalized_env, raise_exception=True, check_inf=True)


def linear_schedule(
    initial_value: float,
    final_value: float = 0.0,
) -> Callable[[float], float]:
    """Linearly decay a scalar from its initial to final value."""

    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return schedule


def make_ppo(
    env: VecEnv,
    learning_rate: float,
    clip_range: float,
    seed: int,
    tensorboard_dir: Path,
    *,
    anneal_learning_rate: bool = False,
    final_learning_rate: float = 0.0,
    n_epochs: int = 10,
    target_kl: float | None = None,
) -> PPO:
    """Create PPO with the fixed project configuration."""
    policy_kwargs = {
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": torch.nn.Tanh,
        "ortho_init": True,
        "log_std_init": 0.0,
        "full_std": True,
        "use_expln": False,
        "squash_output": False,
        "features_extractor_class": FlattenExtractor,
        "features_extractor_kwargs": {},
        "share_features_extractor": True,
        "normalize_images": True,
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"eps": 1e-5},
    }
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=(
            linear_schedule(learning_rate, final_learning_rate)
            if anneal_learning_rate
            else learning_rate
        ),
        n_steps=2_048,
        batch_size=64,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        rollout_buffer_class=RolloutBuffer,
        rollout_buffer_kwargs={},
        target_kl=target_kl,
        stats_window_size=100,
        tensorboard_log=str(tensorboard_dir),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device="auto",
    )


def _prepare_output_dir(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "run": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "monitor": output_dir / "monitor",
        "tensorboard": output_dir / "tensorboard",
        "model": output_dir / "final_model.zip",
        "normalizer": output_dir / "vecnormalize.pkl",
        "metadata": output_dir / "metadata.json",
    }
    for name in ("checkpoints", "monitor", "tensorboard"):
        paths[name].mkdir()
    return paths


def _checkpoint_settings(timesteps: int, num_envs: int) -> dict[str, int]:
    requested_interval = min(100_000, timesteps)
    callback_frequency = max(math.ceil(requested_interval / num_envs), 1)
    return {
        "requested_interval_environment_steps": requested_interval,
        "callback_frequency": callback_frequency,
        "effective_interval_environment_steps": callback_frequency * num_envs,
    }


def _ppo_configuration(
    model: PPO,
    learning_rate: float,
    clip_range: float,
    seed: int,
    *,
    anneal_learning_rate: bool = False,
    final_learning_rate: float = 0.0,
) -> dict[str, object]:
    policy = model.policy
    return {
        "policy": "MlpPolicy",
        "policy_class": type(policy).__name__,
        "policy_kwargs": {
            "net_arch": policy.net_arch,
            "activation_fn": policy.activation_fn.__name__,
            "ortho_init": policy.ortho_init,
            "log_std_init": policy.log_std_init,
            "full_std": True,
            "use_expln": False,
            "squash_output": policy.squash_output,
            "features_extractor_class": type(policy.features_extractor).__name__,
            "features_extractor_kwargs": policy.features_extractor_kwargs,
            "share_features_extractor": policy.share_features_extractor,
            "normalize_images": policy.normalize_images,
            "optimizer_class": type(policy.optimizer).__name__,
            "optimizer_defaults": policy.optimizer.defaults,
        },
        "learning_rate": learning_rate,
        "final_learning_rate": (
            final_learning_rate if anneal_learning_rate else learning_rate
        ),
        "learning_rate_schedule": (
            "linear" if anneal_learning_rate else "constant"
        ),
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "clip_range": clip_range,
        "clip_range_schedule": "constant",
        "clip_range_vf": model.clip_range_vf,
        "normalize_advantage": model.normalize_advantage,
        "ent_coef": model.ent_coef,
        "vf_coef": model.vf_coef,
        "max_grad_norm": model.max_grad_norm,
        "use_sde": model.use_sde,
        "sde_sample_freq": model.sde_sample_freq,
        "rollout_buffer_class": model.rollout_buffer_class.__name__,
        "rollout_buffer_kwargs": model.rollout_buffer_kwargs,
        "target_kl": model.target_kl,
        "stats_window_size": model._stats_window_size,
        "seed": seed,
        "device_requested": "auto",
        "device_resolved": str(model.device),
        "reset_num_timesteps": True,
    }


def _normalization_configuration() -> dict[str, object]:
    return {
        "normalize_observations": True,
        "normalize_rewards_during_training": True,
        "normalize_rewards_during_evaluation": False,
        "clip_observations": NORMALIZATION_CLIP,
        "clip_rewards": NORMALIZATION_CLIP,
        "gamma": 0.99,
    }


def _environment_configuration() -> dict[str, object]:
    return {
        "environment_id": ANT_ENV_ID,
        "healthy_reward": ANT_HEALTHY_REWARD,
        "terminate_when_unhealthy": True,
        "walking_priority": "higher healthy reward makes early falls more costly",
    }


def _damage_configuration(condition: str) -> dict[str, object]:
    robust = condition == "robust"
    return {
        "mode": "random" if robust else "nominal",
        "sample_timing": "episode_reset",
        "fixed_within_episode": True,
        "healthy_probability": 0.25 if robust else 1.0,
        "leg_selection": "uniform" if robust else None,
        "legs": list(LEG_ACTION_INDICES) if robust else [],
        "leg_action_indices": {
            leg: list(indices) for leg, indices in LEG_ACTION_INDICES.items()
        },
        "alpha_distribution": "uniform" if robust else "fixed",
        "alpha_range": [0.25, 1.0] if robust else [1.0, 1.0],
        "actuators_scaled_per_damaged_leg": 2,
    }


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
    versions = {name: version(distribution) for name, distribution in distributions.items()}
    versions["python"] = platform.python_version()
    return versions


def _git_provenance() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, not status.strip()


def _assert_finite_model(model: PPO) -> None:
    for name, value in model.policy.state_dict().items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite policy tensor: {name}")
    for parameter_state in model.policy.optimizer.state.values():
        for name, value in parameter_state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise RuntimeError(f"non-finite optimizer tensor: {name}")


def run_training(args: argparse.Namespace) -> Path:
    """Train, save, and verify one PPO policy."""
    torch.set_num_threads(TORCH_THREADS)
    git_commit, tracked_worktree_clean = _git_provenance()
    paths = _prepare_output_dir(args.output_dir)
    checkpoint_settings = _checkpoint_settings(args.timesteps, args.num_envs)
    env: VecEnv | None = None
    model: PPO | None = None

    print(
        f"Training condition={args.condition} seed={args.seed} "
        f"requested_steps={args.timesteps} num_envs={args.num_envs}"
    )
    try:
        env = make_training_env(
            args.condition,
            args.num_envs,
            args.seed,
            paths["monitor"],
            normalize=True,
        )
        model = make_ppo(
            env,
            args.learning_rate,
            args.clip_range,
            args.seed,
            paths["tensorboard"],
            anneal_learning_rate=True,
            final_learning_rate=FINAL_LEARNING_RATE,
            n_epochs=PPO_EPOCHS,
            target_kl=TARGET_KL,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_settings["callback_frequency"],
            save_path=str(paths["checkpoints"]),
            name_prefix="ppo",
            save_vecnormalize=True,
        )
        progress_callback = TrainingProgressCallback(
            requested_steps=args.timesteps,
            run_index=args.run_index,
            total_runs=args.total_runs,
        )

        started_at = time.perf_counter()
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList([checkpoint_callback, progress_callback]),
            log_interval=1,
            tb_log_name="PPO",
            reset_num_timesteps=True,
        )
        elapsed_seconds = time.perf_counter() - started_at
        actual_timesteps = model.num_timesteps
        model.logger.dump(actual_timesteps)
        model.logger.close()

        _assert_finite_model(model)
        model.save(paths["model"])
        normalizer = model.get_vec_normalize_env()
        if normalizer is None:
            raise RuntimeError("training environment has no normalization state")
        normalizer.save(str(paths["normalizer"]))

        verification_env = load_normalized_env(
            paths["normalizer"],
            args.condition,
            args.num_envs,
            args.seed,
        )
        try:
            reloaded_model = PPO.load(
                paths["model"],
                env=verification_env,
                device="auto",
            )
            if reloaded_model.num_timesteps != actual_timesteps:
                raise RuntimeError("reloaded model has an unexpected timestep count")
            _assert_finite_model(reloaded_model)
            observation = verification_env.reset()
            action, _ = reloaded_model.predict(observation, deterministic=True)
            if not np.isfinite(action).all():
                raise RuntimeError("reloaded model produced a non-finite action")
        finally:
            verification_env.close()

        metadata = {
            "schema_version": 2,
            "condition": args.condition,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "requested_environment_steps": args.timesteps,
            "actual_environment_steps": actual_timesteps,
            "ppo_configuration": _ppo_configuration(
                model,
                args.learning_rate,
                args.clip_range,
                args.seed,
                anneal_learning_rate=True,
                final_learning_rate=FINAL_LEARNING_RATE,
            ),
            "normalization_configuration": _normalization_configuration(),
            "environment_configuration": _environment_configuration(),
            "damage_configuration": _damage_configuration(args.condition),
            "checkpoint_configuration": checkpoint_settings,
            "package_versions": _package_versions(),
            "elapsed_training_seconds": elapsed_seconds,
            "training_fps": actual_timesteps / elapsed_seconds,
            "torch_threads": torch.get_num_threads(),
            "git_commit": git_commit,
            "tracked_worktree_clean": tracked_worktree_clean,
            "validation": {
                "finite_model_parameters": True,
                "final_model_reloaded": True,
                "normalization_state_reloaded": True,
                "finite_reloaded_prediction": True,
            },
            "outputs": {
                "final_model": paths["model"].name,
                "normalizer": paths["normalizer"].name,
                "checkpoints": paths["checkpoints"].name,
                "monitor": paths["monitor"].name,
                "tensorboard": paths["tensorboard"].name,
            },
        }
        temporary_metadata = paths["metadata"].with_suffix(".json.tmp")
        temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary_metadata.replace(paths["metadata"])
    finally:
        if model is not None and hasattr(model, "_logger"):
            model.logger.close()
        if env is not None:
            env.close()

    print(
        f"Finished actual_steps={actual_timesteps} "
        f"elapsed={elapsed_seconds:.1f}s fps={actual_timesteps / elapsed_seconds:.0f}"
    )
    print(f"Model: {paths['model']}")
    return paths["model"]


def main() -> None:
    """Run training from the command line."""
    run_training(parse_args())


if __name__ == "__main__":
    main()
