"""Open an interactive Ant viewer for manual environment checks."""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from damage_robust_ant.damage import AntDamageWrapper, LEG_ACTION_INDICES, make_ant_env
from damage_robust_ant.evaluate import (
    _load_observation_normalizer,
    _normalize_observation,
    _resolve_normalizer_path,
)


DAMAGE_COLOR = (0.9, 0.1, 0.1, 1.0)
START_COLOR = np.array([0.1, 0.8, 0.1, 0.75])
GOAL_COLOR = np.array([0.1, 0.3, 1.0, 0.75])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse viewer command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument(
        "--goal-distance",
        type=float,
        default=5.0,
        help="forward distance target shown during manual viewing",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="viewer steps between terminal progress updates",
    )
    parser.add_argument(
        "--stop-at-goal",
        action="store_true",
        help="stop the viewer once the forward-distance target is reached",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="saved PPO model to use instead of smooth random actions",
    )
    parser.add_argument(
        "--normalizer",
        type=Path,
        help="saved VecNormalize state; defaults to vecnormalize.pkl beside the model",
    )
    parser.add_argument(
        "--damage-mode",
        choices=("nominal", "random", "fixed"),
        default="nominal",
    )
    parser.add_argument(
        "--leg",
        choices=tuple(LEG_ACTION_INDICES),
        default="front_left",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args(argv)


def _highlight_damage(
    env: AntDamageWrapper,
    damage_leg: str | None,
    original_colors: np.ndarray,
) -> None:
    """Tint the selected leg red and restore all other geometry colors."""
    model = env.unwrapped.model
    model.geom_rgba[:] = original_colors
    if damage_leg is None:
        return

    hip_action_index = LEG_ACTION_INDICES[damage_leg][0]
    hip_joint_id = int(model.actuator_trnid[hip_action_index, 0])
    hip_body_id = int(model.jnt_bodyid[hip_joint_id])
    leg_root_body_id = int(model.body_parentid[hip_body_id])

    for geom_id, geom_body_id in enumerate(model.geom_bodyid):
        body_id = int(geom_body_id)
        while body_id != 0:
            if body_id == leg_root_body_id:
                model.geom_rgba[geom_id] = DAMAGE_COLOR
                break
            body_id = int(model.body_parentid[body_id])


def _add_position_markers(
    env: AntDamageWrapper,
    start_x: float,
    start_y: float,
    goal_distance: float,
) -> None:
    """Add simple start and goal markers to the MuJoCo viewer when available."""
    renderer = getattr(env.unwrapped, "mujoco_renderer", None)
    viewer = getattr(renderer, "viewer", None)
    if viewer is None:
        return

    z = 0.03
    marker_size = np.array([0.18, 0.18, 0.03])
    for x_position, color in (
        (start_x, START_COLOR),
        (start_x + goal_distance, GOAL_COLOR),
    ):
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=marker_size,
            pos=np.array([x_position, start_y, z]),
            rgba=color,
        )


def _print_progress(
    step: int,
    start_x: float,
    current_x: float,
    goal_distance: float,
) -> None:
    """Print manual progress in meters from the episode start."""
    progress = current_x - start_x
    print(
        f"step={step} forward_distance={progress:.3f}m "
        f"goal={goal_distance:.3f}m"
    )


def main() -> None:
    """Run Ant with smooth random actions or a saved PPO policy."""
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if not 0.0 <= args.strength <= 1.0:
        raise SystemExit("--strength must be between 0 and 1")
    if args.speed <= 0.0:
        raise SystemExit("--speed must be greater than 0")
    if args.goal_distance <= 0.0:
        raise SystemExit("--goal-distance must be greater than 0")
    if args.progress_interval < 1:
        raise SystemExit("--progress-interval must be at least 1")
    if args.damage_mode == "fixed" and not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1")
    if args.normalizer is not None and args.model is None:
        raise SystemExit("--normalizer requires --model")
    if args.model is not None and not args.model.is_file():
        raise SystemExit(f"model does not exist: {args.model}")
    if args.normalizer is not None and not args.normalizer.is_file():
        raise SystemExit(f"normalizer does not exist: {args.normalizer}")

    policy = PPO.load(args.model, device="auto") if args.model else None
    base_env = make_ant_env(render_mode="human")
    if args.damage_mode == "fixed":
        env = AntDamageWrapper(
            base_env,
            mode="fixed",
            fixed_leg=args.leg,
            fixed_alpha=args.alpha,
        )
    else:
        env = AntDamageWrapper(base_env, mode=args.damage_mode)

    normalizer: VecNormalize | None = None
    if policy is not None:
        normalizer_path = _resolve_normalizer_path(args.model, args.normalizer)
        if normalizer_path is not None:
            normalizer = _load_observation_normalizer(normalizer_path, env)

    original_colors = env.unwrapped.model.geom_rgba.copy()
    observation, info = env.reset(seed=args.seed)
    start_x = float(info["x_position"])
    start_y = float(info["y_position"])
    goal_reached = False
    env.action_space.seed(args.seed)
    _highlight_damage(env, info["damage_leg"], original_colors)
    env.render()
    _add_position_markers(env, start_x, start_y, args.goal_distance)
    env.render()
    print(
        f"damage_leg={info['damage_leg']} "
        f"damage_alpha={info['damage_alpha']:.3f}"
    )
    print(f"controller={'smooth_random' if policy is None else args.model}")
    print(
        f"start_x={start_x:.3f} goal_x={start_x + args.goal_distance:.3f} "
        f"goal_distance={args.goal_distance:.3f}m"
    )

    action = env.action_space.sample()
    action.fill(0.0)
    target = action.copy()

    try:
        for step in range(args.steps):
            if policy is None:
                if step % 40 == 0:
                    target = args.strength * env.action_space.sample()

                action += 0.05 * (target - action)
            else:
                policy_observation = _normalize_observation(
                    observation,
                    normalizer,
                )
                action, _ = policy.predict(
                    policy_observation,
                    deterministic=True,
                )

            _add_position_markers(env, start_x, start_y, args.goal_distance)
            observation, _, terminated, truncated, step_info = env.step(action)
            time.sleep(env.unwrapped.dt / args.speed)
            current_x = float(step_info["x_position"])
            progress = current_x - start_x
            if step % args.progress_interval == 0:
                _print_progress(step + 1, start_x, current_x, args.goal_distance)
            if not goal_reached and progress >= args.goal_distance:
                goal_reached = True
                print(
                    f"goal reached at step={step + 1} "
                    f"forward_distance={progress:.3f}m"
                )
                if args.stop_at_goal:
                    break

            if terminated or truncated:
                observation, info = env.reset()
                start_x = float(info["x_position"])
                start_y = float(info["y_position"])
                goal_reached = False
                _highlight_damage(env, info["damage_leg"], original_colors)
                env.render()
                _add_position_markers(env, start_x, start_y, args.goal_distance)
                env.render()
                print(
                    f"damage_leg={info['damage_leg']} "
                    f"damage_alpha={info['damage_alpha']:.3f}"
                )
                print(
                    f"start_x={start_x:.3f} "
                    f"goal_x={start_x + args.goal_distance:.3f} "
                    f"goal_distance={args.goal_distance:.3f}m"
                )
                action.fill(0.0)
                target.fill(0.0)
    except KeyboardInterrupt:
        pass
    finally:
        if normalizer is not None:
            normalizer.close()
        else:
            env.close()


if __name__ == "__main__":
    main()
