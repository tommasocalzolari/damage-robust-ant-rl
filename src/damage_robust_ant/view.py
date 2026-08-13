"""Open an interactive Ant viewer for manual environment checks."""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from damage_robust_ant.damage import AntDamageWrapper, LEG_ACTION_INDICES


DAMAGE_COLOR = (0.9, 0.1, 0.1, 1.0)


def parse_args() -> argparse.Namespace:
    """Parse viewer command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument(
        "--model",
        type=Path,
        help="saved PPO model to use instead of smooth random actions",
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
    return parser.parse_args()


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


def main() -> None:
    """Run Ant with smooth random actions or a saved PPO policy."""
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if not 0.0 <= args.strength <= 1.0:
        raise SystemExit("--strength must be between 0 and 1")
    if args.speed <= 0.0:
        raise SystemExit("--speed must be greater than 0")
    if args.damage_mode == "fixed" and not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1")

    policy = PPO.load(args.model, device="auto") if args.model else None
    base_env = gym.make("Ant-v5", render_mode="human")
    if args.damage_mode == "fixed":
        env = AntDamageWrapper(
            base_env,
            mode="fixed",
            fixed_leg=args.leg,
            fixed_alpha=args.alpha,
        )
    else:
        env = AntDamageWrapper(base_env, mode=args.damage_mode)

    original_colors = env.unwrapped.model.geom_rgba.copy()
    observation, info = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    _highlight_damage(env, info["damage_leg"], original_colors)
    env.render()
    print(
        f"damage_leg={info['damage_leg']} "
        f"damage_alpha={info['damage_alpha']:.3f}"
    )
    print(f"controller={'smooth_random' if policy is None else args.model}")

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
                action, _ = policy.predict(observation, deterministic=True)

            observation, _, terminated, truncated, _ = env.step(action)
            time.sleep(env.unwrapped.dt / args.speed)

            if terminated or truncated:
                observation, info = env.reset()
                _highlight_damage(env, info["damage_leg"], original_colors)
                env.render()
                print(
                    f"damage_leg={info['damage_leg']} "
                    f"damage_alpha={info['damage_alpha']:.3f}"
                )
                action.fill(0.0)
                target.fill(0.0)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
