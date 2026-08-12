"""Open an interactive Ant viewer for manual environment checks."""

import argparse
import time

import gymnasium as gym


def parse_args() -> argparse.Namespace:
    """Parse viewer command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--speed", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """Run Ant with smooth random actions in the MuJoCo viewer."""
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if not 0.0 <= args.strength <= 1.0:
        raise SystemExit("--strength must be between 0 and 1")
    if args.speed <= 0.0:
        raise SystemExit("--speed must be greater than 0")

    env = gym.make("Ant-v5", render_mode="human")
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    action = env.action_space.sample()
    action.fill(0.0)
    target = action.copy()

    try:
        for step in range(args.steps):
            if step % 40 == 0:
                target = args.strength * env.action_space.sample()

            action += 0.05 * (target - action)
            _, _, terminated, truncated, _ = env.step(action)
            time.sleep(env.unwrapped.dt / args.speed)

            if terminated or truncated:
                env.reset()
                action.fill(0.0)
                target.fill(0.0)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
