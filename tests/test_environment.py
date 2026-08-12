"""Smoke tests for the Ant simulation environment."""

import math

import gymnasium as gym


def test_ant_v5_can_reset_and_step() -> None:
    """Ant-v5 returns valid observations for a short interaction."""
    env = gym.make("Ant-v5")
    try:
        observation, reset_info = env.reset(seed=0)
        env.action_space.seed(0)

        assert env.observation_space.contains(observation)
        assert isinstance(reset_info, dict)

        action = env.action_space.sample()
        observation, reward, _terminated, _truncated, step_info = env.step(action)

        assert env.observation_space.contains(observation)
        assert math.isfinite(reward)
        assert isinstance(step_info, dict)
    finally:
        env.close()
