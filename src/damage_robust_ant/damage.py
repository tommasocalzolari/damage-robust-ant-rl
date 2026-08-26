"""Episode-level actuator degradation for Gymnasium's Ant-v5."""

from typing import Any

import gymnasium as gym
import numpy as np


ANT_ENV_ID = "Ant-v5"
ANT_HEALTHY_REWARD = 3.0

# Ant-v5's actuators are unnamed. The mapping combines their target joints with
# the body directions documented by Gymnasium.
LEG_ACTION_INDICES = {
    "front_left": (2, 3),  # hip_1, ankle_1
    "front_right": (4, 5),  # hip_2, ankle_2
    "back_left": (6, 7),  # hip_3, ankle_3
    "back_right": (0, 1),  # hip_4, ankle_4
}


def make_ant_env(**kwargs: Any) -> gym.Env:
    """Create the Ant environment used by the project."""
    return gym.make(
        ANT_ENV_ID,
        healthy_reward=ANT_HEALTHY_REWARD,
        **kwargs,
    )


class AntDamageWrapper(gym.ActionWrapper):
    """Scale both actuator commands for one Ant leg during an episode."""

    def __init__(
        self,
        env: gym.Env,
        mode: str = "nominal",
        fixed_leg: str | None = None,
        fixed_alpha: float | None = None,
    ) -> None:
        super().__init__(env)
        if mode not in {"nominal", "random", "fixed"}:
            raise ValueError("mode must be 'nominal', 'random', or 'fixed'")
        if mode == "fixed":
            if fixed_leg not in LEG_ACTION_INDICES:
                raise ValueError(f"fixed_leg must be one of {tuple(LEG_ACTION_INDICES)}")
            if fixed_alpha is None or not 0.0 <= fixed_alpha <= 1.0:
                raise ValueError("fixed_alpha must be between 0 and 1")
        elif fixed_leg is not None or fixed_alpha is not None:
            raise ValueError("fixed_leg and fixed_alpha are only valid in fixed mode")

        self.mode = mode
        self.fixed_leg = fixed_leg
        self.fixed_alpha = 1.0 if fixed_alpha is None else float(fixed_alpha)
        self.damage_leg: str | None = None
        self.damage_alpha = 1.0
        self.policy_action: np.ndarray | None = None
        self.applied_action: np.ndarray | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and choose the new episode's damage."""
        observation, info = super().reset(seed=seed, options=options)
        self.policy_action = None
        self.applied_action = None

        if self.mode == "nominal":
            self.damage_leg = None
            self.damage_alpha = 1.0
        elif self.mode == "random":
            self._sample_random_damage()
        else:
            self.damage_leg = self.fixed_leg
            self.damage_alpha = self.fixed_alpha

        return observation, self._add_damage_info(info)

    def action(self, action: np.ndarray) -> np.ndarray:
        """Return a copy of the action with the active damage applied."""
        self.policy_action = np.asarray(action).copy()
        damaged_action = self.policy_action.copy()
        if self.damage_leg is not None:
            indices = list(LEG_ACTION_INDICES[self.damage_leg])
            damaged_action[indices] *= self.damage_alpha
        self.applied_action = damaged_action.copy()
        return damaged_action

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Apply the action and include the active damage in the step info."""
        observation, reward, terminated, truncated, info = super().step(action)
        return (
            observation,
            reward,
            terminated,
            truncated,
            self._add_damage_info(info),
        )

    def _sample_random_damage(self) -> None:
        if self.np_random.random() < 0.25:
            self.damage_leg = None
            self.damage_alpha = 1.0
            return

        self.damage_leg = str(self.np_random.choice(tuple(LEG_ACTION_INDICES)))
        self.damage_alpha = float(self.np_random.uniform(0.25, 1.0))

    def _add_damage_info(self, info: dict[str, Any]) -> dict[str, Any]:
        damage_info = dict(info)
        damage_info["damage_leg"] = self.damage_leg
        damage_info["damage_alpha"] = self.damage_alpha
        return damage_info
