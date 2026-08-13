"""Tests for Ant actuator degradation."""

from collections import Counter
from unittest.mock import Mock, patch

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from damage_robust_ant.damage import AntDamageWrapper, LEG_ACTION_INDICES
from damage_robust_ant.view import DAMAGE_COLOR, _highlight_damage


def test_leg_mapping_matches_ant_model() -> None:
    """The mapping follows the transmission joints loaded by Ant-v5."""
    env = gym.make("Ant-v5")
    try:
        model = env.unwrapped.model
        actuator_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(model.nu)
        ]
        joint_names = [
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(model.actuator_trnid[index, 0]),
            )
            for index in range(model.nu)
        ]

        assert actuator_names == [None] * 8
        assert env.action_space.shape == (8,)
        assert joint_names == [
            "hip_4",
            "ankle_4",
            "hip_1",
            "ankle_1",
            "hip_2",
            "ankle_2",
            "hip_3",
            "ankle_3",
        ]
        assert LEG_ACTION_INDICES == {
            "front_left": (2, 3),
            "front_right": (4, 5),
            "back_left": (6, 7),
            "back_right": (0, 1),
        }
    finally:
        env.close()


def test_nominal_mode_leaves_action_unchanged() -> None:
    """Nominal mode forwards actions without changing caller-owned data."""
    env = AntDamageWrapper(gym.make("Ant-v5"), mode="nominal")
    action = np.linspace(-0.8, 0.8, 8, dtype=np.float32)
    original_action = action.copy()
    try:
        _, reset_info = env.reset(seed=0)
        transformed_action = env.action(action)
        _, _, _, _, step_info = env.step(action)

        np.testing.assert_array_equal(transformed_action, original_action)
        np.testing.assert_array_equal(env.unwrapped.data.ctrl, original_action)
        np.testing.assert_array_equal(action, original_action)
        np.testing.assert_array_equal(env.policy_action, original_action)
        np.testing.assert_array_equal(env.applied_action, original_action)
        assert reset_info["damage_leg"] is None
        assert reset_info["damage_alpha"] == 1.0
        assert step_info["damage_leg"] is None
        assert step_info["damage_alpha"] == 1.0
        assert "reward_forward" in step_info
    finally:
        env.close()


@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_fixed_damage_scales_only_selected_leg(alpha: float) -> None:
    """Fixed damage scales exactly the selected pair of action values."""
    env = AntDamageWrapper(
        gym.make("Ant-v5"),
        mode="fixed",
        fixed_leg="front_right",
        fixed_alpha=alpha,
    )
    action = np.linspace(0.1, 0.8, 8, dtype=np.float32)
    original_action = action.copy()
    expected_action = action.copy()
    expected_action[[4, 5]] *= alpha
    try:
        env.reset(seed=0)
        transformed_action = env.action(action)
        env.step(action)

        np.testing.assert_array_equal(transformed_action, expected_action)
        np.testing.assert_array_equal(env.unwrapped.data.ctrl, expected_action)
        np.testing.assert_array_equal(action, original_action)
        np.testing.assert_array_equal(env.policy_action, original_action)
        np.testing.assert_array_equal(env.applied_action, expected_action)
    finally:
        env.close()


def test_random_damage_is_sampled_only_on_reset() -> None:
    """Random damage remains constant through every step of an episode."""
    env = AntDamageWrapper(gym.make("Ant-v5"), mode="random")
    action = np.zeros(8, dtype=np.float32)
    try:
        with patch.object(
            env,
            "_sample_random_damage",
            wraps=env._sample_random_damage,
        ) as sample_damage:
            _, reset_info = env.reset(seed=7)
            condition = (
                reset_info["damage_leg"],
                reset_info["damage_alpha"],
            )
            assert sample_damage.call_count == 1

            for _ in range(3):
                _, _, _, _, step_info = env.step(action)
                assert (
                    step_info["damage_leg"],
                    step_info["damage_alpha"],
                ) == condition
                assert sample_damage.call_count == 1

            env.reset()
            assert sample_damage.call_count == 2
    finally:
        env.close()


def test_random_damage_is_reproducible() -> None:
    """Equal reset seeds produce equal sequences of episode damage."""

    def damage_sequence(env: AntDamageWrapper) -> list[tuple[str | None, float]]:
        _, info = env.reset(seed=123)
        sequence = [(info["damage_leg"], info["damage_alpha"])]
        for _ in range(19):
            _, info = env.reset()
            sequence.append((info["damage_leg"], info["damage_alpha"]))
        return sequence

    first_env = AntDamageWrapper(gym.make("Ant-v5"), mode="random")
    second_env = AntDamageWrapper(gym.make("Ant-v5"), mode="random")
    try:
        assert damage_sequence(first_env) == damage_sequence(second_env)
    finally:
        first_env.close()
        second_env.close()


def test_random_damage_uses_required_sampling_rules() -> None:
    """Random mode uses the exact healthy threshold, leg set, and alpha range."""
    env = AntDamageWrapper(gym.make("Ant-v5"), mode="random")
    try:
        random_source = Mock()
        env.np_random = random_source

        random_source.random.return_value = 0.249
        env._sample_random_damage()
        assert env.damage_leg is None
        assert env.damage_alpha == 1.0
        random_source.choice.assert_not_called()
        random_source.uniform.assert_not_called()

        random_source.random.return_value = 0.25
        random_source.choice.return_value = "back_left"
        random_source.uniform.return_value = 0.6
        env._sample_random_damage()
        assert env.damage_leg == "back_left"
        assert env.damage_alpha == 0.6
        random_source.choice.assert_called_once_with(tuple(LEG_ACTION_INDICES))
        random_source.uniform.assert_called_once_with(0.25, 1.0)
    finally:
        env.close()


def test_random_damage_has_plausible_healthy_frequency() -> None:
    """Healthy episodes occur at approximately the configured 25% rate."""
    env = AntDamageWrapper(gym.make("Ant-v5"), mode="random")
    counts: Counter[str | None] = Counter()
    try:
        for episode in range(1_000):
            seed = 42 if episode == 0 else None
            _, info = env.reset(seed=seed)
            counts[info["damage_leg"]] += 1

        assert 0.20 <= counts[None] / 1_000 <= 0.30
        assert set(counts) == {None, *LEG_ACTION_INDICES}
    finally:
        env.close()


def test_fixed_damage_remains_fixed_across_resets() -> None:
    """Fixed evaluation conditions do not change between episodes."""
    env = AntDamageWrapper(
        gym.make("Ant-v5"),
        mode="fixed",
        fixed_leg="back_right",
        fixed_alpha=0.5,
    )
    action = np.zeros(8, dtype=np.float32)
    try:
        for seed in range(3):
            _, reset_info = env.reset(seed=seed)
            _, _, _, _, step_info = env.step(action)
            assert reset_info["damage_leg"] == "back_right"
            assert reset_info["damage_alpha"] == 0.5
            assert step_info["damage_leg"] == "back_right"
            assert step_info["damage_alpha"] == 0.5
    finally:
        env.close()


def test_wrapper_preserves_ant_spaces() -> None:
    """Wrapping Ant-v5 leaves its observation and action spaces intact."""
    base_env = gym.make("Ant-v5")
    action_space = base_env.action_space
    observation_space = base_env.observation_space
    env = AntDamageWrapper(base_env, mode="random")
    try:
        observation, _ = env.reset(seed=0)
        assert env.action_space is action_space
        assert env.observation_space is observation_space
        assert env.observation_space.contains(observation)
        assert env.action_space.contains(env.action_space.sample())
    finally:
        env.close()


@pytest.mark.parametrize(
    ("damage_leg", "highlighted_geoms"),
    [
        ("front_left", {2, 3, 4}),
        ("front_right", {5, 6, 7}),
        ("back_left", {8, 9, 10}),
        ("back_right", {11, 12, 13}),
    ],
)
def test_viewer_highlights_complete_damaged_leg(
    damage_leg: str,
    highlighted_geoms: set[int],
) -> None:
    """The viewer colors every geometry of only the selected leg."""
    env = AntDamageWrapper(gym.make("Ant-v5"), mode="nominal")
    model = env.unwrapped.model
    original_colors = model.geom_rgba.copy()
    try:
        _highlight_damage(env, damage_leg, original_colors)
        red_geoms = {
            geom_id
            for geom_id, color in enumerate(model.geom_rgba)
            if np.allclose(color, DAMAGE_COLOR)
        }
        assert red_geoms == highlighted_geoms

        _highlight_damage(env, None, original_colors)
        np.testing.assert_array_equal(model.geom_rgba, original_colors)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("mode", "fixed_leg", "fixed_alpha"),
    [
        ("nominal", None, None),
        ("random", None, None),
        ("fixed", "front_left", 0.5),
    ],
)
def test_stable_baselines3_checker_accepts_wrapper(
    mode: str,
    fixed_leg: str | None,
    fixed_alpha: float | None,
) -> None:
    """The wrapper follows the environment interface expected by SB3."""
    if mode == "fixed":
        env = AntDamageWrapper(
            gym.make("Ant-v5"),
            mode=mode,
            fixed_leg=fixed_leg,
            fixed_alpha=fixed_alpha,
        )
    else:
        env = AntDamageWrapper(gym.make("Ant-v5"), mode=mode)
    try:
        check_env(env, warn=True, skip_render_check=True)
    finally:
        env.close()
