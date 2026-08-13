# Damage-Robust Ant RL

This project studies how training with randomized single-leg actuator
degradation affects the robustness and healthy performance of a PPO controller
in Gymnasium's `Ant-v5` environment.

## Setup

Python 3.10 or newer is required. Create a virtual environment and install the
project with its test dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.3,<3"
python -m pip install -e ".[dev]"
```

The explicit PyTorch command installs the CPU build, which is sufficient for
the tests and avoids downloading unused CUDA libraries on machines without an
NVIDIA GPU.

## Test

Run the test suite from the repository root:

```bash
python -m pytest
```

The smoke test starts `Ant-v5`, resets it with a fixed seed, and takes one
simulation step to verify that the MuJoCo environment is available.

## Manual viewer

Open a MuJoCo window with smooth, low-strength random actions:

```bash
python -m damage_robust_ant.view
```

This is a visual check rather than an automated test. The Ant is not controlled
by a trained policy yet, so it may lose its balance. Use `--speed 0.25` for
slower playback or `--strength 0.2` for gentler actuator commands.

The viewer also supports randomized training damage and fixed evaluation
damage:

```bash
python -m damage_robust_ant.view --damage-mode random
python -m damage_robust_ant.view --damage-mode fixed --leg front_left --alpha 0.5
```

The damaged leg is shown in red. Random mode updates the highlight at each
episode reset, while healthy episodes retain the original colors. The terminal
also prints the selected leg and its remaining actuator strength.

## Actuator damage

Damage is sampled once during reset and remains fixed until the episode ends.
Random mode produces a healthy episode 25% of the time. Otherwise, it selects
one leg uniformly and samples its remaining strength from `[0.25, 1.0]`. Fixed
mode accepts a chosen leg and strength in `[0, 1]`. The same strength multiplier
is applied to that leg's hip and ankle commands.

After each step, the wrapper retains the original command in `policy_action`
and the scaled command sent to Ant in `applied_action`. These values support
later command-magnitude analysis; they are not physical energy measurements.

The MuJoCo actuator objects in `Ant-v5` are unnamed. The mapping combines the
programmatically inspected target joints with the body directions listed in the
[official Ant-v5 documentation](https://gymnasium.farama.org/environments/mujoco/ant/):

| Leg | Action indices | Target joints |
| --- | --- | --- |
| `front_left` | 2, 3 | `hip_1`, `ankle_1` |
| `front_right` | 4, 5 | `hip_2`, `ankle_2` |
| `back_left` | 6, 7 | `hip_3`, `ankle_3` |
| `back_right` | 0, 1 | `hip_4`, `ankle_4` |
