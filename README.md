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
