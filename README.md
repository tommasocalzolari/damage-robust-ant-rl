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

## Smoke training

Run short nominal and robust training checks in separate output directories:

```bash
python -m damage_robust_ant.train --condition nominal --seed 0 --timesteps 10000 --num-envs 4 --output-dir artifacts/smoke/nominal_seed_0
python -m damage_robust_ant.train --condition robust --seed 0 --timesteps 10000 --num-envs 4 --output-dir artifacts/smoke/robust_seed_0
```

Each directory contains the final model, periodic checkpoints, monitor data,
TensorBoard logs and a metadata file. Training refuses to reuse an existing
output directory. Generated files under `artifacts/` are not tracked by Git.

Open TensorBoard to inspect the recorded episode and training metrics:

```bash
tensorboard --logdir artifacts/smoke
```

The manual viewer can also run a saved policy, including under fixed damage:

```bash
python -m damage_robust_ant.view --model artifacts/smoke/nominal_seed_0/final_model.zip
python -m damage_robust_ant.view --model artifacts/smoke/robust_seed_0/final_model.zip --damage-mode fixed --leg front_left --alpha 0.5
```

The 10,000-step smoke policies only verify the training pipeline and are not
expected to walk well. Later trained models can be passed to the same command.

## Controlled evaluation

Evaluate a policy without further learning under a healthy or fixed-damage
condition:

```bash
python -m damage_robust_ant.evaluate --model artifacts/smoke/nominal_seed_0/final_model.zip --training-condition nominal --training-seed 0 --damage-leg healthy --alpha 1.0 --evaluation-seed 100 --episodes 1 --output-csv artifacts/smoke/evaluation/episode_results.csv
python -m damage_robust_ant.evaluate --model artifacts/smoke/nominal_seed_0/final_model.zip --training-condition nominal --training-seed 0 --damage-leg front_left --alpha 0.5 --evaluation-seed 100 --episodes 1 --output-csv artifacts/smoke/evaluation/episode_results.csv --append
```

The evaluation seed is the first episode seed; later episodes use consecutive
seeds so that identical starting states can be reused across policies and
damage conditions. The CSV records return, duration, forward movement and mean
absolute commands for all eight actuators. Raw and applied command magnitudes
are controller-command proxies, not measurements of physical energy.
