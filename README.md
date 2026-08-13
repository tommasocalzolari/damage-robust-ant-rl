# Damage-Robust Ant RL

## Research question

How does training under randomized single-leg actuator degradation affect the
robustness and healthy performance of a PPO locomotion controller?

Stable Baselines3 provides the PPO implementation. This repository implements
the Ant damage wrapper and the commands used to train, inspect and evaluate the
policies.

## Training conditions

Both conditions use the same PPO configuration. Nominal training leaves all
actuator commands unchanged. Robust training samples damage once per episode:
25% of episodes remain healthy; otherwise one leg is selected uniformly and
its remaining actuator strength is sampled uniformly from `[0.25, 1.0]`.

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

Run the remaining commands from the repository root with the environment
active. In a new terminal, reactivate it with `source .venv/bin/activate`.

## Test

Run the test suite from the repository root:

```bash
python -m pytest
```

The suite covers the damage wrapper, training configuration and evaluation
pipeline. It also starts `Ant-v5`, resets it and takes a simulation step to
verify that MuJoCo works.

## Manual viewer

Open a MuJoCo window with smooth, low-strength random actions:

```bash
python -m damage_robust_ant.view
```

This is a visual check rather than an automated test. Without `--model`, the
Ant uses random actions and may lose its balance. Use `--speed 0.25` for slower
playback or `--strength 0.2` for gentler actuator commands.

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
python -m damage_robust_ant.train \
  --condition nominal --seed 0 --timesteps 10000 --num-envs 4 \
  --output-dir artifacts/smoke/nominal_seed_0

python -m damage_robust_ant.train \
  --condition robust --seed 0 --timesteps 10000 --num-envs 4 \
  --output-dir artifacts/smoke/robust_seed_0
```

Each directory contains the final model, periodic checkpoints, monitor data,
TensorBoard logs and a metadata file. All smoke outputs are stored under
`artifacts/smoke/`, and training refuses to reuse an existing output directory.
Generated models, checkpoints, raw Monitor and TensorBoard logs, and videos
under `artifacts/` are ignored by Git.

Open TensorBoard to inspect the recorded episode and training metrics:

```bash
tensorboard --logdir artifacts/smoke
```

Leave the command running and open <http://localhost:6006> in a browser. Press
`Ctrl+C` in the terminal to stop it. The reduced-feature warning about
TensorFlow is harmless here because policy training uses PyTorch.

The manual viewer can also run a saved policy, including under fixed damage:

```bash
python -m damage_robust_ant.view \
  --model artifacts/smoke/nominal_seed_0/final_model.zip

python -m damage_robust_ant.view \
  --model artifacts/smoke/robust_seed_0/final_model.zip \
  --damage-mode fixed --leg front_left --alpha 0.5
```

The 10,000-step smoke policies only verify the training pipeline and are not
expected to walk well. Later trained models can be passed to the same command.

## Controlled evaluation

Evaluate a policy without further learning under a healthy or fixed-damage
condition:

```bash
python -m damage_robust_ant.evaluate \
  --model artifacts/smoke/nominal_seed_0/final_model.zip \
  --training-condition nominal --training-seed 0 \
  --damage-leg healthy --alpha 1.0 \
  --evaluation-seed 100 --episodes 1 \
  --output-csv artifacts/smoke/evaluation/episode_results.csv

python -m damage_robust_ant.evaluate \
  --model artifacts/smoke/nominal_seed_0/final_model.zip \
  --training-condition nominal --training-seed 0 \
  --damage-leg front_left --alpha 0.5 \
  --evaluation-seed 100 --episodes 1 \
  --output-csv artifacts/smoke/evaluation/episode_results.csv --append
```

The evaluation seed is the first episode seed; later episodes use consecutive
seeds so that identical starting states can be reused across policies and
damage conditions. The CSV records return, duration, forward movement and mean
absolute commands for all eight actuators. Raw and applied command magnitudes
are controller-command proxies, not measurements of physical energy.

## Fixed main experiment

The main experiment uses the committed PPO configuration without further
tuning: `Ant-v5`, an `MlpPolicy` with separate `[256, 256]` actor and critic
networks, four environments, a learning rate of `3e-4`, a clip range of `0.2`
and 1,000,000 requested environment steps per run. The remaining PPO settings
are fixed in the training module and recorded in each run's metadata. Nominal
and robust runs use identical PPO settings; only the training damage
distribution described above differs.

After this configuration is reviewed and committed, start the complete main
experiment with one command:

```bash
./run_main_experiment.sh
```

The launcher uses the project's `.venv` directly, so it does not need to be
activated first. It runs the tests, trains all six policies sequentially and
then performs the complete evaluation automatically. Keep the terminal open;
the full process should take roughly four hours on the machine used for the
pilot. If a command fails or the process is interrupted, it stops and preserves
the partial outputs instead of deleting or reusing them.

The normal PPO tables remain visible as training runs. About every five minutes,
an additional progress line reports the current run, its completed steps, the
number of full runs still waiting and a rough training-time estimate. These
updates are also saved in that run's console log. The estimate is recalculated
from the current run's speed, so it can change during the experiment.

To inspect all planned commands without training or writing any files, run
`./run_main_experiment.sh --dry-run`.

This fixes the training matrix as follows:

| Training condition | Seeds | Requested steps per run | Output directory |
| --- | --- | ---: | --- |
| Nominal | 0, 1, 2 | 1,000,000 | `artifacts/main/nominal_seed_<seed>/` |
| Robust | 0, 1, 2 | 1,000,000 | `artifacts/main/robust_seed_<seed>/` |

Every command initializes a new network; smoke and pilot models and checkpoints
are not reused. PPO finishes complete rollouts, so each run is expected to
record 1,007,616 actual steps rather than stop partway through a rollout. Each
run also records its package versions, Git commit, tracked-worktree status,
resolved configuration, elapsed time and training speed in `metadata.json`.

The six final policies will be evaluated deterministically with the same ten
episode seeds, 100 through 109, under this fixed matrix:

| Evaluation condition | Damage leg | Alpha | Episodes per policy |
| --- | --- | ---: | ---: |
| Healthy | `healthy` | 1.0 | 10 |
| Moderate damage | Each of the four legs | 0.5 | 10 per leg |
| Complete failure | Each of the four legs | 0.0 | 10 per leg |

The four legs are `front_left`, `front_right`, `back_left` and `back_right`.
This gives nine conditions per policy and 540 episode rows across the six
policies. The rows will be written to `results/main_episode_results.csv`.

These settings are frozen before the main results are inspected. Main training
starts only after this configuration is reviewed and committed, and parameters
will not be changed in response to the main results.

## Repository structure

- `src/damage_robust_ant/`: damage wrapper and training, evaluation and viewing
  commands
- `tests/`: automated tests for the environment and experiment pipeline
- `run_main_experiment.sh`: fixed main training and evaluation launcher
- `artifacts/`: local generated models, checkpoints and raw logs; ignored by
  Git
- `pyproject.toml`: package metadata and dependencies
