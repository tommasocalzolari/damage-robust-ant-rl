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

## Manual PPO walking gate

The corrected training path adds observation and reward normalization, a KL
early-stop guard, ten PPO update epochs, a learning-rate schedule from
`3e-4` to `1e-4`, and a walking-first Ant reward setting
(`healthy_reward=3.0`) so early falls are more costly without rewarding
standing still too strongly. Before spending more
time on robust training, run one bounded
nominal pilot yourself:

```bash
./run_ppo_gate.sh
```

The script runs the test suite, trains 500,000 requested steps, evaluates the
200,000-step and final checkpoints on ten deterministic healthy episodes, and
checks model reloads, normalization files, TensorBoard KL/clip diagnostics,
finite values, survival and horizontal distance from the episode start. It
reports speed and forward-only distance, but neither is used as a pass/fail rule. It exits nonzero
with `STOP` if the predefined walking gate is not met. It never starts robust
training, reuses an existing directory, deletes artifacts, or resumes a
partial run. Use `--dry-run` first to inspect the commands. Review a passing
run manually before choosing the next experiment.

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

The final experiment uses the walking configuration established by the
successful nominal seed 6 recovery run: `Ant-v5` with `healthy_reward=3.0`, an
`MlpPolicy` with separate `[256, 256]` actor and critic networks, four
environments, normalized observations and rewards, a learning rate annealed
from `3e-4` to `1e-4`, a clip range of `0.2`, `target_kl=0.02`, and 5,000,000
requested environment steps per run. Nominal and robust runs use identical PPO
settings; only the training damage distribution described above differs.

After this configuration is reviewed and committed, start the complete main
experiment with one command:

```bash
./run_main_experiment.sh
```

The launcher uses the project's `.venv` directly, so it does not need to be
activated first. It runs the tests, trains all six policies sequentially and
then performs the complete evaluation automatically. Keep the terminal open;
the six 5-million-step runs will take several hours. If a command fails or the
process is interrupted, it stops and preserves the partial outputs instead of
deleting or reusing them.

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
| Nominal | 5, 6, 7 | 5,000,000 | `artifacts/final/nominal_seed_<seed>/` |
| Robust | 5, 6, 7 | 5,000,000 | `artifacts/final/robust_seed_<seed>/` |

Every command initializes a new network; smoke and pilot models and checkpoints
are not reused. In particular, `artifacts/recovery/nominal_seed_6_5m/` is kept
untouched as the known working policy. PPO finishes complete rollouts, so each
new run is expected to record 5,005,312 actual steps rather than stop partway
through a rollout. Each
run also records its package versions, Git commit, tracked-worktree status,
resolved configuration, elapsed time and training speed in `metadata.json`.

The six final policies will be evaluated deterministically with the same ten
episode seeds, 300 through 309, under this fixed matrix:

| Evaluation condition | Damage leg | Alpha | Episodes per policy |
| --- | --- | ---: | ---: |
| Healthy | `healthy` | 1.0 | 10 |
| Moderate damage | Each of the four legs | 0.5 | 10 per leg |
| Complete failure | Each of the four legs | 0.0 | 10 per leg |

The four legs are `front_left`, `front_right`, `back_left` and `back_right`.
This gives nine conditions per policy and 540 episode rows across the six
policies. The rows, evaluation log, models, normalizers and training logs will
all be written under `artifacts/final/`; the combined CSV is
`artifacts/final/evaluation/episode_results.csv`.

These settings are frozen before the main results are inspected. Main training
starts only after this configuration is reviewed and committed, and parameters
will not be changed in response to the main results.

## Post-hoc overnight locomotion run

The main evaluation identified nominal seed 1 as the weakest nominal policy by
healthy forward distance and robust seed 2 as the strongest robust policy by
mean forward distance across the four legs at `alpha=0.5`. A separate overnight
launcher develops those two selected seeds further:

```bash
./run_overnight_gait_training.sh
```

This is an exploratory follow-up, not part of the six-policy nominal-versus-
robust comparison. Each policy starts again from a new network with its selected
seed and the original fixed PPO settings; the one-million-step model weights are
not resumed. Each run starts at timestep zero and continues uninterrupted for
3–5 million requested steps; it is not a continuation of the main model. The
saved models do not contain the simulator and random-number-generator states
needed for an exact continuation.

The launcher trains the policies sequentially, saves a complete-rollout
checkpoint about every 250,000 requested steps and evaluates every 500,000
steps from one million onward. It makes stopping decisions after 3, 4 and 5
million requested steps. The five-million hard limit is never crossed. If no
checkpoint passes by then, the best checkpoint is still saved but is clearly
marked as not meeting the criteria. “Best” means highest target-condition
forward distance, followed by lower early termination, higher speed, higher
distance in the other condition and then the earlier checkpoint.

Every validation uses deterministic actions and episode seeds 200 through 209
under healthy operation and under each of the four damaged legs at `alpha=0.5`.
The nominal policy is judged on its ten healthy episodes. The robust policy is
judged on its forty pooled damaged episodes. A candidate passes when its early
termination rate is at most 10%, mean forward speed is at least 0.5 m/s and at
least 90% of its episodes move forward. Seeds 200 through 209 are checkpoint-
selection data and must not be reused for the later held-out evaluation.

These are practical sustained-locomotion criteria, not a biomechanical gait-
cycle detector: hopping or sliding could also pass. The launcher prints viewer
commands at the end so the selected policies can be checked visually. Keep the
terminal open; the full run is expected to take roughly 3.5 to 6 hours on the
machine used for the main experiment, with a status update about every five
minutes. Outputs and logs are written under `artifacts/overnight/` and remain
untracked. An interrupted run preserves its partial files and requires review
before a deliberate fresh restart.

Use `./run_overnight_gait_training.sh --dry-run` to inspect both commands
without running tests, training or evaluation. Before a real run, the launcher
also verifies that `results/main_episode_results.csv` is the unchanged source
used for the two seed choices.

## Repository structure

- `src/damage_robust_ant/`: damage wrapper and training, evaluation and viewing
  commands
- `tests/`: automated tests for the environment and experiment pipeline
- `run_main_experiment.sh`: fixed main training and evaluation launcher
- `run_overnight_gait_training.sh`: selected-seed staged locomotion launcher
- `artifacts/`: local generated models, checkpoints and raw logs; ignored by
  Git
- `pyproject.toml`: package metadata and dependencies
