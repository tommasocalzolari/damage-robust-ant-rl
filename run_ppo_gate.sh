#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_dir"
python_bin="$repo_dir/.venv/bin/python"
timesteps=500000
seed=6
episodes=10
output_dir="$repo_dir/artifacts/recovery/manual_gate_nominal_seed_${seed}"
dry_run=0

usage() {
    cat <<'EOF'
Usage: ./run_ppo_gate.sh [options]

Run tests, one fresh nominal PPO pilot, deterministic evaluations, and a
go/no-go check. This script never starts robust training.

Options:
  --output-dir PATH   fresh output directory
  --timesteps N       pilot budget (default: 500000)
  --seed N            training seed (default: 6)
  --episodes N        episodes per checkpoint (default: 10)
  --dry-run           print the planned commands and exit
  -h, --help          show this help
EOF
}

while (($#)); do
    case "$1" in
        --output-dir) [[ $# -ge 2 ]] || { echo "missing --output-dir value" >&2; exit 2; }; output_dir=$2; shift 2 ;;
        --timesteps) [[ $# -ge 2 ]] || { echo "missing --timesteps value" >&2; exit 2; }; timesteps=$2; shift 2 ;;
        --seed) [[ $# -ge 2 ]] || { echo "missing --seed value" >&2; exit 2; }; seed=$2; shift 2 ;;
        --episodes) [[ $# -ge 2 ]] || { echo "missing --episodes value" >&2; exit 2; }; episodes=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$timesteps" =~ ^[1-9][0-9]*$ ]] || { echo "--timesteps must be positive" >&2; exit 2; }
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "--seed must be non-negative" >&2; exit 2; }
[[ "$episodes" =~ ^[1-9][0-9]*$ ]] || { echo "--episodes must be positive" >&2; exit 2; }
((timesteps >= 200000 && timesteps % 100000 == 0)) || {
    echo "--timesteps must be at least 200000 and a multiple of 100000" >&2
    exit 2
}
[[ -x "$python_bin" ]] || { echo "missing executable: $python_bin" >&2; exit 2; }

if [[ "$output_dir" != /* ]]; then output_dir="$repo_dir/$output_dir"; fi
output_dir="$(dirname "$output_dir")/$(basename "$output_dir")"
log_file="${output_dir}.train.log"
if [[ -e "$output_dir" || -e "$log_file" ]]; then
    echo "Refusing to overwrite an existing pilot or log: $output_dir" >&2
    echo "Choose a new --output-dir; preserved artifacts are never deleted." >&2
    exit 3
fi

echo "Repository: $repo_dir"
echo "Pilot output: $output_dir"
echo "Budget: $timesteps requested steps, seed $seed, $episodes episodes"
if [[ -n "$(git status --porcelain=v1 --untracked-files=no)" ]]; then
    echo "Warning: tracked worktree changes are present; metadata will record this."
fi
if ((dry_run)); then
    echo "DRY RUN: $python_bin -m pytest -q"
    echo "DRY RUN: $python_bin -u -m damage_robust_ant.train --condition nominal --seed $seed --timesteps $timesteps --num-envs 4 --output-dir $output_dir"
    echo "DRY RUN: evaluate checkpoints at 200000 and $timesteps with $episodes episodes"
    exit 0
fi

echo "[1/4] Running the automated test suite..."
"$python_bin" -m pytest -q

mkdir -p "$(dirname "$output_dir")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[2/4] Training one fresh nominal pilot. Keep this terminal open."
"$python_bin" -u -m damage_robust_ant.train \
    --condition nominal --seed "$seed" --timesteps "$timesteps" --num-envs 4 \
    --output-dir "$output_dir" 2>&1 | tee "$log_file"

for required_file in "$output_dir/final_model.zip" "$output_dir/vecnormalize.pkl" "$output_dir/metadata.json"; do
    [[ -f "$required_file" ]] || { echo "missing output: $required_file" >&2; exit 4; }
done

echo "[3/4] Evaluating healthy behavior at 200k and the final checkpoint..."
for checkpoint in 200000 "$timesteps"; do
    model="$output_dir/checkpoints/ppo_${checkpoint}_steps.zip"
    normalizer="$output_dir/checkpoints/ppo_vecnormalize_${checkpoint}_steps.pkl"
    [[ -f "$model" && -f "$normalizer" ]] || { echo "missing checkpoint sidecar: $checkpoint" >&2; exit 5; }
    "$python_bin" -m damage_robust_ant.evaluate \
        --model "$model" --normalizer "$normalizer" \
        --training-condition nominal --training-seed "$seed" \
        --damage-leg healthy --alpha 1 --evaluation-seed 300 \
        --episodes "$episodes" --output-csv "$output_dir/eval_${checkpoint}.csv"
done

echo "[4/4] Checking saved files, PPO diagnostics, and walking criteria..."
"$python_bin" - "$output_dir" "$timesteps" "$episodes" <<'PY'
import json
import math
import sys
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = Path(sys.argv[1])
requested = int(sys.argv[2])
episodes = int(sys.argv[3])
metadata = json.loads((run / "metadata.json").read_text())
actual = int(metadata["actual_environment_steps"])
if actual < requested:
    raise SystemExit(f"FAIL: actual steps {actual} are below requested {requested}")
validation = metadata["validation"]
if not validation["final_model_reloaded"] or not validation["normalization_state_reloaded"]:
    raise SystemExit("FAIL: model or normalization reload validation is false")
for step in range(100000, requested + 1, 100000):
    for suffix in ("zip", "pkl"):
        path = run / "checkpoints" / f"ppo{'_' if suffix == 'zip' else '_vecnormalize_'}{step}_steps.{suffix}"
        if not path.is_file():
            raise SystemExit(f"FAIL: missing checkpoint artifact: {path.name}")

event_files = sorted((run / "tensorboard").rglob("events.out.tfevents.*"))
if not event_files:
    raise SystemExit("FAIL: missing TensorBoard event files")
accumulators = []
tags = set()
for event_file in event_files:
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    accumulators.append(accumulator)
    tags.update(accumulator.Tags().get("scalars", []))
required_tags = {"train/approx_kl", "train/clip_fraction", "rollout/ep_len_mean"}
missing = required_tags - tags
if missing:
    raise SystemExit(f"FAIL: missing TensorBoard diagnostics: {sorted(missing)}")

def finite_series(tag):
    values = [
        item.value
        for accumulator in accumulators
        if tag in accumulator.Tags().get("scalars", [])
        for item in accumulator.Scalars(tag)
    ]
    if not values or not all(math.isfinite(value) for value in values):
        raise SystemExit(f"FAIL: non-finite or empty TensorBoard series: {tag}")
    return values

kl = finite_series("train/approx_kl")
clip = finite_series("train/clip_fraction")
if max(kl) > 0.05:
    raise SystemExit(f"FAIL: approximate KL safety limit exceeded ({max(kl):.4f})")
if max(clip) > 0.50:
    raise SystemExit(f"FAIL: clip-fraction safety limit exceeded ({max(clip):.3f})")

def inspect_eval(step):
    frame = pd.read_csv(run / f"eval_{step}.csv")
    columns = {
        "episode_return",
        "episode_length",
        "terminated_before_time_limit",
        "forward_distance",
        "horizontal_distance",
        "mean_forward_speed",
        "mean_horizontal_speed",
    }
    if len(frame) != episodes or not columns <= set(frame.columns):
        raise SystemExit(f"FAIL: malformed evaluation CSV at checkpoint {step}")
    if not set(frame["terminated_before_time_limit"].unique()) <= {True, False}:
        raise SystemExit(f"FAIL: malformed termination flags at checkpoint {step}")
    numeric = frame[
        [
            "episode_return",
            "episode_length",
            "forward_distance",
            "horizontal_distance",
            "mean_forward_speed",
            "mean_horizontal_speed",
        ]
    ]
    if not numeric.map(lambda value: math.isfinite(float(value))).all().all():
        raise SystemExit(f"FAIL: non-finite evaluation value at checkpoint {step}")
    if not frame["episode_length"].between(1, 1000).all():
        raise SystemExit(f"FAIL: invalid episode length at checkpoint {step}")
    early = float(frame["terminated_before_time_limit"].mean())
    forward_distance = float(frame["forward_distance"].mean())
    horizontal_distance = float(frame["horizontal_distance"].mean())
    speed = float(frame["mean_forward_speed"].mean())
    horizontal_speed = float(frame["mean_horizontal_speed"].mean())
    print(
        f"checkpoint {step}: early={early:.3f} "
        f"forward={forward_distance:.3f}m "
        f"horizontal={horizontal_distance:.3f}m "
        f"forward_speed={speed:.3f}m/s "
        f"horizontal_speed={horizontal_speed:.3f}m/s"
    )
    return early, horizontal_distance, horizontal_speed

inspect_eval(200000)
early, horizontal_distance, horizontal_speed = inspect_eval(requested)
if early > 0.10 or horizontal_distance < 5.00:
    print("STOP: the pilot did not demonstrate sustained walking; do not launch robust training.")
    raise SystemExit(10)
print("PASS: nominal pilot meets the predefined walking gate.")
print(f"Mean horizontal distance was {horizontal_distance:.3f} m.")
print(f"Mean horizontal speed was {horizontal_speed:.3f} m/s; it is reported but not used as a pass/fail rule.")
print("Review this run manually before starting any robust pilot.")
PY

echo "Gate complete. Outputs are under $output_dir."
