#!/usr/bin/env bash

set -Eeuo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

python_bin="$repo_dir/.venv/bin/python"
final_dir="artifacts/final"
results_csv="$final_dir/evaluation/episode_results.csv"
evaluation_log="$final_dir/evaluation/evaluation.log"
training_conditions=(nominal robust)
training_seeds=(5 6 7)
damage_legs=(front_left front_right back_left back_right)
frozen_commit=""
dry_run=false

usage() {
  printf 'Usage: %s [--dry-run]\n' "${0##*/}"
}

if (( $# == 1 )) && [[ "$1" == "--dry-run" ]]; then
  dry_run=true
elif (( $# != 0 )); then
  usage >&2
  exit 2
fi

report_failure() {
  local status=$?
  if (( status != 0 )); then
    printf '\nExperiment stopped. Any partial outputs were kept for inspection.\n' >&2
  fi
}

trap report_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

print_command() {
  local label=$1
  shift
  printf 'DRY RUN %s:' "$label"
  printf ' %q' "$@"
  printf '\n'
}

assert_frozen_state() {
  local current_commit tracked_status
  current_commit="$(git rev-parse --verify HEAD)"
  tracked_status="$(git status --porcelain=v1 --untracked-files=no)"

  if [[ "$current_commit" != "$frozen_commit" ]]; then
    printf 'Git commit changed while the experiment was running.\n' >&2
    exit 1
  fi
  if [[ -n "$tracked_status" ]]; then
    printf 'Tracked files changed while the experiment was running.\n' >&2
    exit 1
  fi
}

preflight() {
  local git_root worktree_status

  if [[ ! -x "$python_bin" ]]; then
    printf 'Missing project environment: %s\n' "$python_bin" >&2
    exit 1
  fi

  git_root="$(git rev-parse --show-toplevel)"
  if [[ "$git_root" != "$repo_dir" ]]; then
    printf 'The script must be located at the repository root.\n' >&2
    exit 1
  fi
  if ! git cat-file -e "HEAD:${0##*/}" 2>/dev/null; then
    printf 'Commit %s before starting the experiment.\n' "${0##*/}" >&2
    exit 1
  fi

  worktree_status="$(git status --porcelain=v1 --untracked-files=no)"
  if [[ -n "$worktree_status" ]]; then
    printf 'Commit tracked changes before starting:\n%s\n' \
      "$worktree_status" >&2
    exit 1
  fi

  if [[ -e "$final_dir" ]]; then
    printf 'Refusing to overwrite existing final output: %s\n' "$final_dir" >&2
    printf 'Move it aside only after inspecting it, then start a fresh run.\n' >&2
    exit 1
  fi

  frozen_commit="$(git rev-parse --verify HEAD)"
}

# Freeze the code and refuse to overwrite a previous experiment.
if "$dry_run"; then
  printf 'Dry run only: no tests, training, evaluation or file writes.\n'
  printf 'Live training summaries are printed every 5 minutes.\n\n'
else
  preflight
  printf 'Frozen Git commit: %s\n' "$frozen_commit"
  printf 'Running the test suite before training...\n'
  PYTHONDONTWRITEBYTECODE=1 \
    "$python_bin" -m pytest -p no:cacheprovider -q
  assert_frozen_state
  mkdir -p "$final_dir/evaluation"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  printf '\nLeave this terminal open until training and evaluation finish.\n'
fi

training_index=0
# Train sequentially to keep laptop resource use predictable.
for training_condition in "${training_conditions[@]}"; do
  for training_seed in "${training_seeds[@]}"; do
    training_index=$((training_index + 1))
    run_dir="$final_dir/${training_condition}_seed_${training_seed}"
    log_file="${run_dir}.log"
    training_command=(
      "$python_bin" -u -m damage_robust_ant.train
      --condition "$training_condition"
      --seed "$training_seed"
      --timesteps 5000000
      --num-envs 4
      --learning-rate 0.0003
      --clip-range 0.2
      --run-index "$training_index"
      --total-runs 6
      --output-dir "$run_dir"
    )

    printf '\n[Training %d/6] condition=%s seed=%s\n' \
      "$training_index" "$training_condition" "$training_seed"
    if "$dry_run"; then
      print_command TRAINING "${training_command[@]}"
      continue
    fi

    assert_frozen_state
    "${training_command[@]}" 2>&1 | tee "$log_file"
    if [[ ! -f "$run_dir/final_model.zip" || \
          ! -f "$run_dir/vecnormalize.pkl" || \
          ! -f "$run_dir/metadata.json" ]]; then
      printf 'Training finished without the required outputs: %s\n' \
        "$run_dir" >&2
      exit 1
    fi
  done
done

if (( training_index != 6 )); then
  printf 'Internal error: expected six training runs.\n' >&2
  exit 1
fi

evaluation_index=0
# Reuse the same episode seeds for every policy and damage condition.
evaluate_condition() {
  local training_condition=$1
  local training_seed=$2
  local damage_leg=$3
  local damage_alpha=$4
  local run_dir model normalizer append_args=() evaluation_command

  evaluation_index=$((evaluation_index + 1))
  run_dir="$final_dir/${training_condition}_seed_${training_seed}"
  model="$run_dir/final_model.zip"
  normalizer="$run_dir/vecnormalize.pkl"
  if (( evaluation_index > 1 )); then
    append_args=(--append)
  fi
  evaluation_command=(
    "$python_bin" -m damage_robust_ant.evaluate
    --model "$model"
    --normalizer "$normalizer"
    --training-condition "$training_condition"
    --training-seed "$training_seed"
    --damage-leg "$damage_leg"
    --alpha "$damage_alpha"
    --evaluation-seed 300
    --episodes 10
    --output-csv "$results_csv"
    "${append_args[@]}"
  )

  printf '\n[Evaluation %d/54] condition=%s seed=%s leg=%s alpha=%s\n' \
    "$evaluation_index" "$training_condition" "$training_seed" \
    "$damage_leg" "$damage_alpha"
  if "$dry_run"; then
    print_command EVALUATION "${evaluation_command[@]}"
    return
  fi

  assert_frozen_state
  "${evaluation_command[@]}" 2>&1 | tee -a "$evaluation_log"
}

if ! "$dry_run"; then
  printf '\nAll six training runs finished. Starting evaluation.\n'
  for training_condition in "${training_conditions[@]}"; do
    for training_seed in "${training_seeds[@]}"; do
      run_dir="$final_dir/${training_condition}_seed_${training_seed}"
      if [[ ! -f "$run_dir/final_model.zip" || \
            ! -f "$run_dir/vecnormalize.pkl" || \
            ! -f "$run_dir/metadata.json" ]]; then
        printf 'Required training output is missing: %s\n' "$run_dir" >&2
        exit 1
      fi
    done
  done
fi

for training_condition in "${training_conditions[@]}"; do
  for training_seed in "${training_seeds[@]}"; do
    evaluate_condition "$training_condition" "$training_seed" healthy 1.0
    for damage_alpha in 0.5 0.0; do
      for damage_leg in "${damage_legs[@]}"; do
        evaluate_condition \
          "$training_condition" "$training_seed" "$damage_leg" "$damage_alpha"
      done
    done
  done
done

if (( evaluation_index != 54 )); then
  printf 'Internal error: expected 54 evaluation commands.\n' >&2
  exit 1
fi

if "$dry_run"; then
  printf '\nDry run complete: 6 training commands and 54 evaluation commands.\n'
  printf 'The full evaluation will produce 540 episode rows.\n'
  exit 0
fi

assert_frozen_state
row_count="$(
  "$python_bin" -c \
    'import csv, sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline=""))))' \
    "$results_csv"
)"
if [[ "$row_count" != "540" ]]; then
  printf 'Expected 540 evaluation rows, found %s.\n' "$row_count" >&2
  exit 1
fi

printf '\nMain experiment complete.\n'
printf 'Training and evaluation outputs: %s/\n' "$final_dir"
printf 'Evaluation results: %s\n' "$results_csv"
