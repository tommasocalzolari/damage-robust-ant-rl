#!/usr/bin/env bash

set -Eeuo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

python_bin="$repo_dir/.venv/bin/python"
conditions=(nominal robust)
seeds=(1 2)
output_dirs=(
  artifacts/overnight/nominal_seed_1
  artifacts/overnight/robust_seed_2
)
log_files=(
  artifacts/overnight/nominal_seed_1.log
  artifacts/overnight/robust_seed_2.log
)
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
    printf '\n%s\n' \
      'Overnight training stopped. Partial outputs were kept for inspection.' >&2
  fi
}

trap report_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

assert_frozen_state() {
  local current_commit tracked_status
  current_commit="$(git rev-parse --verify HEAD)"
  tracked_status="$(git status --porcelain=v1 --untracked-files=no)"
  if [[ "$current_commit" != "$frozen_commit" ]]; then
    printf 'Git commit changed while overnight training was running.\n' >&2
    exit 1
  fi
  if [[ -n "$tracked_status" ]]; then
    printf 'Tracked files changed while overnight training was running.\n' >&2
    exit 1
  fi
}

preflight() {
  local git_root tracked_status target
  if [[ ! -x "$python_bin" ]]; then
    printf 'Missing project environment: %s\n' "$python_bin" >&2
    exit 1
  fi
  git_root="$(git rev-parse --show-toplevel)"
  if [[ "$git_root" != "$repo_dir" ]]; then
    printf 'The launcher must be located at the repository root.\n' >&2
    exit 1
  fi
  for source_file in "${0##*/}" src/damage_robust_ant/gait_train.py; do
    if ! git cat-file -e "HEAD:$source_file" 2>/dev/null; then
      printf 'Commit %s before starting overnight training.\n' "$source_file" >&2
      exit 1
    fi
  done
  if [[ ! -f results/main_episode_results.csv ]]; then
    printf 'Missing policy-selection data: results/main_episode_results.csv\n' >&2
    exit 1
  fi
  tracked_status="$(git status --porcelain=v1 --untracked-files=no)"
  if [[ -n "$tracked_status" ]]; then
    printf 'Commit or restore tracked changes before starting:\n%s\n' \
      "$tracked_status" >&2
    exit 1
  fi
  for target in "${output_dirs[@]}" "${log_files[@]}"; do
    if [[ -e "$target" ]]; then
      printf 'Refusing to overwrite existing output: %s\n' "$target" >&2
      exit 1
    fi
  done
  frozen_commit="$(git rev-parse --verify HEAD)"
}

if "$dry_run"; then
  printf 'Dry run only: no tests, training, evaluation or file writes.\n'
  printf '%s\n' \
    'Both policies start as new networks; one-million-step models are not resumed.'
  printf 'Status summaries will appear about every five minutes.\n'
else
  preflight
  printf 'Frozen Git commit: %s\n' "$frozen_commit"
  printf 'Running the test suite before the overnight job...\n'
  PYTHONDONTWRITEBYTECODE=1 \
    "$python_bin" -m pytest -p no:cacheprovider -q
  assert_frozen_state
  mkdir -p artifacts/overnight
  printf '\nLeave this terminal open. Both policies train sequentially.\n'
fi

for index in 0 1; do
  condition="${conditions[$index]}"
  seed="${seeds[$index]}"
  output_dir="${output_dirs[$index]}"
  log_file="${log_files[$index]}"
  policy_number=$((index + 1))
  command=(
    "$python_bin" -u -m damage_robust_ant.gait_train
    --condition "$condition"
    --policy-index "$policy_number"
    --total-policies 2
    --output-dir "$output_dir"
  )

  printf '\n[Overnight policy %d/2] condition=%s selected_seed=%s\n' \
    "$policy_number" "$condition" "$seed"
  if "$dry_run"; then
    printf 'DRY RUN TRAINING:'
    printf ' %q' "${command[@]}"
    printf '\n'
    continue
  fi

  assert_frozen_state
  "${command[@]}" 2>&1 | tee "$log_file"
  for required in selected_model.zip metadata.json validation_summary.csv; do
    if [[ ! -f "$output_dir/$required" ]]; then
      printf 'Training finished without required output: %s/%s\n' \
        "$output_dir" "$required" >&2
      exit 1
    fi
  done
done

if "$dry_run"; then
  printf '\nDry run complete: two fresh long-training commands.\n'
  exit 0
fi

assert_frozen_state
printf '\n%s\n' \
  'Overnight training complete. Review the criteria results before conclusions.'
printf 'Nominal model: %s/selected_model.zip\n' "${output_dirs[0]}"
printf 'Robust model: %s/selected_model.zip\n' "${output_dirs[1]}"
printf '\nVisual checks:\n'
printf '.venv/bin/python -m damage_robust_ant.view --model %s/selected_model.zip\n' \
  "${output_dirs[0]}"
printf '%s %s%s\n' \
  '.venv/bin/python -m damage_robust_ant.view --model' \
  "${output_dirs[1]}" \
  '/selected_model.zip --damage-mode fixed --leg front_left --alpha 0.5'
