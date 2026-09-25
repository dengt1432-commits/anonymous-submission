#!/usr/bin/env bash
set -euo pipefail

# Resolve local packages and weights independently of the caller's directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/LLaVA-NeXT-main:$PROJECT_ROOT/lmms-eval-main${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi
export TRANSFORMERS_LLAMA_TRITON_ROUTER="${TRANSFORMERS_LLAMA_TRITON_ROUTER:-1}"
export TRANSFORMERS_LLAMA_SKIP_FUSIONS="${TRANSFORMERS_LLAMA_SKIP_FUSIONS:-1}"

NUM_PROC="${NUM_PROC:-1}"
CKPT="${CKPT:-liuhaotian/llava-v1.6-vicuna-7b}"
ROUTER_CKPT="${ROUTER_CKPT:-$PROJECT_ROOT/router_weight/16_1.6.safetensors}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/log}"
RUN_TAG="${RUN_TAG:-llava_1.6}"

if [[ ! -f "$ROUTER_CKPT" ]]; then
  echo "Router checkpoint not found: $ROUTER_CKPT" >&2
  exit 1
fi

TASKS=(
  mme
  # ocrbench
  # gqa
  # mmstar
  # seedbench
  # textvqa_val
  # mmbench_en_dev
  # vizwiz_vqa_val
  # pope
)
#  mmbench_cn vizwiz_vqa_val vqav2_val textvqa_val pope

mkdir -p "$OUT_ROOT"
FAIL_LOG="$OUT_ROOT/failed_tasks_${RUN_TAG}.txt"
: > "$FAIL_LOG"

for task in "${TASKS[@]}"; do
  ts="$(date +%Y%m%d_%H%M%S)"
  out_dir="$OUT_ROOT/${task}"
  mkdir -p "$out_dir"

  log_file="$out_dir/run_${RUN_TAG}_${ts}.log"

  echo "=============================="
  echo "[RUN] task=$task  time=$ts"
  echo "out_dir=$out_dir"
  echo "log=$log_file"
  echo "=============================="

  set +e
  "$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes=$NUM_PROC \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$CKPT",attn_implementation="sdpa",router_weight_path="$ROUTER_CKPT" \
    --tasks "$task" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${RUN_TAG}_${task}" \
    --output_path "$out_dir" \
    "$@" \
    2>&1 | tee "$log_file"
  ret="${PIPESTATUS[0]}"
  set -e

  if [[ "$ret" -ne 0 ]]; then
    echo "[FAIL] $task (exit=$ret)"
    echo "$task  exit=$ret  log=$log_file" >> "$FAIL_LOG"
  else
    echo "[OK]   $task"
  fi
done

echo "All done. Failed tasks (if any): $FAIL_LOG"
[[ ! -s "$FAIL_LOG" ]]
