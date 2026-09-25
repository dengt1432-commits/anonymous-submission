#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
: "${TRAIN_SCRIPT:?Set TRAIN_SCRIPT to the training script to run}"
: "${CKPT:?Set CKPT to the trained model checkpoint}"

# Authentication and proxy settings are supplied by the caller or an existing login.
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -e "$PROJECT_ROOT/lmms-eval-main"
"$PYTHON_BIN" -m pip install -e "$PROJECT_ROOT/LLaVA-NeXT-main"
"$PYTHON_BIN" -m pip install ninja
"$PYTHON_BIN" -m pip install flash-attn --no-build-isolation

bash "$TRAIN_SCRIPT"

"$PYTHON_BIN" -m accelerate.commands.launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llava \
    --model_args pretrained="$CKPT" \
    --tasks ok_vqa,textcaps_val,mme_test,mmmu,cmmmu,coco2017_cap_val,vizwiz_vqa_val,ai2d,chartqa,pope \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix debug \
    --output_path ./logs/ \
    --wandb_args 'project=llava-next-lmms-eval,job_type=eval'
