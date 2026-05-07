#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="/mnt/data1/dmx/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="vllm_qw"
UV_VLLM_BIN="/home/dengmingxi/vllm/bin/vllm"
MODEL_PATH="/mnt/data1/dmx/Models/Qwen35_2b"

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen35_2b}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda init script not found: $CONDA_SH" >&2
  exit 1
fi

if [[ ! -x "$UV_VLLM_BIN" ]]; then
  echo "vLLM binary not found or not executable: $UV_VLLM_BIN" >&2
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path not found: $MODEL_PATH" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

exec "$UV_VLLM_BIN" serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gdn-prefill-backend "$GDN_PREFILL_BACKEND" \
  --trust-remote-code
