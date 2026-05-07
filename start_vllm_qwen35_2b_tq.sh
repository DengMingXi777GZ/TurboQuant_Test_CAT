#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="/mnt/data1/dmx/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="vllm_qw"
UV_VLLM_BIN="/home/deng/vllm/bin/vllm"
MODEL_PATH="/mnt/data1/dmx/Models/Qwen35_2b"

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen35_2b}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"

# TurboQuant 配置
TQ_ENABLED="${TQ_ENABLED:-0}"  # 0=关闭, 1=开启
TQ_KEY_BITS="${TQ_KEY_BITS:-3}"
TQ_VALUE_BITS="${TQ_VALUE_BITS:-2}"
TQ_BUFFER_SIZE="${TQ_BUFFER_SIZE:-128}"

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

# 设置 TurboQuant 环境变量
if [[ "$TQ_ENABLED" == "1" ]]; then
  echo "🚀 启动 vLLM (TurboQuant 已开启: key_bits=$TQ_KEY_BITS, value_bits=$TQ_VALUE_BITS)"
  export TURBOQUANT_ENABLED=1
  export TURBOQUANT_KEY_BITS="$TQ_KEY_BITS"
  export TURBOQUANT_VALUE_BITS="$TQ_VALUE_BITS"
  export TURBOQUANT_BUFFER_SIZE="$TQ_BUFFER_SIZE"
else
  echo "🚀 启动 vLLM (TurboQuant 关闭)"
  unset TURBOQUANT_ENABLED
fi

exec "$UV_VLLM_BIN" serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gdn-prefill-backend "$GDN_PREFILL_BACKEND" \
  --trust-remote-code
