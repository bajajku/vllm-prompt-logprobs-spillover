#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BIN="${VENV_BIN:-$SCRIPT_DIR/.venv/bin}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8025}"
HOST="${HOST:-127.0.0.1}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_gpu${GPU_ID}_port${PORT}.log"

CUSTOM_BUCKETS='{"request_latency":[0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.3,0.5,0.75,1.0,1.5,2.0,3.0,5.0,10.0,30.0],"time_to_first_token":[0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.3,0.5,0.75,1.0,1.5,2.0,3.0,5.0,10.0,30.0]}'

echo "Starting vLLM v0.34.0 server on GPU ${GPU_ID}, ${HOST}:${PORT}..."
echo "Model: ${MODEL} (bfloat16), max_model_len=${MAX_MODEL_LEN}, gpu_mem_util=${GPU_MEM_UTIL}"
echo "Log: ${LOG_FILE}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
# Enables POST /reset_prefix_cache for clean inter-run prefix-cache isolation (bound to 127.0.0.1 only).
export VLLM_SERVER_DEV_MODE=1

# Redirect stdout/stderr to tee while letting `exec` keep vllm as the direct PID for clean signals.
exec > >(tee -a "${LOG_FILE}") 2>&1

exec "${VENV_BIN}/vllm" serve "${MODEL}" \
  --dtype bfloat16 \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --enable-prefix-caching \
  --generation-config vllm \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --custom-histogram-buckets "${CUSTOM_BUCKETS}" \
  "$@"
