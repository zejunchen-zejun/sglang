#!/bin/bash

set -euo pipefail

TYPE=${1:-launch}
MODEL_NAME=${2:-offical_qwen3p5_397B_ptpc}
MODEL_PATH=${3:-/data/models/Qwen/uc/offical_qwen3p5_397B_ptpc}
TP=${4:-8}
EP=${5:-1}
TIMEOUT=${6:-90}
PORT=${SGLANG_BENCHMARK_PORT:-8080}
MMMU_CONCURRENCY=${SGLANG_BENCHMARK_MMMU_CONCURRENCY:-16}
SERVER_LOG=${SGLANG_BENCHMARK_SERVER_LOG:-/tmp/sglang_qwen35_server.log}

export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
# Override to 0 for Ali-specific runs when required.
export SGLANG_VLM_CACHE_SIZE_MB="${SGLANG_VLM_CACHE_SIZE_MB:-8192}"
export SGLANG_USE_AITER=1
export SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1
export SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=1
export SGLANG_USE_AITER_NEW_CA=true

echo "Detect TYPE: ${TYPE}"
echo "Detect model_name: ${MODEL_NAME}"
echo "Detect model_path: ${MODEL_PATH}"
echo "Detect TP: ${TP}"
echo "Detect EP: ${EP}"
echo "Detect TIMEOUT: ${TIMEOUT}"
echo "Detect PORT: ${PORT}"

wait_for_server() {
  local max_retries=${1}
  local retry_interval=60

  echo
  echo "========== WAITING FOR SERVER TO BE READY =========="
  for ((i=1; i<=max_retries; i++)); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null; then
      echo "SGLang server is up."
      return 0
    fi

    echo "Waiting for SGLang server to be ready... (${i}/${max_retries})"

    if [[ -f "${SERVER_LOG}" ]]; then
      tail -n 20 "${SERVER_LOG}" || true
    fi

    sleep "${retry_interval}"
  done

  echo "SGLang server did not start after $((max_retries * retry_interval)) seconds."
  if [[ -f "${SERVER_LOG}" ]]; then
    echo "========== SERVER LOG =========="
    tail -n 200 "${SERVER_LOG}" || true
  fi
  return 1
}

smoke_test_server() {
  echo
  echo "========== TESTING SERVER =========="
  curl --fail --request POST \
    --url "http://localhost:${PORT}/v1/chat/completions" \
    --header "Content-Type: application/json" \
    --data "$(cat <<EOF
{
  "model": "${MODEL_PATH}",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "https://sf-maas-uat-prod.oss-cn-shanghai.aliyuncs.com/dog.png"
          }
        },
        {
          "type": "text",
          "text": "请简要描述图片是什么内容？"
        }
      ]
    }
  ],
  "temperature": 0.0,
  "top_p": 0.0001,
  "top_k": 1,
  "max_tokens": 100
}
EOF
)"
}

if [[ "${TYPE}" == "launch" ]]; then
  echo
  echo "========== LAUNCHING SERVER =========="

  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model path does not exist: ${MODEL_PATH}"
    exit 1
  fi

  pkill -f "python3 -m sglang.launch_server" || true
  rm -f "${SERVER_LOG}"
  model="${MODEL_PATH}"

  nohup python3 -m sglang.launch_server \
    --port "${PORT}" \
    --model-path "${model}" \
    --tp-size "${TP}" \
    --attention-backend triton \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-multimodal \
    --trust-remote-code \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.9 \
    --max-prefill-tokens 32768 \
    --max-running-requests 128 \
    --disable-radix-cache \
    --mm-attention-backend aiter_attn \
    > "${SERVER_LOG}" 2>&1 &

  if ! wait_for_server "${TIMEOUT}"; then
    pkill -f "python3 -m sglang.launch_server" || true
    exit 1
  fi

  smoke_test_server

elif [[ "${TYPE}" == "evaluation" ]]; then
  echo
  echo "========== STARTING MODEL EVALUATION =========="
  python3 benchmark/mmmu/bench_sglang.py \
    --port "${PORT}" \
    --concurrency "${MMMU_CONCURRENCY}" \
    | tee "vision_model_evaluation_${MODEL_NAME}_TP${TP}_EP${EP}.log"

elif [[ "${TYPE}" == "performance" ]]; then
  echo
  echo "========== STARTING PERFORMANCE BENCHMARK =========="
  python3 -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --port "${PORT}" \
    --model "${MODEL_PATH}" \
    --dataset-name image \
    --num-prompts 10 \
    --image-count 5 \
    --image-resolution 960x1280 \
    --random-input-len 8000 \
    --random-output-len 650 \
    --max-concurrency 1 \
    --random-range-ratio 1.0 \
    | tee "performance_benchmark_${MODEL_NAME}_TP${TP}_EP${EP}.log"

else
  echo "Unknown TYPE: ${TYPE}"
  echo "Usage: $0 {launch|evaluation|performance} [model_name] [model_path] [TP] [EP] [timeout]"
  exit 1
fi

exit_code=$?
if [[ ${exit_code} -eq 0 ]]; then
  echo
  echo "========== SGLANG BENCHMARK ${TYPE} COMPLETED SUCCESSFULLY =========="
else
  echo
  echo "========== SGLANG BENCHMARK ${TYPE} FAILED WITH EXIT CODE ${exit_code} =========="
  exit "${exit_code}"
fi
