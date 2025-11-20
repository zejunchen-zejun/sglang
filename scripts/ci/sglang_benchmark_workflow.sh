#!/bin/bash
set -euo pipefail

model=${1:-/models/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic/}
TP=${2:-8}
EP=${3:-8}

echo
echo "========== LAUNCHING SERVER ========"
python3 -m sglang.launch_server \
    --model-path ${model} \
    --host localhost \
    --port 9000 \
    --tp-size ${TP} \
    --ep-size ${EP} \
    --trust-remote-code \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.6 \
    --disable-radix-cache \
    --max-prefill-tokens 32768 \
    --cuda-graph-max-bs 128 \

echo
echo "========== WAITING FOR SERVER TO BE READY ========"
max_retries=60
retry_interval=60
for ((i=1; i<=max_retries; i++)); do
    if curl -s http://localhost:9000/v1/completions -o /dev/null; then
        echo "SGLang server is up."
        break
    fi
    echo "Waiting for SGLang server to be ready... ($i/$max_retries)"
    sleep $retry_interval
done

if ! curl -s http://localhost:9000/v1/completions -o /dev/null; then
    echo "SGLang server did not start after $((max_retries * retry_interval)) seconds."
    kill $vllm_pid
    exit 1
fi

echo
echo "========== TESTING SERVER ========"
echo "Downloading test image"
wget https://sf-maas-uat-prod.oss-cn-shanghai.aliyuncs.com/dog.png
echo "Testing server with test image"
curl --request POST \
    --url "http://localhost:9000/v1/chat/completions" \
    --header "Content-Type: application/json" \
    --data '{
        "model": "/data/models/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic/",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "image_url",
                "image_url": {
                    "url": "/workspace/my_sgl/sglang/evaluation/dog.png"
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
    }'