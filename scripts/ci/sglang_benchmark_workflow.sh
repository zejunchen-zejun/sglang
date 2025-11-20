model=${1}
TP=8
EP=8

echo "launching ${model}"
echo "TP=${TP}"
echo "EP=${EP}"

echo "Launching server"
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
    2>&1 | tee log.server.log &


echo "Sleeping for 60 seconds"
sleep 60
echo "Testing server"
wget https://sf-maas-uat-prod.oss-cn-shanghai.aliyuncs.com/dog.png
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