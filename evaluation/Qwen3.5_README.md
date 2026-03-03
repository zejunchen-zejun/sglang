# Set Environment

1. Docker image:
   For MI30X:
   ```
   rocm/ali-private:alinux3_rocm7.2.0.43_cp310_torch2.9.1_qwen3_5_mi30x_20260303
   ```
   For MI35X:
   ```
   rocm/ali-private:alinux3_rocm7.2.0.43_cp310_torch2.9.1_qwen3_5_20260302
   ```
2. Install aiter dev/perf branch:
   ```
    pip uninstall aiter
    git clone git@github.com:ROCm/aiter.git
    cd aiter
    git submodule sync && git submodule update --init --recursive
    rm -rf aiter/jit/*so aiter/jit/build
    # for MI308
    PREBUILD_KERNELS=1 GPU_ARCHS=gfx942 python3 setup.py install
    # for MI355
    PREBUILD_KERNELS=1 GPU_ARCHS=gfx950 python3 setup.py install
    ```

3. Install zejun/sglang dev/perf branch:
   ```
   git clone -b Qwen3.5_v0.5.9 https://github.com/zejunchen-zejun/sglang.git
   cd sglang
   pip install --upgrade pip
   cd sgl-kernel
   python setup_rocm.py install
   export PYTHONPATH=<you_sglang_path/sglang/python>
   ```

# Launch server
1. Qwen3.5-397B-A17B
- download BF16 model weight: https://huggingface.co/Qwen/Qwen3.5-397B-A17B

- launch server:
    ```
    bash launch_qwen35.sh
    ```
    The example command:
    ```bash
    export SGLANG_DISABLE_CUDNN_CHECK=1

    python3 -m sglang.launch_server \
            --port 9000 \
            --model-path /mnt/raid0/models/Qwen3.5-397B-A17B/ \
            --tp-size 8 \
            --attention-backend triton \
            --reasoning-parser qwen3 \
            --tool-call-parser qwen3_coder \
            --enable-multimodal \
            2>&1 | tee log.server.log &

    ```

# Curl request
1. curl a single request to quickly check the functionality
    First, download the test picture.
    ```bash
    wget https://sf-maas-uat-prod.oss-cn-shanghai.aliyuncs.com/dog.png
    ```
    Then curl a single quickly request
   ```
    curl --request POST \
        --url "http://localhost:9000/v1/chat/completions" \
        --header "Content-Type: application/json" \
        --data '{
            "model": "/mnt/raid0/models/Qwen3.5-397B-A17B/",
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
   ```
   The result should be:
   ```
    {"id":"97325912ba814dc8976a5bd7dc8044ba","object":"chat.completion","created":1772522712,"model":"/mnt/raid0/models/Qwen3.5-397B-A17B/","choices":[{"index":0,"message":{"role":"assistant","content":"这是一张**金毛寻回犬（或类似品种）的高清特写照片**。\n\n主要特征如下：\n*   **","reasoning_content":"用户要求简要描述图片内容。\n\n1.  **主体识别**：图片的主体是一只狗。\n2.  **品种/外观特征**：\n    *   毛色：金黄色/棕黄色（Golden/Brown）。\n    *   毛发：看起来比较蓬松、柔软，特别是耳朵和脖子周围。\n    *   眼睛：深棕色，眼神看起来很温和、专注，直视镜头。\n    *   鼻子：黑色，湿润。\n    *   耳朵：下垂的耳朵，典型的寻回犬特征。\n    *   整体感觉：很像金毛寻回犬（Golden Retriever）或者拉布拉多（Labrador）的混血，或者就是一只很漂亮的金毛。\n3.  **背景**：模糊的绿色背景，看起来像是在户外的草地或树林边。这种虚化效果（Bokeh）突出了主体。\n4.  **构图**：特写镜头（Close-up），主要聚焦在头部和上半身。\n5.  **风格**：看起来非常清晰，甚至有点过于完美，可能是一张AI生成的图片或者经过高度修饰的照片（毛发纹理非常细腻，眼神光很亮）。\n\n**综合描述草稿**：\n这是一张金毛寻回犬（或类似品种）的特写照片。它有着金黄色的毛发，垂下的耳朵，黑色的鼻子和深棕色的眼睛。它正看着镜头。背景是模糊的绿色。\n\n**润色（中文）**：\n这是一张狗狗的特写照片。\n- 主体是一只看起来像金毛寻回犬（Golden Retriever）的狗。\n- 它有着金棕色的蓬松毛发，耳朵下垂。\n- 它的眼睛是深褐色的，眼神看起来很温柔、专注，正直视着镜头。\n- 鼻子是黑色的，胡须清晰可见。\n- 背景是模糊的绿色，暗示它可能在户外的草地上。\n- 图片画质非常清晰，光影柔和。\n\n**最终简化版**：\n这是一张金毛寻回犬（或类似品种）的高清特写照片。这只狗有着金棕色的柔软毛发和下垂的耳朵，正用它深棕色、温柔的眼睛直视镜头。背景是模糊的绿色，突出了狗狗可爱的面部表情。\n","tool_calls":null},"logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":1043,"total_tokens":1543,"completion_tokens":500,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}
   ```

# Benchmark
1. To benchmark image dataset with 1 images per request, 128 prompts, 1000 input length, and 2000 output length, you can run:
    ```bash
    export SGLANG_USE_AITER=1
    model=/mnt/raid0/models/Qwen3.5-397B-A17B/

    input_tokens=8000
    output_tokens=650
    max_concurrency=1
    num_prompts=10
    image_count=5
    image_resolution=960x1280

    python3 -m sglang.bench_serving \
        --backend sglang-oai-chat \
        --port 9000 \
        --model ${model} \
        --dataset-name image \
        --num-prompts ${num_prompts} \
        --image-count ${image_count} \
        --image-resolution ${image_resolution} \
        --random-input-len ${input_tokens} \
        --random-output-len ${output_tokens} \
        --max-concurrency ${max_concurrency} \
        --random-range-ratio 1.0 \
        2>&1 | tee log.client.log
    ```

# Profile
1. set the env flags
    ```bash
    export SGLANG_TORCH_PROFILER_DIR=./
    export SGLANG_PROFILE_WITH_STACK=1
    export SGLANG_PROFILE_RECORD_SHAPES=1
    <launch the server>
    <launch the client with the additional --profile argument>
    ```
Please make sure that the `SGLANG_TORCH_PROFILER_DIR` should be set at both server and client side, otherwise the trace file cannot be generated correctly.

# Evaluation

## Vision Model Evaluation

Vision model is evaluated on MMMU dataset.

1. First, install the lmms-eval package:
    ```bash
    git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    cd lmms-eval
    pip3 install -e .
    ```
2. Start evaluating:
    ```bash
    export OPENAI_API_KEY=EMPTY
    export OPENAI_API_BASE=http://localhost:9000/v1
    export PYTHONPATH=/the/path/to/your/sglang/python

    python3 -m lmms_eval \
        --model=openai_compatible \
        --model_args model_version=/mnt/raid0/models/Qwen3.5-397B-A17B/ \
        --tasks mmmu_val   \
        --batch_size 16 \
    ```

3. Other

    We can also use benchmark to evaluate VLM accuracy.  More information you can find in the [benchmark/mmmu/README.md](../benchmark/mmmu/README.md).
    ```bash
    python benchmark/mmmu/bench_sglang.py --port 9000 --concurrency 16
    ```
