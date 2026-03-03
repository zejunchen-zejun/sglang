# export PYTHONPATH=/home/qichu/my_sgl/upstream_sglang/sglang/python
export SGLANG_USE_AITER=1
model=/mnt/raid0/models/Qwen3.5-397B-A17B/

input_tokens=8000
output_tokens=650
max_concurrency=1
num_prompts=10
image_count=5
image_resolution=960x1280

# for profiling
# export SGLANG_TORCH_PROFILER_DIR=./qwen3vl-profile-layernorm
# export SGLANG_PROFILE_WITH_STACK=1
#add --profile as argument

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
    # --profile \
    # 2>&1 | tee log.without-stack-profile.log
