# Set Environment

1. Docker image:
   For MI30X:
   ```
   rocm/sgl-dev:v0.5.7-rocm700-mi30x-20260106
   ```
   For MI35X:
   ```
   rocm/sgl-dev:v0.5.7-rocm700-mi35x-20260106
   ```

2. Install zejun/sglang dev/perf branch:
   ```
   git clone -b Qwen-Image https://github.com/zejunchen-zejun/sglang.git
   cd sglang
   pip install --upgrade pip
   cd sgl-kernel
   python3 setup_rocm.py install
   export PYTHONPATH=<you_sglang_path/sglang/python>
   ```
3. Other requirements:
    ```
   pip install remote_pdb
   pip install imageio
   pip install diffusers
   pip install cache_dit
    ```

# Inference

Here's a offline example to generate a picture:

First, download picture form https://github.com/modelscope/DiffSynth-Engine/blob/dev/qz/qwen_app_amd/examples/input/qwen_image_edit_input.png

```python
from sglang.multimodal_gen import DiffGenerator

def main():
    # Create a diff generator from a pre-trained model
    generator = DiffGenerator.from_pretrained(
        model_path="/data/models/Qwen-Image-Edit",
        num_gpus=4,
        ulysses_degree=4,
        enable_torch_compile=True,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        image_encoder_cpu_offload=False,
        image_encoder_precision="bf16",
        vae_precision="bf16",
    )

    # Generate the image
    for i in range(3):
        image = generator.generate(
            sampling_params_kwargs=dict(
                prompt="将画面中墙上的透明广告牌上的霓虹灯组词的'通义千问'四个字清除，重新写上'Muse平台'几个字，黑板上的内容不要改动",
                image_path="/workspace/sglang/evaluation/qwen_image_edit_input.png",  # need a absolute path
                output_path="qwen_image_edit/",
                save_output=True,
                height=1024,
                width=1024,
                num_inference_steps=15,
                seed=123,
            )
        )

if __name__ == '__main__':
    main()
```

Or, more simply, with the CLI:

```bash
SERVER_ARGS=(
  --model-path /data/models/Qwen-Image-Edit
  --num-gpus 4
  --ulysses-degree 4
  --enable-torch-compile
  --image-encoder-precision bf16
  --vae-precision bf16
)

SAMPLING_ARGS=(
  --prompt "将画面中墙上的透明广告牌上的霓虹灯组词的'通义千问'四个字清除，重新写上'Muse平台'几个字，黑板上的内容不要改动"
  --image-path /workspace/sglang/evaluation/qwen_image_edit_input.png
  --output-path qwen_image_edit/
  --save-output
  --height 1024
  --width 1024
  --num-inference-steps 15
  --output-file-name "qwen_image_edit.png"
)

sglang generate "${SERVER_ARGS[@]}" "${SAMPLING_ARGS[@]}"

# Or, users can set `SGLANG_CACHE_DIT_ENABLED` env as `true` to enable cache acceleration
SGLANG_CACHE_DIT_ENABLED=true sglang generate "${SERVER_ARGS[@]}" "${SAMPLING_ARGS[@]}"
```
Once the generation task has finished, the server will shut down automatically.

### LoRA support

Download picture form https://github.com/modelscope/DiffSynth-Engine/blob/dev/qz/qwen_app_amd/examples/input/768x1024.png

Apply LoRA adapters via `--lora-path`:

```bash
SERVER_ARGS=(
  --model-path /data/models/Qwen-Image-Edit-2511
  --lora-path /the/path/to/your/lora/file.safetensors
  --num-gpus 2
  --tp-size 1
  --ulysses-degree 2
)

SAMPLING_ARGS=(
  --prompt "make the clothes to red"
  --image-path ./768x1024.png
  --output-path qwen_image_edit/
  --save-output
  --height 1024
  --width 768
  --num-inference-steps 8
  --guidance-scale 1
  --output-file-name "red_clothes_edit.png"
)

sglang generate "${SERVER_ARGS[@]}" "${SAMPLING_ARGS[@]}"
```

# Online Server

## Custom Dataset
Download picture form https://github.com/modelscope/DiffSynth-Engine/blob/dev/qz/qwen_app_amd/examples/input/768x1024.png
To create a custom dataset for benchmarking, follow this structure:

```
/home/yajizhan/dev/benchmark_data/
├── data/
│   ├── i2v-bench-info.json
│   └── origin/
│       └── 768x1024.png
```

Example `i2v-bench-info.json`:

```json
[
    {"file_name": "768x1024.png", "caption": "make the clothes to red"},
    {"file_name": "768x1024.png", "caption": "make the clothes to red"},
    {"file_name": "768x1024.png", "caption": "make the clothes to red"},
    {"file_name": "768x1024.png", "caption": "make the clothes to red"},
    {"file_name": "768x1024.png", "caption": "make the clothes to red"}
]
```

### 1. Start the server with profiling environment

```bash
export PYTHONPATH=/path/to/sglang/python
export SGLANG_TORCH_PROFILER_DIR=./sglang_qwen_profiling
export SGLANG_PROFILE_WITH_STACK=1
export SGLANG_PROFILE_RECORD_SHAPES=1
export CUDA_VISIBLE_DEVICES=0,1
export SGLANG_CACHE_DIT_ENABLED=true

sglang serve \
    --model-path /mnt/raid0/pretrained_model/Qwen-Image-Edit \
    --lora_path /mnt/raid0/pretrained_model/Qwen-Image-Edit-2511-Lightning \
    --num-gpus 2 \
    --ulysses-degree 2 \
    --image-encoder-precision bf16 \
    --vae-precision bf16 \
    --host 0.0.0.0 \
    --port 30000
```

### 2. Run benchmark with profiling
To profile the server performance, add the `--profile` flag:
Then run benchmark with `--dataset-path`:
```bash
export PYTHONPATH=/path/to/sglang/python

python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
    --backend sglang-image \
    --task ti2i \
    --port 30000 \
    --dataset vbench \
    --dataset-path /home/yajizhan/dev/benchmark_data \
    --num-prompts 5 \
    --max-concurrency 1 \
    --width 768 \
    --height 1024 \
    --num-inference-steps 8 \
    --guidance-scale 1 \
    --profile
```

### 3. View trace files

After profiling, trace files are saved to `SGLANG_TORCH_PROFILER_DIR`:

```
./sglang_qwen_profiling/
├── 1736694000-host.trace.json.gz     # HTTP server process
├── 1736694000-rank-0.trace.json.gz   # GPU worker rank 0
└── 1736694000-rank-1.trace.json.gz   # GPU worker rank 1
```

Open these files in [Perfetto UI](https://ui.perfetto.dev/) or Chrome's `chrome://tracing` to visualize the performance.

# Evaluation (WIP)
