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

# Benchmark
Online benchmark ti2i(qwen image edit) model, first, one window starts the server with the command below:

```bash
sglang serve --model-path Qwen/Qwen-Image-Edit-2511 --lora_path lightx2v/Qwen-Image-Edit-2511-Lightning  --num-gpus 1 --ulysses-degree=1 --image-encoder-precision bf16 --vae-precision bf16  --host 0.0.0.0 --port 30000
```
Second, Use another window to send requests, with the default "i2v-bench-info.json" parameter(s) from the VBench dataset.
```bash
#!/bin/bash
num_prompts=10
image_resolution="768x1024"
port=30000
width=768
height=1024

echo "num prompts: ${num_prompts}"
echo "image-resolution: ${image_resolution} (${width}x${height})"
echo "port: ${port}"

python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
    --backend sglang-image \
    --task ti2i \
    --dataset vbench \
    --num-prompts ${num_prompts} \
    --width ${width} \
    --height ${height} \
    --port ${port} \

```
# Evaluation (WIP)
