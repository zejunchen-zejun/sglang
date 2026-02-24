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

2. Install xFuser:
   ```
   pip install xfuser
   ```
   Or install from source:
   ```
   git clone https://github.com/xdit-project/xDiT.git
   cd xDiT
   pip install -e .
   ```

3. Install sglang dev/perf branch:
   ```
   git clone https://github.com/sgl-project/sglang.git
   cd sglang
   pip install --upgrade pip
   cd sgl-kernel
   python3 setup_rocm.py install
   export PYTHONPATH=<your_sglang_path/sglang/python>
   ```

4. Other requirements:
    ```
   pip install torch
   pip install diffusers
   pip install accelerate
   pip install transformers
    ```

5. Optional: For AMD GPU optimization:
   ```
   pip install opt_groupnorm
   ```

# Download Model

Download the Z-Image-Turbo model from ModelScope:

```bash
export HF_HOME=/path/to/huggingface/cache
huggingface-cli download Tongyi-MAI/Z-Image-Turbo
```

Or download from ModelScope:
```bash
modelscope download --model Tongyi-MAI/Z-Image-Turbo
```

Model page: https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo/summary

# Inference

## Simple Test Script

Here's a simple offline example to generate an image without argument parsing:

### Using sglang DiffGenerator (Recommended)

```python
from sglang.multimodal_gen import DiffGenerator

def main():
    # Create a diff generator from a pre-trained model
    generator = DiffGenerator.from_pretrained(
        model_path="Tongyi-MAI/Z-Image-Turbo",
        num_gpus=1,
        ulysses_degree=1,
        enable_torch_compile=True,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
    )

    # Generate the image
    image = generator.generate(
        sampling_params_kwargs=dict(
            prompt="A crowded beach",
            output_path="z_image_output/",
            save_output=True,
            height=720,
            width=1280,
            num_inference_steps=9,
            guidance_scale=0.0,
            seed=42,
        )
    )

if __name__ == '__main__':
    main()
```

Save this as `test_zimage.py` and run:
```bash
python test_zimage.py
```

## Advanced Usage with xFuser (Distributed Inference)

For distributed inference with xFuser, you can use the provided benchmark script with proper argument parsing. See the **CLI Script** section below for details.

## CLI Script

Or, use the `sglang generate` command:

```bash
SERVER_ARGS=(
  --model-path Tongyi-MAI/Z-Image-Turbo
  --num-gpus 1
  --ulysses-degree 1
  --enable-torch-compile
  --dit-cpu-offload False
  --text-encoder-cpu-offload False
  --vae-cpu-offload False
)

SAMPLING_ARGS=(
  --prompt "A crowded beach"
  --output-path z_image_output/
  --save-output
  --height 720
  --width 1280
  --num-inference-steps 9
  --guidance-scale 0.0
  --seed 42
  --output-file-name "z_image_result.png"
)

sglang generate "${SERVER_ARGS[@]}" "${SAMPLING_ARGS[@]}"
```

Once the generation task has finished, the server will shut down automatically and the image will be saved to `z_image_output/z_image_result.png`.

# Online Server

Start the SGLang diffusion server with the Z-Image-Turbo model:

```bash
sglang serve --model-path Tongyi-MAI/Z-Image-Turbo \
  --num-gpus 1 \
  --ulysses-degree 1 \
  --host 0.0.0.0 \
  --port 40000
```

For local model paths (e.g. pre-downloaded weights), point `--model-path` to the directory:

```bash
sglang serve --model-path /path/to/Z-Image-Turbo \
  --num-gpus 1 \
  --ulysses-degree 1 \
  --host 0.0.0.0 \
  --port 40000
```

# Evaluation

## Online Benchmark

Once the server is running, evaluate with the benchmark script:

```bash
python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
  --backend sglang-image \
  --task text-to-image \
  --port 40000 \
  --dataset vbench \
  --num-prompts 10 \
  --max-concurrency 1 \
  --width 1280 \
  --height 720 \
  --num-inference-steps 9 \
  --guidance-scale 0.0
```

Key arguments:

| Argument | Description |
| --- | --- |
| `--task` | Generation task: `text-to-image`, `image-to-image`, `text-to-video`, `image-to-video`, `video-to-video`. When the model is loaded from a local path, `--task` is required because the HuggingFace `pipeline_tag` cannot be inferred automatically. |
| `--dataset` | Dataset source: `vbench` or `random` |
| `--dataset-path` | Path to a local dataset directory (optional) |
| `--num-prompts` | Number of prompts to benchmark |
| `--max-concurrency` | Maximum concurrent requests |
| `--width` / `--height` | Output resolution |
| `--num-inference-steps` | Number of denoising steps |
| `--guidance-scale` | Classifier-free guidance scale |

## Image-to-Image (Qwen-Image-Edit)

For Qwen-Image-Edit evaluation, launch with the image editing model:

```bash
sglang serve --model-path Qwen/Qwen-Image-Edit-2511 \
  --num-gpus 1 \
  --host 0.0.0.0 \
  --port 40000
```

Then benchmark:

```bash
python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
  --backend sglang-image \
  --task image-to-image \
  --port 40000 \
  --dataset vbench \
  --dataset-path /path/to/benchmark_data \
  --num-prompts 5 \
  --max-concurrency 1 \
  --width 768 \
  --height 1024 \
  --num-inference-steps 40 \
  --guidance-scale 4
```
