# Qwen3-VL best practice with SGLang on MI300X

In the rapidly evolving field of artificial intelligence, efficiently deploying large language models (LLMs) and vision-language models (VLMs) is essential for real-time applications. Efficient inference turns foundational models into viable products: it lowers latency for better user experience, increases throughput for higher hardware utilization, and improves overall system efficiency—enabling larger batch sizes, longer contexts, and richer multimodal inputs within tight GPU memory and bandwidth constraints.

**SGLang** is designed to meet these demands. It provides a fast serving runtime with a flexible frontend programming model and broad model support, making it practical to integrate kernel-level acceleration, reduce host overhead, and expose an easy-to-use OpenAI-compatible serving interface for real workloads. These capabilities are particularly critical for serving Qwen3‑VL/MoE, whose pipeline integrates image preprocessing and a ViT/encoder stage with a large Mixture‑of‑Experts language model. While MoE architectures enable high-quality outputs with sparse expert activation, they introduce new performance challenges—such as expert dispatch and communication overhead. For vision-language serving, the multimodal path also adds non-trivial preprocessing and encoder cost that must be optimized together with decoding.

In this blog, we present a best practice for implementing Qwen3‑VL/MoE in SGLang on AMD MI300X GPUs, and we also provide a reproducible workflow to deploy the server and benchmark prefill/decode latency and throughput.


## Optimization in SGLang

For Qwen3-VL/Moe’s architecture, we have boost performance in SGLang by using the following several optimizations:

**1. Host overhead removed**
  - enable CUDA IPC
  - remove D2D, H2D, D2H memory copy
  - remove cpu synchronization

**2. VIT optimization**
  - enable DP VIT
  - fuse add & layerNorm kernel
  - optimize image resize as vLLM way
  - optimize hash function computation
  - accelerate image loading by replacing PIL with torchvision

**3. LLM framework optimization**
  - fuse rope kernel
  - fuse qknorm & rope/mrope kernel
  - fuse rmsnorm & quant_fp8 kernel
  - fuse all_reduce & rmsnorm & quant_fp8 kernel
  - tune moe kernel as 2 stage way

**4. High-performance operators Integrated**
  - integrate fast gelu kernel for VIT
  - integrate TRTLLM all_reduce kernel
  - integrate asm paged attention kernel
  - integrate fp8 flash_attn varlen kernel
  - Integrate hipblaslt swizzle fp8/bf16 gemm kernel
  - Integrate cuda allocate memory kernel

Next, we will delve into the details of optimizations above.

**Host overhead removed**

xxxxxx……


**VIT optimization**

xxxxxx……


**LLM framework optimization**

xxxxxx……


**High-performance operators Integrated**

xxxxxx……


## Performance Reproduction with SGLang

This guide outlines the steps to reproduce the performance results of Qwen3‑VL/MoE with SGLang v0.5.4.

**Step 1:** Run the Docker Container

```bash
docker run -itd --cap-add=SYS_PTRACE \
    --network=host --security-opt seccomp=unconfined \
    --name sgl_qwen3vl \
    --device=/dev/kfd \
    --device=/dev/dri \
    --shm-size 60g \
    -v /mnt:/mnt \
    -v /data:/data \
    --group-add video \
    --ipc=host \
    -e SGLANG_USE_AITER=1 \
    rocm/ali-private:ubuntu22.04_rocm6.4.3.127_sglang_619764_aiter_017f0e /bin/bash
```

**Step 2:** Launch Server

Launch the SGLang server with the following command:
```bash
export SGLANG_USE_AITER=1
export SGLANG_IO_WORKERS=8
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_ROCM_USE_AITER_PA_ASM_PRESHUFFLE_LAYOUT=0
export SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1
export SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=1

model=/the/path/to/your/model
TP=8
EP=1

python3 -m sglang.launch_server \
    --model-path ${model} \
    --host localhost \
    --port 9000 \
    --tp-size ${TP} \
    --ep-size ${EP} \
    --trust-remote-code \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --max-prefill-tokens 32768 \
    --cuda-graph-max-bs 128 \
    --mm-attention-backend aiter_attn \
    --mm-enable-dp-encoder \
    --enable-aiter-allreduce-fusion \
    --mm-processor-kwargs '{"max_pixels": 1638400, "min_pixels": 740}' \
    2>&1 | tee log.server.log &
```
**Step 3:** Run Benchmark

Run the benchmark script:
```bash
model=/the/path/to/your/model

input_tokens=8000
output_tokens=500
num_prompts=10
max_concurrency=1
image_count=5
image_resolution=960x1280

python3 -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --port 9000 \
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

**Results**

These results highlight the effectiveness of kernel-level optimizations and modular acceleration strategies within the SGLang pipeline.

- **Prefill Latency: ↓ xx%**

- **Decode Latency: ↓ xx%**

- **Total Throughput: ↑ xx%**

## Summary

This blog shares best practices for serving **Qwen3‑VL/MoE** efficiently on **MI300X** with **SGLang**, and summarizes the optimizations integrated in **SGLang v0.5.4** to reduce latency and improve throughput.

- **End-to-end overhead reduction**: minimize CPU synchronization and redundant memory copies, and leverage CUDA IPC transport to lower host-side overhead.
- **VIT/encoder acceleration**: enable DP encoder, fuse common kernels (e.g., add + layernorm), and speed up image preprocessing/loading with optimized resize and torchvision-based pipelines.
- **LLM runtime optimization**: fuse RoPE/QKNorm/RMSNorm paths, improve FP8 quant + all-reduce fusion, and tune MoE kernels for higher throughput.
- **High-performance operator integration**: adopt AITER/TRTLLM/ASM and FP8 FlashAttention varlen kernels, plus optimized GEMM and memory allocation paths to maximize GPU utilization.

Looking ahead, we will continue to closely monitor developments in the SGLang community,such as hierarchical KV caching, piecewise CUDA Graphs, and overlapping scheduling with speculative decoding, and will actively contribute by expanding operator coverage and pursuing further performance gains.

## Acknowledgement

We extend our sincere gratitude to the AMD AITER, CK & Framework Core teams for their invaluable support, technical collaboration, and insightful contributions throughout this work.


## Disclaimers
Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
