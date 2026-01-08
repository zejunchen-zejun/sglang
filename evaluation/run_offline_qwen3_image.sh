export PYTHONPATH=/the/path/to/your/sglang/python
export SGLANG_CACHE_DIT_ENABLED=true

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
