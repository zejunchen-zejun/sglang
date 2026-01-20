from sglang.multimodal_gen import DiffGenerator

def main():
    # Create a diff generator from a pre-trained model
    generator = DiffGenerator.from_pretrained(
        model_path="Qwen/Qwen-Image-Edit",
        lora_path="lightx2v/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        num_gpus=2,
        ulysses_degree=2,
        tp_size=1,
        enable_torch_compile=True,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        image_encoder_cpu_offload=False,
        image_encoder_precision="bf16",
        vae_precision="bf16",
    )

    # warm-up
    generator.generate(
            sampling_params_kwargs=dict(
                prompt="make the clothes to red",
                image_path="./qwen_image_inputs/768x1024.png",
                output_path="./qwen_image_edit/",  # Controls where images are saved
                height=1024,
                width=768,
                num_inference_steps=8,
                seed=42,
                guidance_scale=1, 
                # profile=True, # profiling flag
                # profile_all_stages=True, # profiling flag
            )
        )
    # Generate the image
    generator.generate(
            sampling_params_kwargs=dict(
                prompt="make the clothes to red",
                image_path="./qwen_image_inputs/768x1024.png",
                output_path="./qwen_image_edit/",  # Controls where images are saved
                height=1024,
                width=768,
                num_inference_steps=8,
                seed=42,
                guidance_scale=1, 
                # profile=True, # profiling flag
                # profile_all_stages=True, # profiling flag
            )
        )

if __name__ == '__main__':
    main()