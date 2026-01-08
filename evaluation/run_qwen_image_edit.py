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
                image_path="/data/users/qichu/ali_sglang/sglang/evaluation/qwen_image_edit_input.png",  # need a absolute path
                output_path="qwen_image_edit/",
                save_output=True,
                height=1024,
                width=1024,
                num_inference_steps=15,
                seed=123,
            )
        )


if __name__ == "__main__":
    main()
