from sglang.multimodal_gen import DiffGenerator


def main():
    # Create a diff generator from a pre-trained model
    generator = DiffGenerator.from_pretrained(
        model_path="/data/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        num_gpus=8,
        ulysses_degree=8,
        tp_size=1,
        enable_torch_compile=True,
        dit_layerwise_offload=False,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        image_encoder_cpu_offload=False,
        image_encoder_precision="bf16",
        vae_precision="bf16",
    )

    # Generate the video
    # warmup
    video = generator.generate(
        sampling_params_kwargs=dict(
            prompt="On a bright, sunny day with a clear blue sky stretching endlessly, a vast coastline unfolds—pristine golden sand meets the gently rolling turquoise waves of the ocean. Scattered across the horizon, small rocky reefs occasionally break the surface of the calm sea. A young woman in a flowing, light-colored long dress walks barefoot along the shoreline, her steps light and graceful. Her expression is serene and dreamy, utterly lost in thought, as if she’s the only person in the world—completely free from worry or pressure. Suddenly, a sleek, metallic UFO descends silently from the sky and lands smoothly on the beach behind her. The craft’s hatch opens with a soft hiss, and two humanoid robots step out, their movements precise and mechanical. Without hesitation, they approach the woman, gently but firmly take her by the arms, and escort her into the UFO. The vessel lifts off moments later, leaving only ripples in the sand and the endless sea behind.",
            return_frames=True,  # Also return frames from this call (defaults to False)
            output_path="my_videos/",  # Controls where videos are saved
            save_output=False,
            num_frames=61,
            height=480,
            width=832,
            num_inference_steps=20,
            seed=42,
        )
    )
    video = generator.generate(
        sampling_params_kwargs=dict(
            prompt="On a bright, sunny day with a clear blue sky stretching endlessly, a vast coastline unfolds—pristine golden sand meets the gently rolling turquoise waves of the ocean. Scattered across the horizon, small rocky reefs occasionally break the surface of the calm sea. A young woman in a flowing, light-colored long dress walks barefoot along the shoreline, her steps light and graceful. Her expression is serene and dreamy, utterly lost in thought, as if she’s the only person in the world—completely free from worry or pressure. Suddenly, a sleek, metallic UFO descends silently from the sky and lands smoothly on the beach behind her. The craft’s hatch opens with a soft hiss, and two humanoid robots step out, their movements precise and mechanical. Without hesitation, they approach the woman, gently but firmly take her by the arms, and escort her into the UFO. The vessel lifts off moments later, leaving only ripples in the sand and the endless sea behind.",
            return_frames=True,  # Also return frames from this call (defaults to False)
            output_path="my_videos/",  # Controls where videos are saved
            save_output=True,
            num_frames=61,
            height=480,
            width=832,
            num_inference_steps=20,
            seed=42,
        )
    )


if __name__ == "__main__":
    main()
