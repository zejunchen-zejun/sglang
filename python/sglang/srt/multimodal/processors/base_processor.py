import concurrent
import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import BaseImageProcessorFast

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.utils import (
    get_bool_env_var,
    is_npu,
    load_audio,
    load_image_tensor,
    load_video,
    logger,
)
from sglang.srt.utils.cuda_ipc_transport_utils import (
    MM_FEATURE_CACHE_SIZE,
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
)

_is_npu = is_npu()

SGL_USE_CUDA_IPC = get_bool_env_var("SGLANG_USE_CUDA_IPC_TRANSPORT")


@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    # input_text, with each frame of video/image represented with a image_token
    input_text: str

    # frames loaded from image, in given order
    images: Optional[list[Union[Image.Image, dict]]] = dataclasses.field(
        default_factory=list
    )

    # videos
    videos: Optional[list[Union[torch.Tensor, dict]]] = dataclasses.field(
        default_factory=list
    )

    # audios
    audios: Optional[list[Union[np.ndarray, dict]]] = dataclasses.field(
        default_factory=list
    )

    def organize_results(self) -> List[Tuple[Modality, Any]]:
        """

        :return: a list of results, with their corresponding modalities
        """
        return (
            [(Modality.IMAGE, data) for data in self.images]
            + [(Modality.VIDEO, data) for data in self.videos]
            + [(Modality.AUDIO, data) for data in self.audios]
        )


@dataclasses.dataclass
class MultimodalSpecialTokens:
    image_token: Optional[Union[str, List[str]]] = None
    video_token: Optional[Union[str, List[str]]] = None
    audio_token: Optional[Union[str, List[str]]] = None

    image_token_id: Optional[int] = None
    video_token_id: Optional[int] = None
    audio_token_id: Optional[int] = None

    image_token_regex: Optional[re.Pattern] = None
    video_token_regex: Optional[re.Pattern] = None
    audio_token_regex: Optional[re.Pattern] = None

    combined_regex: Optional[re.Pattern] = None

    def build(self, processor):
        self.convert_to_strs(processor)
        self.parse_regex()
        self.get_combined_regex()
        return self

    def convert_to_str(self, token: Union[str, int], processor) -> str:
        if token is None:
            return token
        if isinstance(token, str):
            return token
        return processor.tokenizer.convert_ids_to_tokens([token])[0]

    def convert_to_strs(self, processor):
        if not self.image_token:
            self.image_token = self.convert_to_str(self.image_token_id, processor)
        if not self.video_token:
            self.video_token = self.convert_to_str(self.video_token_id, processor)
        if not self.audio_token:
            self.audio_token = self.convert_to_str(self.audio_token_id, processor)

    def get_modality_of_token(self, token: str) -> Optional[Modality]:
        """
        :return: the modality associated with the given token, if the token is a special_token or matches with the multimodal token regex
        """
        modality = {
            self.image_token: Modality.IMAGE,
            self.video_token: Modality.VIDEO,
            self.audio_token: Modality.AUDIO,
        }.get(token)
        if modality:
            return modality

        for regex, modality in [
            (self.image_token_regex, Modality.IMAGE),
            (self.video_token_regex, Modality.VIDEO),
            (self.audio_token_regex, Modality.AUDIO),
        ]:
            if regex and regex.match(token):
                return modality

        return None

    def get_token_id_by_modality(self, modality: Modality) -> Optional[int]:
        return {
            Modality.IMAGE: self.image_token_id,
            Modality.MULTI_IMAGES: self.image_token_id,
            Modality.VIDEO: self.video_token_id,
            Modality.AUDIO: self.audio_token_id,
        }.get(modality)

    def parse_regex(self):
        if self.image_token_regex is None and self.image_token is not None:
            self.image_token_regex = re.compile(re.escape(self.image_token))
        if self.video_token_regex is None and self.video_token is not None:
            self.video_token_regex = re.compile(re.escape(self.video_token))
        if self.audio_token_regex is None and self.audio_token is not None:
            self.audio_token_regex = re.compile(re.escape(self.audio_token))

    def get_combined_regex(self) -> re.Pattern:
        """
        Builds and returns a regex, used to split input str into tokens (with mm special tokens)
        """
        if self.combined_regex:
            return self.combined_regex
        tokens = [
            self.image_token_regex,
            self.video_token_regex,
            self.audio_token_regex,
        ]
        patterns = []
        flags = 0
        for t in tokens:
            if t is not None:
                patterns.append(t.pattern)
                flags |= t.flags
        combined = "(" + "|".join(f"(?:{p})" for p in patterns) + ")"
        self.combined_regex = re.compile(combined, flags)
        return self.combined_regex


class BaseMultimodalProcessor(ABC):
    models = []

    def __init__(
        self, hf_config, server_args, _processor, transport_mode, *args, **kwargs
    ):
        self.hf_config = hf_config
        self._processor = _processor
        self.server_args = server_args
        self.transport_mode = transport_mode

        # FIXME: not accurate, model and image specific
        self.NUM_TOKEN_PER_FRAME = 330

        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_IO_WORKERS", 4))
        )
        self.cpu_executor = concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("fork"),
            max_workers=int(os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())),
        )

        # Mapping from attribute names to modality types
        self.ATTR_NAME_TO_MODALITY = {
            # Image-related attributes
            "pixel_values": Modality.IMAGE,
            "image_sizes": Modality.IMAGE,
            "image_grid_thw": Modality.IMAGE,
            "image_attention_mask": Modality.IMAGE,
            "image_emb_mask": Modality.IMAGE,
            "images_spatial_crop": Modality.IMAGE,
            "images_crop": Modality.IMAGE,
            "tgt_size": Modality.IMAGE,
            "image_grid_hws": Modality.IMAGE,
            "aspect_ratio_ids": Modality.IMAGE,
            "aspect_ratio_mask": Modality.IMAGE,
            "num_patches": Modality.IMAGE,
            "patch_pixel_values": Modality.IMAGE,
            "block_sizes": Modality.IMAGE,
            # Audio-related attributes
            "audio_features": Modality.AUDIO,
            "audio_feature_lens": Modality.AUDIO,
            "input_features": Modality.AUDIO,
            "input_features_mask": Modality.AUDIO,
            "audio_attention_mask": Modality.AUDIO,
            "feature_attention_mask": Modality.AUDIO,
            # Video-related attributes
            "pixel_values_videos": Modality.VIDEO,
            "second_per_grid_ts": Modality.VIDEO,
            "video_grid_thw": Modality.VIDEO,
            # Generic attributes that could apply to multiple modalities
            # "precomputed_embeddings" - handled specially as it can be any modality
        }

        # name of the feature filed
        # TODO: pass from processors
        self.FEATURE_NAMES = [
            "pixel_values",
            "pixel_values_videos",
            "audio_features",
            "input_features",
        ]

        if SGL_USE_CUDA_IPC:
            logger.info(
                f"[CUDA IPC] CUDA IPC transport enabled for multimodal processor. "
                f"Initializing memory pool with size {MM_FEATURE_CACHE_SIZE / (1024*1024):.2f} MB"
            )
            self.cudaipc_mmfeature_pool = MmItemMemoryPool(MM_FEATURE_CACHE_SIZE)
        else:
            logger.info(
                "[CUDA IPC] CUDA IPC transport disabled, using default CPU transport"
            )

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ) -> dict:
        """
        process multimodal data with transformers AutoProcessor
        """
        import time

        logger.info(f"[DEBUG_TRACE] Entering process_mm_data")

        # Step 5.1.1: Prepare kwargs
        logger.info(f"[DEBUG_TRACE] Preparing kwargs for processor")
        substep1_start = time.time()
        if images:
            kwargs["images"] = images
            logger.info(f"[DEBUG_TRACE] Added {len(images)} images to kwargs")
        if videos:
            kwargs["videos"] = videos
            logger.info(f"[DEBUG_TRACE] Added {len(videos)} videos to kwargs")
        if audios:
            if self._processor.__class__.__name__ in {
                "Gemma3nProcessor",
                "Qwen2AudioProcessor",
                "Qwen3OmniMoeProcessor",
            }:
                # Note(Xinyuan): for gemma3n, ref: https://github.com/huggingface/transformers/blob/ccf2ca162e33f381e454cdb74bf4b41a51ab976d/src/transformers/models/gemma3n/processing_gemma3n.py#L107
                kwargs["audio"] = audios
                kwargs["audio_kwargs"] = {}
                kwargs["audio_kwargs"].setdefault("truncation", False)
            else:
                kwargs["audios"] = audios

        processor = self._processor
        if (
            hasattr(processor, "image_processor")
            and isinstance(processor.image_processor, BaseImageProcessorFast)
            and not self.server_args.disable_fast_image_processor
        ):
            if not _is_npu:
                kwargs["device"] = "cuda"
            elif processor.__class__.__name__ not in {
                "Qwen2_5_VLProcessor",
                "Qwen3VLProcessor",
            }:
                # Note: for qwen-vl, processor has some reshape issue because of dims restriction on Ascend.
                kwargs["device"] = "npu"
        substep1_duration = (time.time() - substep1_start) * 1000
        logger.info(
            f"[MM_PREPROC_DETAIL]     Step 5.1.1 - Prepare kwargs: {substep1_duration:.2f} ms"
        )

        # Step 5.1.2: Call processor.__call__()
        logger.info(f"[DEBUG_TRACE] About to call processor.__call__()")
        substep2_start = time.time()
        try:
            result = processor.__call__(
                text=[input_text],
                padding=True,
                return_tensors="pt",
                **kwargs,
            )
            logger.info(
                f"[DEBUG_TRACE] processor.__call__() completed, result keys: {list(result.keys())}"
            )
        except Exception as e:
            logger.error(
                f"[DEBUG_TRACE] Exception in processor.__call__(): {e}", exc_info=True
            )
            raise
        # end = time.time()
        # logger.info(f"processor call time taken: {end - start}")
        substep2_duration = (time.time() - substep2_start) * 1000
        logger.info(
            f"[MM_PREPROC_DETAIL]     Step 5.1.2 - processor.__call__() execution: {substep2_duration:.2f} ms"
        )

        # Step 5.1.3: Move feature tensors to CPU (if needed)
        substep3_start = time.time()
        if not self.server_args.keep_mm_feature_on_device:
            # move feature tensors to cpu
            for feature_name in self.FEATURE_NAMES:
                if SGL_USE_CUDA_IPC:
                    pass
                else:
                    if feature_name in result and isinstance(
                        result[feature_name], torch.Tensor
                    ):
                        result[feature_name] = result[feature_name].to("cpu")
        substep3_duration = (time.time() - substep3_start) * 1000
        logger.info(
            f"[MM_PREPROC_DETAIL]     Step 5.1.3 - Move tensors to CPU: {substep3_duration:.2f} ms"
        )

        return result

    @abstractmethod
    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        pass

    def get_estimated_frames_list(self, image_data):
        """
        estimate the total frame count from all visual input
        """
        # Lazy import because decord is not available on some arm platforms.
        from decord import VideoReader, cpu

        # Before processing inputs
        if not image_data or len(image_data) == 0:
            return []
        estimated_frames_list = []
        for image in image_data:
            if isinstance(image, str) and image.startswith("video:"):
                path = image[len("video:") :]
                # Estimate frames for the video
                vr = VideoReader(path, ctx=cpu(0))
                num_frames = len(vr)
            else:
                # For images, each contributes one frame
                num_frames = 1
            estimated_frames_list.append(num_frames)

        return estimated_frames_list

    @staticmethod
    def _load_single_item(
        data,
        modality: Modality,
        frame_count_limit=None,
        audio_sample_rate: Optional[int] = None,
        discard_alpha_channel=True,
    ):
        """
        Load a single multimodal data.

        If data is precomputed, returns directly.

        Static method that can be pickled for multiprocessing"""
        if isinstance(data, dict):
            return data
        try:
            if modality == Modality.IMAGE:
                img_tensor, _ = load_image_tensor(data, discard_alpha_channel)
                img_tensor = img_tensor.to("cuda")
                return img_tensor
            elif modality == Modality.VIDEO:
                return load_video(data, frame_count_limit)
            elif modality == Modality.AUDIO:
                return load_audio(data, audio_sample_rate)

        except Exception as e:
            raise RuntimeError(f"Error while loading data {data}: {e}")

    def submit_data_loading_tasks(
        self,
        text_parts: List[str],
        multimodal_tokens: MultimodalSpecialTokens,
        data_iterators: dict[Modality, Iterator[Any]],
        discard_alpha_channel: bool = True,
        image_estimated_frames_iter: Optional[iter] = None,
        image_scaling_factor: float = 1.0,
        max_image_frames: int = 30,
        audio_sample_rate: Optional[int] = None,
    ) -> Tuple[List, List]:
        """
        load multimodal data parallelly using iterators.
        """
        logger.info(
            f"[DEBUG_TRACE]   → submit_data_loading_tasks 开始: text_parts={len(text_parts)}, data_iterators={list(data_iterators.keys())}"
        )
        logger.info(f"[DEBUG_TRACE]   , text_parts: {text_parts}")
        futures = []
        task_info = []

        for idx, text_part in enumerate(text_parts):
            logger.info(f"[DEBUG_TRACE]   处理 text_part[{idx}]: {text_part}")
            modality = multimodal_tokens.get_modality_of_token(text_part)
            if modality is not None:
                logger.info(
                    f"[DEBUG_TRACE]     处理 text_part[{idx}]: modality={modality}"
                )
                data_iterator = data_iterators.get(modality)
                if data_iterator is None:
                    logger.error(
                        f"[DEBUG_TRACE] ✗ 没有找到 {modality} 的 data_iterator"
                    )
                    raise ValueError(f"No data iterator found for token: {text_part}")

                try:
                    data = next(data_iterator)
                except StopIteration:
                    logger.error(
                        f"[DEBUG_TRACE] ✗ data_iterator 耗尽: modality={modality}, text_part={text_part}"
                    )
                    raise ValueError(
                        f"Mismatch: More '{text_part}' tokens found than corresponding data items provided."
                    )

                frame_count_limit = None
                if modality == Modality.IMAGE and image_estimated_frames_iter:
                    try:
                        estimated_frames = next(image_estimated_frames_iter)
                        # Use the pre-calculated scaling factor and max frames
                        frame_count_limit = max(
                            1, int(estimated_frames * image_scaling_factor)
                        )
                        # Ensure we don't exceed the absolute max (redundant if scaling_factor handles it)
                        # frame_count_limit = min(frame_count_limit, max_image_frames)
                    except StopIteration:
                        logger.error(
                            f"[DEBUG_TRACE] ✗ image_estimated_frames_iter 耗尽"
                        )
                        raise ValueError(
                            "Mismatch between image tokens and estimated frame counts."
                        )

                futures.append(
                    self.io_executor.submit(
                        BaseMultimodalProcessor._load_single_item,
                        data,
                        modality,
                        frame_count_limit,
                        audio_sample_rate,
                        discard_alpha_channel,
                    )
                )
                task_info.append((modality, data, frame_count_limit))
                if idx < 5:
                    logger.info(
                        f"[DEBUG_TRACE]     ✓ 已提交 future[{len(futures)-1}] for {modality}"
                    )

        logger.info(f"[DEBUG_TRACE]   → 检查剩余数据项")
        for modality, iterator in data_iterators.items():
            try:
                next(iterator)
                logger.warning(
                    f"Warning: More {modality.name.lower()} data items provided than corresponding tokens found in the prompt."
                )
            except StopIteration:
                pass
            except Exception:
                pass

        logger.info(
            f"[DEBUG_TRACE]   ✓ submit_data_loading_tasks 完成: 返回 {len(futures)} futures, {len(task_info)} task_info"
        )
        return futures, task_info

    def load_mm_data(
        self,
        prompt: str,
        multimodal_tokens: MultimodalSpecialTokens,
        image_data: Optional[list] = None,
        video_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        return_text: Optional[bool] = True,
        discard_alpha_channel: bool = True,
        audio_sample_rate: Optional[int] = None,
    ) -> BaseMultiModalProcessorOutput:
        """
        Each frame of video/image will be replaced by a single image token

        Args:
            multimodal_tokens (list[str]): list of special token which denoting a single multimodal data
                e.g. image token or audio token
            discard_alpha_channel: if True, discards the alpha channel in the returned images

        """
        import time

        overall_start = time.time()

        # Step 1: Parse prompt and prepare iterators
        step1_start = time.time()
        multimodal_tokens_pattern = multimodal_tokens.get_combined_regex()

        if isinstance(prompt, list) and return_text:
            assert len(prompt) and isinstance(prompt[0], int)
            prompt = self._processor.tokenizer.decode(prompt)
        else:
            prompt = prompt

        assert isinstance(prompt, str)
        # split text into list of normal text and special tokens
        text_parts = re.split(multimodal_tokens_pattern, prompt)

        # collect all data
        data_iterators = {}
        if multimodal_tokens.image_token and image_data:
            data_iterators[Modality.IMAGE] = iter(image_data)
        if multimodal_tokens.video_token and video_data:
            data_iterators[Modality.VIDEO] = iter(video_data)
        if multimodal_tokens.audio_token and audio_data:
            data_iterators[Modality.AUDIO] = iter(audio_data)
        step1_duration = (time.time() - step1_start) * 1000
        logger.info(
            f"[MM_PREPROC_DETAIL] Step 1 - Prompt parsing: {step1_duration:.2f} ms"
        )
        logger.info(
            f"[DEBUG_TRACE] ✓ Step 1 完成: text_parts={len(text_parts)}, data_iterators={list(data_iterators.keys())}"
        )

        # Step 2: Submit data loading tasks (parallel I/O)
        logger.info(f"[DEBUG_TRACE] → 即将开始 Step 2 - submit_data_loading_tasks")
        step2_start = time.time()
        try:
            futures, task_info = self.submit_data_loading_tasks(
                text_parts=text_parts,
                multimodal_tokens=multimodal_tokens,
                data_iterators=data_iterators,
                discard_alpha_channel=discard_alpha_channel,
                audio_sample_rate=audio_sample_rate,
            )
            logger.info(
                f"[DEBUG_TRACE] ✓ submit_data_loading_tasks 完成: futures={len(futures)}, task_info={len(task_info)}"
            )
        except Exception as e:
            logger.error(
                f"[DEBUG_TRACE] ✗ submit_data_loading_tasks 异常: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise
        step2_duration = (time.time() - step2_start) * 1000
        logger.info(
            f"[MM_PREPROC_DETAIL] Step 2 - Submit I/O tasks ({len(futures)} files): {step2_duration:.2f} ms"
        )

        logger.info(
            f"[DEBUG_TRACE] → 创建迭代器: task_info={len(task_info)}, futures={len(futures)}"
        )
        task_info_iter = iter(task_info)
        futures_iter = iter(futures)
        logger.info(f"[DEBUG_TRACE] ✓ 迭代器创建完成")

        # Step 3: Collect I/O results
        logger.info(f"[DEBUG_TRACE] → 即将开始 Step 3 - 收集 I/O 结果")
        step3_start = time.time()
        images, videos, audios = [], [], []
        new_text_parts = []
        logger.info(f"[DEBUG_TRACE] → 开始遍历 {len(text_parts)} 个 text_parts")
        for idx, text_part in enumerate(text_parts):
            if idx % 10 == 0:  # Print every 10 iterations to avoid log spam
                logger.info(
                    f"[DEBUG_TRACE]   处理中 text_part [{idx}/{len(text_parts)}]"
                )
            try:
                if multimodal_tokens_pattern.match(text_part):
                    modality, raw_data, frame_limit = next(task_info_iter)
                    is_precomputed = isinstance(raw_data, dict)
                    result = next(futures_iter).result()

                    if modality == Modality.IMAGE:
                        # If data is already processed it will be a
                        # dictionary(precomputed). In this case we want to keep the
                        # expanded tokens in text_part. Otherwise, we will
                        # call the processor code, so keep only a single image
                        # token.
                        mm_tokens = (
                            text_part
                            if is_precomputed
                            else multimodal_tokens.image_token
                        )
                        frames = [result] if not isinstance(result, list) else result
                        if frames:
                            # only for minicpmv
                            images += frames
                            new_text_parts += mm_tokens * len(frames)
                    elif modality == Modality.VIDEO:
                        # load as video
                        mm_tokens = (
                            text_part
                            if is_precomputed
                            else multimodal_tokens.video_token
                        )
                        videos += [result]
                        new_text_parts += mm_tokens
                    elif modality == Modality.AUDIO:
                        # audio
                        mm_tokens = (
                            text_part
                            if is_precomputed
                            else multimodal_tokens.audio_token
                        )
                        audios += [result]
                        new_text_parts += mm_tokens
                else:
                    # normal text
                    new_text_parts += [text_part]

            except Exception as e:
                logger.error(
                    f"[DEBUG_TRACE] ✗ text_part[{idx}] 处理异常: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                raise RuntimeError(
                    f"An exception occurred while loading multimodal data: {e}"
                )

        logger.info(
            f"[DEBUG_TRACE] ✓ 遍历完成: images={len(images)}, videos={len(videos)}, audios={len(audios)}"
        )
        step3_duration = (time.time() - step3_start) * 1000
        logger.info(
            f"[MM_PREPROC_DETAIL] Step 3 - Collect I/O results: {step3_duration:.2f} ms"
        )

        logger.info(f"[DEBUG_TRACE] → 计算总耗时")
        overall_duration = (time.time() - overall_start) * 1000
        total_steps = step1_duration + step2_duration + step3_duration
        logger.info(
            f"[MM_PREPROC_DETAIL] === load_mm_data TOTAL: {overall_duration:.2f} ms (sum of steps: {total_steps:.2f} ms) ==="
        )

        logger.info(f"[DEBUG_TRACE] → 创建 BaseMultiModalProcessorOutput 对象")
        result = BaseMultiModalProcessorOutput(
            images=images,
            audios=audios,
            videos=videos,
            input_text="".join(new_text_parts),
        )
        logger.info(f"[DEBUG_TRACE] load_mm_data returning successfully")
        return result

    @staticmethod
    def get_mm_items_offset(
        input_ids: torch.Tensor, mm_token_id: int
    ) -> List[Tuple[int, int]]:
        """
        Get a set of range for mm_items from input_ids
        Example:
            input_ids = [1, 2, 3, 3, 3, 4, 3, 3]
            mm_token_id = 3
            return result = [(2,4),(6,7)]
        """
        mask = input_ids == mm_token_id
        start_positions = (mask & ~torch.roll(mask, 1)).nonzero(as_tuple=True)[0]
        end_positions = (mask & ~torch.roll(mask, -1)).nonzero(as_tuple=True)[0]

        return list(zip(start_positions.tolist(), end_positions.tolist()))

    @staticmethod
    def get_mm_items_offset_by_pair(
        input_ids: torch.Tensor, mm_start_id: int, mm_end_id: int
    ) -> List[Tuple[int, int]]:
        indices_start = (input_ids == mm_start_id).nonzero(as_tuple=True)[0] + 1
        indices_end = (input_ids == mm_end_id).nonzero(as_tuple=True)[0] - 1

        return list(zip(indices_start.tolist(), indices_end.tolist()))

    def collect_mm_items_from_processor_output(
        self, data_dict: dict
    ) -> List[MultimodalDataItem]:
        """Create mm_items directly from processor output."""
        items: dict[Modality, MultimodalDataItem] = {}
        for attr_name, value in data_dict.items():
            if attr_name == "input_ids":
                continue

            # Get modality for this attribute
            modality = self.ATTR_NAME_TO_MODALITY.get(attr_name)

            if attr_name == "precomputed_embeddings":
                modality_str = data_dict.get("modality")
                modality = Modality.IMAGE
                if modality_str:
                    try:
                        modality = Modality.from_str(modality_str)
                    except ValueError:
                        pass

            if modality:
                # Create item if needed
                if modality not in items:
                    items[modality] = MultimodalDataItem(modality=modality)

                if attr_name in self.FEATURE_NAMES:
                    attr_name = "feature"

                items[modality].set(attr_name, value)

        return list(items.values())

    def _process_and_collect_mm_items(
        self, input_text: str, images=None, audios=None, videos=None, **kwargs
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """
        Helper method to process multimodal data and create mm_items in one step.

        Returns:
            Tuple of (created mm_items, input_ids)
        """
        import time

        logger.info(
            f"[DEBUG_TRACE] Entering _process_and_collect_mm_items with {len(images) if images else 0} images, {len(audios) if audios else 0} audios, {len(videos) if videos else 0} videos"
        )

        # Step 5.1: Call HF processor
        logger.info(f"[DEBUG_TRACE] About to call process_mm_data")
        substep1_start = time.time()
        try:
            ret = self.process_mm_data(
                input_text=input_text,
                images=images,
                audios=audios,
                videos=videos,
                **kwargs,
            )
            logger.info(f"[DEBUG_TRACE] process_mm_data returned successfully")
        except Exception as e:
            logger.error(
                f"[DEBUG_TRACE] Exception in process_mm_data: {e}", exc_info=True
            )
            raise
        substep1_duration = (time.time() - substep1_start) * 1000
        logger.info(
            f"[MM_PREPROC_DETAIL]   Step 5.1 - HF processor __call__: {substep1_duration:.2f} ms"
        )

        # Step 5.2: Extract input_ids
        logger.info(f"[DEBUG_TRACE] About to extract input_ids")
        substep2_start = time.time()
        try:
            input_ids = ret["input_ids"].flatten()
            logger.info(
                f"[DEBUG_TRACE] Extracted input_ids with shape: {input_ids.shape}"
            )
        except Exception as e:
            logger.error(
                f"[DEBUG_TRACE] Exception extracting input_ids: {e}", exc_info=True
            )
            raise
        substep2_duration = (time.time() - substep2_start) * 1000
        logger.info(
            f"[MM_PREPROC_DETAIL]   Step 5.2 - Extract input_ids: {substep2_duration:.2f} ms"
        )

        # Step 5.3: Collect mm_items from processor output
        logger.info(f"[DEBUG_TRACE] About to collect mm_items from processor output")
        substep3_start = time.time()
        try:
            collected_items = self.collect_mm_items_from_processor_output(ret)
            logger.info(f"[DEBUG_TRACE] Collected {len(collected_items)} mm_items")
        except Exception as e:
            logger.error(
                f"[DEBUG_TRACE] Exception collecting mm_items: {e}", exc_info=True
            )
            raise
        substep3_duration = (time.time() - substep3_start) * 1000
        logger.info(
            f"[MM_PREPROC_DETAIL]   Step 5.3 - Collect mm_items: {substep3_duration:.2f} ms"
        )

        logger.info(
            f"[DEBUG_TRACE] _process_and_collect_mm_items returning successfully"
        )
        return collected_items, input_ids, ret

    def process_and_combine_mm_data(
        self,
        base_output: BaseMultiModalProcessorOutput,
        mm_tokens: MultimodalSpecialTokens,
        **kwargs,
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """
        Process multimodal data and return the combined multimodal items and input_ids.
        Supports mixed modalities (images and audio in the same request).

        Returns:
            Tuple of (list of mm_items, input_ids)
        """
        import time

        logger.info(f"[DEBUG_TRACE] Entering process_and_combine_mm_data")
        overall_start = time.time()

        # Step 1: Categorize items
        logger.info(f"[DEBUG_TRACE] Step 4 - About to organize results")
        step1_start = time.time()
        try:
            all_items = base_output.organize_results()
            logger.info(f"[DEBUG_TRACE] Organized {len(all_items)} items")
        except Exception as e:
            logger.error(
                f"[DEBUG_TRACE] Exception in organize_results: {e}", exc_info=True
            )
            raise
        # Handle text-only case
        if not all_items:
            logger.info(f"[DEBUG_TRACE] Text-only case, no multimodal items")
            input_ids = self._processor.tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()
            logger.info(f"[DEBUG_TRACE] Returning early for text-only")
            return [], input_ids, {}

        logger.info(f"[DEBUG_TRACE] Categorizing {len(all_items)} items by type")
        dict_items, raw_images, raw_audios, raw_videos = [], [], [], []
        for idx, (modality, item) in enumerate(all_items):
            if idx % 10 == 0:
                logger.info(f"[DEBUG_TRACE] Categorizing item {idx}/{len(all_items)}")
            if isinstance(item, dict):
                dict_items.append(item)
            elif modality == Modality.IMAGE:
                raw_images.append(item)
            elif modality == Modality.AUDIO:
                raw_audios.append(item)
            elif modality == Modality.VIDEO:
                raw_videos.append(item)
            else:
                raise ValueError(f"Unknown multimodal item type: {type(item)}")
        step1_duration = (time.time() - step1_start) * 1000
        logger.info(
            f"[MM_PREPROC_DETAIL] Step 4 - Categorize items: {step1_duration:.2f} ms"
        )
        logger.info(
            f"[DEBUG_TRACE] Categorized: {len(dict_items)} dict, {len(raw_images)} images, {len(raw_audios)} audios, {len(raw_videos)} videos"
        )

        # Step 2: Process raw items (HF processor call)
        all_collected_items: list[MultimodalDataItem] = []
        input_ids = None
        step2_duration = (
            0.0  # Initialize to handle case where no raw items need processing
        )

        if raw_images or raw_audios or raw_videos:
            logger.info(
                f"[DEBUG_TRACE] Step 5 - About to call _process_and_collect_mm_items"
            )
            step2_start = time.time()
            try:
                collected_items, input_ids, ret = self._process_and_collect_mm_items(
                    input_text=base_output.input_text,
                    images=raw_images,
                    audios=raw_audios,
                    videos=raw_videos,
                    **kwargs,
                )
                logger.info(
                    f"[DEBUG_TRACE] _process_and_collect_mm_items returned {len(collected_items)} items"
                )
            except Exception as e:
                logger.error(
                    f"[DEBUG_TRACE] Exception in _process_and_collect_mm_items: {e}",
                    exc_info=True,
                )
                raise
            step2_duration = (time.time() - step2_start) * 1000
            logger.info(
                f"[MM_PREPROC_DETAIL] Step 5 - HF processor call: {step2_duration:.2f} ms"
            )
            all_collected_items = collected_items
        else:
            logger.info(
                f"[DEBUG_TRACE] No raw items to process, skipping HF processor call"
            )
            ret = None

        # Handle dict items (already processed)
        for dict_item in dict_items:
            all_collected_items.extend(
                self.collect_mm_items_from_processor_output(dict_item)
            )

        # Fallback tokenization if no raw items were processed
        if input_ids is None:
            input_ids = self._processor.tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()

        # Add offsets to all items
        for mm_item in all_collected_items:
            mm_token_id = mm_tokens.get_token_id_by_modality(mm_item.modality)
            if mm_token_id is None:
                raise ValueError(f"No token id found for modality: {mm_item.modality}")
            mm_item.offsets = self.get_mm_items_offset(
                input_ids=input_ids,
                mm_token_id=mm_token_id,
            )

        """
        solution for cuda-ipc memory-leak:
        1. memory-pool:  each time get a slice from memory-pool and use it as transport-data (with async lock guard)
        2. if can not get a slice , transport normal tensor
        3. copy tensor in scheduler and release it (use position mark)
        4. copy
        """

        if SGL_USE_CUDA_IPC:
            # post-process
            for item in all_collected_items:
                if isinstance(item.feature, torch.Tensor) and item.feature.is_cuda:
                    sync_flag, available_slice = (
                        self.cudaipc_mmfeature_pool.return_a_slice_tensor_with_flag(
                            item.feature
                        )
                    )
                    if isinstance(available_slice, torch.Tensor):
                        available_slice.copy_(
                            item.feature.view(torch.int8).view(-1), non_blocking=True
                        )
                        item.feature = CudaIpcTensorTransportProxy(
                            data=available_slice,
                            info_data=item.feature,
                            sync_buffer_meta=sync_flag,
                        )
                elif (
                    isinstance(item.precomputed_embeddings, torch.Tensor)
                    and item.precomputed_embeddings.is_cuda
                ):
                    sync_flag, available_slice = (
                        self.cudaipc_mmfeature_pool.return_a_slice_tensor_with_flag(
                            item.precomputed_embeddings
                        )
                    )
                    if isinstance(available_slice, torch.Tensor):
                        available_slice.copy_(
                            item.precomputed_embeddings.view(torch.int8).view(-1),
                            non_blocking=True,
                        )
                        item.precomputed_embeddings = CudaIpcTensorTransportProxy(
                            data=available_slice,
                            info_data=item.precomputed_embeddings,
                            sync_buffer_meta=sync_flag,
                        )

        logger.info(
            f"[DEBUG_TRACE] About to finalize and return from process_and_combine_mm_data"
        )
        overall_duration = (time.time() - overall_start) * 1000
        # Calculate sum of steps
        total_steps = step1_duration + step2_duration
        logger.info(
            f"[MM_PREPROC_DETAIL] === process_and_combine_mm_data TOTAL: {overall_duration:.2f} ms (sum of steps: {total_steps:.2f} ms) ==="
        )

        logger.info(
            f"[DEBUG_TRACE] process_and_combine_mm_data returning successfully with {len(all_collected_items)} items"
        )
        return all_collected_items, input_ids, ret
