# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper around `atom` models."""
import logging
from typing import Any, Iterable, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

logger = logging.getLogger(__name__)

_DEEPSEEK_ARCHS = {
    "DeepseekV3ForCausalLM",
}


class ATOMForCausalLM(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        logger.info("Using Atom backend.")

        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.unpadded_vocab_size = config.vocab_size

        import atom

        self.model = atom.prepare_model(config=config, engine="sglang")
        if self.model is None:
            model_arch = getattr(config, "architectures", ["unknown"])[0]
            raise ValueError(f'This model{model_arch} is not supported by atom')

        self.logits_processor = LogitsProcessor(config)
        arch = getattr(config, "architectures", [""])[0]
        self._uses_forward_batch_context = arch in _DEEPSEEK_ARCHS
        if self._uses_forward_batch_context:
            from atom.plugin.sglang.attention_backend.sgl_attention_mla import (
                setup_deepseek_for_sglang,
            )

            setup_deepseek_for_sglang(self.model)


    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **model_kwargs: Any,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        model_inputs = dict(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=pp_proxy_tensors,
            inputs_embeds=input_embeds,
        )

        if self._uses_forward_batch_context:
            from atom.plugin.sglang.models import base_model_wrapper as atom_base_wrapper

            token = atom_base_wrapper._current_forward_batch.set(forward_batch)
            try:
                hidden_states = self.model(**model_inputs)
            finally:
                atom_base_wrapper._current_forward_batch.reset(token)
        else:
            hidden_states = self.model(
                **model_inputs,
                forward_batch=forward_batch,
                get_embedding=get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
                **model_kwargs,
            )

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.model.lm_head, forward_batch
            )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        if hasattr(self.model, "load_weights"):
            return self.model.load_weights(weights)

        from atom.model_loader.loader import load_model_in_plugin_mode

        return load_model_in_plugin_mode(
            model=self.model, config=self.model.atom_config, prefix="model."
        )

EntryClass = [ATOMForCausalLM]
