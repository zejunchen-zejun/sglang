# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group
from sglang.srt.utils import get_bool_env_var
_use_aiter = get_bool_env_var("SGLANG_USE_AITER")


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)

if _use_aiter:
    def tensor_model_parallel_fused_allreduce_rmsnorm(
        input_: torch.Tensor, residual_inp_: torch.Tensor, weight_: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return get_tp_group().fused_allreduce_rmsnorm(input_, residual_inp_, weight_, eps)


    def tensor_model_parallel_fused_allreduce_rmsnorm_quant(
        input_: torch.Tensor, residual_inp_: torch.Tensor, weight_: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return get_tp_group().fused_allreduce_rmsnorm_quant(input_, residual_inp_, weight_, eps)