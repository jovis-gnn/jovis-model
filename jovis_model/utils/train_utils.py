import functools

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.roberta.modeling_roberta import RobertaEncoder


fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
    cast_forward_inputs=True,
)

bfSixteen_mixed = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)


def get_klue_wrapper():
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={RobertaEncoder},
    )
    return auto_wrap_policy


def get_policies(config):
    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    mixed_precision_policy = None
    wrapping_policy = None

    if config.params.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not config.params.use_fp16:
            mixed_precision_policy = bfSixteen
        elif config.params.use_fp16:
            mixed_precision_policy = fpSixteen
    wrapping_policy = get_klue_wrapper()
    return mixed_precision_policy, wrapping_policy


class MemoryTrace:
    def __enter__(self):
        gc.collect()
        if is_xpu_available():
            torch.xpu.empty_cache()
            torch.xpu.reset_max_memory_allocated()
            self.begin = byte2gb(torch.xpu.memory_allocated())
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
