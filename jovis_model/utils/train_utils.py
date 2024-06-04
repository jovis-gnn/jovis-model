import gc
import psutil
import functools
import threading

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.roberta.modeling_roberta import RobertaEncoder

from accelerate.utils import is_xpu_available


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


def generate_peft_config():
    pass


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
    ) or (is_xpu_available())

    mixed_precision_policy = None
    wrapping_policy = None

    if config.params.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready:
            mixed_precision_policy = bfSixteen
        elif config.params.use_fp16:
            mixed_precision_policy = fpSixteen
    wrapping_policy = get_klue_wrapper()
    return mixed_precision_policy, wrapping_policy


def byte2gb(x):
    return int(x / 2**30)


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
            self.begin = byte2gb(torch.cuda.memory_allocated())
        self.process = psutil.Process()
        self.cpu_begin = byte2gb(self.cpu_mem_used())
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        if is_xpu_available():
            torch.xpu.empty_cache()
            self.end = byte2gb(torch.xpu.memory_allocated())
            self.peak = byte2gb(torch.xpu.max_memory_allocated())
            xpu_info = torch.xpu.memory_stats()
            self.peak_active_gb = byte2gb(xpu_info["active_bytes.all.peak"])
            self.malloc_retries = xpu_info.get("num_alloc_retries", 0)
            self.peak_active_gb = byte2gb(xpu_info["active_bytes.all.peak"])
            self.m_ooms = xpu_info.get("num_ooms", 0)
            self.used = byte2gb(self.end - self.begin)
            self.peaked = byte2gb(self.peak - self.begin)
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.end = byte2gb(torch.cuda.memory_allocated())
            self.peak = byte2gb(torch.cuda.max_memory_allocated())
            cuda_info = torch.cuda.memory_stats()
            self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
            self.malloc_retries = cuda_info.get("num_alloc_retries", 0)
            self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
            self.m_ooms = cuda_info.get("num_ooms", 0)
            self.used = byte2gb(self.end - self.begin)
            self.peaked = byte2gb(self.peak - self.begin)
            self.max_reserved = byte2gb(torch.cuda.max_memory_reserved())

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = byte2gb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = byte2gb(self.cpu_peak - self.cpu_begin)

    def print_stats(self):
        device_str = None
        if is_xpu_available():
            device_str = "XPU"
        elif torch.cuda.is_available():
            device_str = "CUDA"

        if device_str:
            print(f"Max {device_str} memory allocated was {self.peak} GB")
            print(f"Max {device_str} memory reserved was {self.max_reserved} GB")
            print(f"Peak active {device_str} memory was {self.peak_active_gb} GB")
            print(f"{device_str} Malloc retries : {self.malloc_retries}")
        print(
            f"CPU Total Peak Memory consumed during the train (max): {self.cpu_peaked + self.cpu_begin} GB"
        )
