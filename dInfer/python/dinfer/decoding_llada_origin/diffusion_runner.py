import torch
import gc
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Callable
import tqdm
import time
from sglang.srt.utils import (
    get_available_gpu_memory,
    get_bool_env_var,
    is_hip,
)
import bisect
from sglang.srt.layers.torchao_utils import save_gemlite_cache
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tp_group,
    set_custom_all_reduce,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator, graph_capture
import logging
from .utils import KVCache
import os
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_rank,
    get_attention_tp_size,
    set_dp_buffer_len,
)
import math
from sglang.srt.custom_op import CustomOp
from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode
from sglang.srt.utils.patch_torch import monkey_patch_torch_compile
_is_hip = is_hip()

logger = logging.getLogger(__name__)
# 假设的上下文管理器，用于在捕获期间冻结垃圾回收
@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    """
    Optimize garbage collection during CUDA graph capture.
    Clean up, then freeze all remaining objects from being included
    in future collections if GC is disabled during capture.
    """
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
            gc.collect()

def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub.leave_torch_compile()
            else:
                sub.enter_torch_compile(num_tokens=num_tokens)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)

def set_torch_compile_config():
    import torch._dynamo.config
    import torch._inductor.config
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
    # FIXME: tmp workaround
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024
    monkey_patch_torch_compile()

@contextmanager
def patch_model(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    """Patch the model to make it compatible with with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            _to_torch(model, reverse=False, num_tokens=num_tokens)
            backup_ca_comm = tp_group.ca_comm
            # Use custom-allreduce here.
            # We found the custom allreduce is much faster than the built-in allreduce in torch,
            # even with ENABLE_INTRA_NODE_COMM=1.
            # tp_group.ca_comm = None
            yield torch.compile(
                torch.no_grad()(model.forward),
                mode=os.environ.get(
                    "SGLANG_TORCH_COMPILE_MODE", "default"
                ),
                dynamic=_is_hip and get_bool_env_var("SGLANG_TORCH_DYNAMIC_SHAPE"),
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True, num_tokens=num_tokens)
            tp_group.ca_comm = backup_ca_comm
            


# Reuse this memory pool across all cuda graph runners.
global_graph_memory_pool = None


def get_global_graph_memory_pool():
    return global_graph_memory_pool


def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val


class ModelRunner:
    def __init__(self, model: torch.nn.Module, device: str = "cuda", enable_cuda_graph: bool = True, supported_batch_sizes: Optional[list] = None, server_args=None,
            max_length=2048, block_length=32, prefill_lengths=[64, 96, 128], decoding_lengths=[32], enable_compile=True, cache_lengths=None, use_cross_block=True):
        self.model = model.to(device)
        device = str(device)
        self.server_args = server_args
        if device.startswith("cuda:"):
            self.device = "cuda"
            self.gpu_id = int(device.split(":")[1])
        else:
            self.device = device
            self.gpu_id = torch.cuda.current_device()
        self.enable_compile = enable_compile
        self.enable_cuda_graph = enable_cuda_graph and (device != "cpu") # disable when device is CPU 
        self.supported_batch_sizes = supported_batch_sizes or [1, ] 
        if cache_lengths is None:
            # we need to make sure the maximum length of buffer >= max_length and align to power of 2
            # we capture cache lengths ranging from 128 to 128 * 2**n, where n ensures that 128 * 2**n >= max_length
            self.max_length = max_length
            n = int(math.log2((self.max_length-1) // 128)) + 1
            cache_lengths = [128 * 2**i for i in range(n+1)]
        self.max_length = max(max_length, max(cache_lengths))

        self.block_length = block_length
        self.prefill_lengths=prefill_lengths
        self.decoding_lengths = decoding_lengths
        self.cache_lengths = cache_lengths
        self.use_cross_block = use_cross_block
        if block_length not in decoding_lengths:
            self.decoding_lengths.append(block_length)

        x = torch.arange(block_length, dtype=torch.long, device=device).unsqueeze(0)
        
        self.model.eval()
        self.tp_group = get_tp_group()
        set_custom_all_reduce(True)

        self.forward_normal(x, use_cache=True)
        self.init_device_graphs()
        

    def init_device_graphs(self):
        """Capture device graphs."""
        self.graph_runner = None
        self.graph_mem_usage = 0

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture {'cpu graph' if self.device == 'cpu' else 'cuda graph'} begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        self.graph_runner = CudaGraphRunner(self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.graph_mem_usage = before_mem - after_mem
        logger.info(
            f"Capture {'cpu graph' if self.device == 'cpu' else 'cuda graph'} end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={self.graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def forward_normal(
        self,
        input_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[torch.Tensor] = None,
        past_key_values=None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        backup_ca_comm = self.tp_group.ca_comm
        if attention_mask is not None and past_key_values is not None:
            attention_mask_partial = torch.zeros((attention_mask.shape[0], attention_mask.shape[1], 
                        past_key_values.shape[4]), dtype=torch.bool, device=attention_mask.device)
            attention_mask_partial[:, :, :attention_mask.shape[2]] = attention_mask
            attention_mask = attention_mask_partial
        ret = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
            past_key_values=past_key_values,
            replace_position=replace_position,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        self.tp_group.ca_comm = backup_ca_comm

        return ret
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[torch.Tensor] = None,
        past_key_values=None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        if isinstance(past_key_values, KVCache):
            past_key_values = past_key_values._data
        
        is_decode_phase = input_ids is not None and use_cache==True and past_key_values is not None
        length = input_ids.shape[1]
        if past_key_values is not None:
            cache_length = past_key_values.shape[4]
        else:
            cache_length = 0

        can_run_graph = bool(
            self.graph_runner
            and self.graph_runner.can_run(input_ids, position_ids, past_key_values, is_decode_phase, length, cache_length)
        )
        if can_run_graph and self.enable_cuda_graph:
            logger.debug('run cuda graph')
            ret = self.graph_runner.replay(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                is_decode_phase=is_decode_phase,
                length=length,
                attention_mask=attention_mask,
                cache_length=cache_length,
            )
            return ret

        logger.debug('run normal')
        ret = self.forward_normal(input_ids, position_ids, inputs_embeds, pp_proxy_tensors, past_key_values, replace_position, use_cache, attention_mask)
        return ret
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
        
        
        
class CudaGraphRunner:
    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.capture_bs, self.compile_bs = model_runner.supported_batch_sizes, model_runner.supported_batch_sizes
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.graphs = {}
        self.output_buffers = {}
        self.disable_padding = False

        self.max_bs = max(self.capture_bs)
        self.seq_len_fill_value = 0
        self.num_tokens_per_bs = self.model_runner.block_length
        self.prefill_lengths = model_runner.prefill_lengths
        self.cache_lengths = model_runner.cache_lengths
        self.decoding_lengths = model_runner.decoding_lengths
        self.max_num_token = self.max_bs * max(self.num_tokens_per_bs*2, max(self.prefill_lengths))
        self.tp_size = get_attention_tp_size()
        self.enable_compile = model_runner.enable_compile
        if self.enable_compile:
            set_torch_compile_config()
        with torch.device(self.device):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.position_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            num_layers = self.model_runner.model.config.num_hidden_layers
            num_kv_heads = self.model_runner.model.config.num_key_value_heads
            num_heads = self.model_runner.model.config.num_attention_heads
            head_dim = self.model_runner.model.config.hidden_size // num_heads
            self.past_key_values = torch.zeros((num_layers, 2, self.max_bs, max(1, num_kv_heads//self.tp_size), self.model_runner.max_length, head_dim), dtype=torch.bfloat16)
            self.attention_mask = torch.ones((self.max_bs, self.model_runner.max_length, self.model_runner.max_length), dtype=torch.bool)
            self.attention_mask[0, 0, 0] = False # make sure self.attention_mask is not all False or all True to avoid potential op select problem
        # Capture
        if self.model_runner.enable_cuda_graph:
            try:
                with model_capture_mode():
                    self.capture()
            except RuntimeError as e:
                raise Exception(
                    f"Capture cuda graph failed: {e}\n"
                )
            
    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with self.device_module.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _create_device_graph(self):
        return torch.cuda.CUDAGraph()
    
    def capture(self) -> None:
        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with freeze_gc(
            self.model_runner.server_args.enable_cudagraph_gc
        ), graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            avail_mem = get_available_gpu_memory(
                self.model_runner.device,
                self.model_runner.gpu_id,
                empty_cache=False,
            )
            # Reverse the order to enable better memory sharing across cuda graphs.
            logger.info('capture batch sizes', self.capture_bs)
            for bs in reversed(self.capture_bs):
                capture_cache_length_range = (
                    tqdm.tqdm(list(reversed(self.cache_lengths)))
                    if get_tensor_model_parallel_rank() == 0
                    else reversed(self.cache_lengths)
                )
                for cache_length in capture_cache_length_range:
                    length = self.model_runner.block_length
                    if get_tensor_model_parallel_rank() == 0:
                        avail_mem = get_available_gpu_memory(
                            self.model_runner.device,
                            self.model_runner.gpu_id,
                            empty_cache=False,
                        )
                        capture_cache_length_range.set_description(
                            f"Capturing batches ({bs=} {length=} {cache_length=} {avail_mem=:.2f} GB)"
                        )

                    with patch_model(
                        self.model_runner.model,
                        bs in self.compile_bs and self.enable_compile,
                        num_tokens=bs * length,
                        tp_group=self.model_runner.tp_group,
                    ) as forward:
                        (
                            graph,
                            output_buffers,
                        ) = self.capture_one_batch_size(bs, forward, True, length, cache_length)
                        self.graphs[(bs, True, length, cache_length)] = graph
                        self.output_buffers[(bs, True, length, cache_length)] = output_buffers

                    # Save gemlite cache after each capture
                    save_gemlite_cache()

                    if self.model_runner.use_cross_block:
                        length = self.model_runner.block_length*2
                        if get_tensor_model_parallel_rank() == 0:
                            avail_mem = get_available_gpu_memory(
                                self.model_runner.device,
                                self.model_runner.gpu_id,
                                empty_cache=False,
                            )
                            capture_cache_length_range.set_description(
                                f"Capturing batches ({bs=} {length=} {cache_length=} {avail_mem=:.2f} GB)"
                            )

                        with patch_model(
                            self.model_runner.model,
                            bs in self.compile_bs and self.enable_compile,
                            num_tokens=bs * length,
                            tp_group=self.model_runner.tp_group,
                        ) as forward:
                            (
                                graph,
                                output_buffers,
                            ) = self.capture_one_batch_size(bs, forward, True, length, cache_length, use_mask = True)
                            self.graphs[(bs, True, length, cache_length)] = graph
                            self.output_buffers[(bs, True, length, cache_length)] = output_buffers

                        # Save gemlite cache after each capture
                        save_gemlite_cache()
            print('finished capturing decode')

            capture_prefilling_range = (
                tqdm.tqdm(list(self.prefill_lengths))
                if get_tensor_model_parallel_rank() == 0
                else self.prefill_lengths
            )
            # we only need to capture the max (mini) batch size for prefilling, only the last batch may contain one
            # other number of sequences and need a different prefilling batch size, we don't need to capture it
            bs = max(self.capture_bs)
            for length in capture_prefilling_range:
                if get_tensor_model_parallel_rank() == 0:
                    avail_mem = get_available_gpu_memory(
                        self.model_runner.device,
                        self.model_runner.gpu_id,
                        empty_cache=False,
                    )
                    capture_prefilling_range.set_description(
                        f"Capturing prefilling batches ({bs=} {length=} {avail_mem=:.2f} GB)"
                        )
                with patch_model(
                    self.model_runner.model,
                    False,
                    num_tokens=bs * length,
                    tp_group=self.model_runner.tp_group,
                ) as forward:
                    (
                        graph,
                        output_buffers,
                    ) = self.capture_one_batch_size(bs, forward, False, length, use_mask=True)
                    self.graphs[(bs, False, length, 0)] = graph
                    self.output_buffers[(bs, False, length, 0)] = output_buffers

                # Save gemlite cache after each capture
                save_gemlite_cache()

    def capture_one_batch_size(self, bs: int, forward: Callable, is_decode_phase:bool=True, length:int=0, cache_length:int=0, use_mask:bool=False):
        graph = self._create_device_graph()
        stream = self.stream
        num_tokens_per_bs = length
        num_tokens = bs * num_tokens_per_bs

        # Graph inputs
        input_ids = self.input_ids[:num_tokens].view(bs, num_tokens_per_bs)
        position_ids = self.position_ids[:num_tokens].view(bs, num_tokens_per_bs)
        # print(input_ids.shape)
        if is_decode_phase:
            past_key_values = self.past_key_values[:, :, :bs, :, :cache_length]
        else:
            past_key_values=None
        
        attn_mask=None
        if use_mask:
            if not is_decode_phase:
                attn_mask = self.attention_mask[:bs, :num_tokens_per_bs, :num_tokens_per_bs]
            else:
                attn_mask = self.attention_mask[:bs, :num_tokens_per_bs, :cache_length]




        # Run and capture
        def run_once():
            # print('run once', input_ids.shape, position_ids.shape, past_key_values.shape)
            logits_output = forward(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=None,
                pp_proxy_tensors=None,
                past_key_values=past_key_values,
                replace_position=(0, 0),
                use_cache=True,
                attention_mask=attn_mask,
            )
            return logits_output

        for _ in range(2):
            self.device_module.synchronize()
            self.model_runner.tp_group.barrier()
            run_once()

        if get_global_graph_memory_pool() is None:
            set_global_graph_memory_pool(self.device_module.graph_pool_handle())
        # Set graph pool id globally to be able to use symmetric memory
        set_graph_pool_id(get_global_graph_memory_pool())
        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )

        return graph, out

    def can_run(self, input_ids, position_ids, past_key_values, is_decode_phase=True, length=0, cache_length=0):
        cuda_graph_bs = input_ids.shape[0]
        is_bs_supported = (cuda_graph_bs, is_decode_phase, length, cache_length) in self.graphs.keys()
        if not self.disable_padding and is_decode_phase:
            is_bs_supported = is_bs_supported or (self.max_bs, is_decode_phase, length, cache_length) in self.graphs.keys()
        if is_bs_supported == False:
            logger.debug('not supported', cuda_graph_bs, is_decode_phase, length, cache_length, self.graphs.keys())
        return is_bs_supported


    def replay_prepare(self, input_ids, position_ids, past_key_values, is_decode_phase, length, attention_mask, cache_length):
        raw_bs = input_ids.shape[0]
        raw_num_token = raw_bs * length

        # 查找最接近的支持的 bs
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]

        # 拷贝真实数据到静态 buffer
        self.input_ids[:raw_num_token].copy_(input_ids.flatten())
        self.position_ids[:raw_num_token].copy_(position_ids.flatten())
        if is_decode_phase:
            minimal_length = min(self.past_key_values.shape[4], past_key_values.shape[4], cache_length)
            self.past_key_values[:, :, :, :, minimal_length:].fill_(0)
            self.past_key_values[:, :, :raw_bs, :, :minimal_length].copy_(past_key_values[:, :, :, :, :minimal_length])
        if attention_mask is not None:
            minimal_length = min(self.attention_mask.shape[2], attention_mask.shape[2])
            self.attention_mask[:raw_bs, :length, :minimal_length].copy_(attention_mask[:, :, :minimal_length])
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs
        self.is_decode_phase = is_decode_phase
        self.length = length


    def replay(
        self, input_ids, position_ids, past_key_values, is_decode_phase, length, attention_mask, cache_length
    ):
        self.replay_prepare(input_ids, position_ids, past_key_values, is_decode_phase, length, attention_mask, cache_length)

        # Replay
        self.graphs[(self.bs, is_decode_phase, length, cache_length)].replay()

        output = self.output_buffers[(self.bs, is_decode_phase, length, cache_length)]
        return output