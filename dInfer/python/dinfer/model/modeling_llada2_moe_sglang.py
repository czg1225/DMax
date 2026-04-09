# coding=utf-8
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
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
""" SGLang LLaDA2 model."""
import logging
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
import tqdm
from pathlib import Path
import json
import re
from safetensors.torch import load_file
import torch.distributed as dist

import sglang.srt.distributed as sglang_distributed
import traceback
def torch_all_reduce(tensor):
    torch.distributed.all_reduce(tensor)
    return tensor
sglang_distributed.tensor_model_parallel_all_reduce = torch_all_reduce
import numpy as np

def torch_all_gather(input_: torch.Tensor) -> torch.Tensor:
    # trace_stack = traceback.format_stack()
    # for i, frame in enumerate(trace_stack):
    #     print(frame)
    # print()
    # exit()
    # world_size = sglang_distributed.get_tensor_model_parallel_world_size()
    # # 2. 给每个 rank 预分配输出缓冲区
    # output_list = [torch.empty_like(tensor) for _ in range(world_size)]
    # # 3. 集合通信
    # torch.distributed.all_gather(output_list, tensor)
    # # 4. 在 0 维拼接，和 SGLang 原生行为保持一致
    # return torch.cat(output_list, dim=0)
    input_size = input_.size()
    world_size = sglang_distributed.get_tensor_model_parallel_world_size()
    output_size = (input_size[0] * world_size,) + input_size[1:]
    # Allocate output tensor.
    output_tensor = torch.empty(
        output_size, dtype=input_.dtype, device=input_.device
    )

    torch.distributed.all_gather_into_tensor(
        output_tensor, input_
    )
    return output_tensor

# 5. 原地 monkey-patch
sglang_distributed.tensor_model_parallel_all_gather = torch_all_gather

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    parallel_state, divide,
    tensor_model_parallel_all_reduce,
    get_moe_expert_parallel_world_size,
    get_moe_expert_parallel_rank,
    get_moe_tensor_parallel_world_size,
    get_moe_tensor_parallel_rank,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
# from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_deepep_mode
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import (
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.utils import add_prefix, is_cuda, is_non_idle_and_non_empty, make_layers
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.normalized_shape = tuple((hidden_size,))

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x

            x =  F.rms_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.variance_epsilon)
            return x.to(input_dtype), residual
        out =  F.rms_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.variance_epsilon)
        return out.to(input_dtype)

class LLaDA2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LLaDA2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

LoraConfig = None
logger = logging.getLogger(__name__)
_is_cuda = is_cuda()


def _all_gather_cat(
    tensor: torch.Tensor,
    dim: int = 1,
    group: Optional[dist.ProcessGroup] = None,
    normal_len: int = 0,
    last_len: int = 0,
) -> torch.Tensor:
    """
    Gather tensors along `dim` from all ranks and concatenate them.
    Only the last chunk may be shorter than `normal_len`; all others are exactly `normal_len`.

    Args:
        tensor: local tensor on current rank
        dim: dimension along which to concatenate
        normal_len: length of the first (world_size-1) ranks along `dim`
        last_len: length of the last rank along `dim`

    Returns:
        Concatenated tensor of shape [total_len, ...] along `dim`
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if world_size == 1:
        return tensor

    # 1. Move the concatenation dimension to 0 for easier all_gather
    tensor = tensor.movedim(dim, 0)          # [L_local, ...]
    L_local = tensor.size(0)

    # 2. Compute global length across all ranks
    total_len = normal_len * (world_size - 1) + last_len

    # 3. Pre-allocate receive buffers (same shape for all ranks, sized for the largest chunk)
    max_len = max(normal_len, last_len)
    gather_list = [
        torch.empty([max_len] + list(tensor.shape[1:]),
                   dtype=tensor.dtype,
                   device=tensor.device)
        for _ in range(world_size)
    ]

    # 4. Copy local data into the corresponding buffer (only first L_local rows are valid)
    gather_list[rank][:L_local] = tensor

    # 5. All-gather (communicate only valid parts)
    dist.all_gather(gather_list, gather_list[rank], group=group)

    # 6. Trim padding and concatenate
    gathered = torch.cat(gather_list, dim=0)[:total_len]

    # 7. Move dimension back to original position
    return gathered.movedim(0, dim)


class H2Embed:
    def __init__(self, embedding: nn.Embedding, tau: float = 1.0):
        """
        W_e : token embedding weights [V, d]
        tau : temperature; lower values yield sharper distributions
        """
        self.embedding = embedding
        self.W_e = embedding.weight
        self.tau = tau
        self.sp_size = get_tensor_model_parallel_world_size()  # no sequence parallel by default

    def __call__(
        self,
        x: torch.Tensor,
        mask_index: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        iter_cont_weight: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L] token ids
            mask_index: [B, L] bool tensor, True where continuous embedding should be used
            logits: [B, L, V] logits used to produce continuous embeddings
            iter_cont_weight: blending weight between continuous and discrete embeddings

        Returns:
            Embedded representations [B, L, d]
        """
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        seq_len = x.shape[1]
        # print('h2e in', x.shape)

        # If sequence parallel is enabled, each rank handles a slice of the sequence
        if self.sp_size > 1:
            normal_seq_len = (seq_len + self.sp_size - 1) // self.sp_size
            last_seq_len = seq_len - normal_seq_len * (self.sp_size - 1)

            part_start = normal_seq_len * rank
            part_end = min(normal_seq_len * (rank + 1), seq_len)
            x_part = x[:, part_start:part_end]

            if mask_index is not None:
                mask_part = mask_index[:, part_start:part_end]
                logits_part = logits[:, part_start:part_end] if logits is not None else None
            else:
                mask_part = None
                logits_part = None
        else:
            x_part = x
            mask_part = mask_index
            logits_part = logits

        # Base discrete embedding
        result_part = self.embedding(x_part)

        # Replace selected positions with continuous embeddings
        if mask_part is not None and logits_part is not None:
            prob = torch.softmax(logits_part / self.tau, dim=-1)  # [B, L_part, V]
            input_embeds_h = prob @ self.W_e  # [B, L_part, d]

            # Blend continuous and discrete embeddings
            result_part = torch.where(
                mask_part.unsqueeze(-1),
                iter_cont_weight * input_embeds_h + 1 * result_part,
                result_part
            )

        # 4. Gather and concatenate sequence slices across ranks
        if self.sp_size > 1:
            out = _all_gather_cat(
                result_part,
                dim=1,
                group=None,
                normal_len=normal_seq_len,
                last_len=last_seq_len
            )
        else:
            out = result_part
        # print('h2e out', out.shape)
        return out


class LLaDA2MLP(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: Optional[bool] = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            # reduce_results=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

        if config.hidden_act != "silu":
            raise ValueError("Unsupported activation. Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        # print('1hidden_states:', hidden_states.shape)
        if (self.tp_size == 1) and hidden_states.shape[0] == 0:
            return hidden_states

        gate_up, _ = self.gate_up_proj(hidden_states)
        # print('2gate_up:', gate_up.shape)
        hidden_states = self.act_fn(gate_up)
        # print('3hidden_states:', hidden_states.shape)
        hidden_states, _ = self.down_proj(
            hidden_states, skip_all_reduce=use_reduce_scatter
        )
        # print('4hidden_states:', hidden_states.shape)
        return hidden_states


class LLaDA2Gate(nn.Module):
    def __init__(
        self,
        config,
        params_dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.weight = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                dtype=self.params_dtype,
            ),
        )
        if getattr(config, "moe_router_enable_expert_bias", False):
            self.expert_bias = nn.Parameter(
                torch.empty((config.num_experts,), dtype=torch.get_default_dtype()),
            )
        else:
            self.expert_bias = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, None)
        # .to(
        #     hidden_states.dtype
        # )
        return logits


class LLaDA2SparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.num_shared_experts = config.num_shared_experts
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.score_function = getattr(config, "score_function", None)

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        # Gate always runs at half / full precision for now.
        router_dtype = getattr(config, "router_dtype", None)
        if router_dtype is None:
            self.router_dtype = None
        elif router_dtype == "fp32":
            self.router_dtype = torch.float32
        else:
            self.router_dtype = torch.bfloat16
        # self.topk_filename = 'mini_topk_gsm8k.npy'
        # with open(self.topk_filename, 'wb') as f:
        #     pass
        # check group topk
        self.num_expert_group = getattr(config, "n_group", 0)
        self.topk_group = getattr(config, "topk_group", 0)
        if self.num_expert_group > 0 or self.topk_group > 0:
            assert (
                self.num_expert_group > 0
                and 0 < self.topk_group <= self.num_expert_group
            )
            self.use_grouped_topk = True
        else:
            self.num_expert_group = self.topk_group = None
            self.use_grouped_topk = False

        self.num_experts = (
            config.num_experts 
        )

        self.gate = LLaDA2Gate(
            config=config,
            params_dtype=self.router_dtype,
            prefix=add_prefix("gate", prefix),
        )
        # self.gate.correction_bias = self.gate.correction_bias.clip(0, 1)
        self.correction_bias = (
            self.gate.expert_bias if self.gate.expert_bias is not None else None
        )

        if self.score_function is not None:
            assert (
                self.score_function == "softmax" and self.correction_bias is None
            ) or (
                self.score_function == "sigmoid" and self.correction_bias is not None
            ), "score_function and correction_bias should be in 2 combination (softmax, None) or (sigmoid, not None)"

        self.topk = TopK(
            top_k=self.top_k,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            # num_fused_shared_experts=self.num_fused_shared_experts,
            topk_group=self.topk_group,
            correction_bias=self.correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        # self.experts = get_moe_impl_class(quant_config)(
        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            layer_id=self.layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
            inplace=False,
        )
        # shared expert
        if config.num_shared_experts is not None:
            if hasattr(config, "moe_shared_expert_intermediate_size"):
                intermediate_size = config.moe_shared_expert_intermediate_size
            else:
                intermediate_size = config.moe_intermediate_size
            intermediate_size *= config.num_shared_experts
            # disable tp for shared experts when enable deepep moe
            self.shared_experts = LLaDA2MLP(
                intermediate_size=intermediate_size,
                config=config,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    else {}
                ),
            )
        # dispatcher
        if get_moe_a2a_backend().is_deepep():
            # TODO: we will support tp < ep in the future
            self.ep_size = get_tensor_model_parallel_world_size()

            self.deepep_dispatcher = DeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=self.num_experts,
                num_local_experts=config.num_experts // self.tp_size,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=get_deepep_mode(),
                async_finish=True,  # TODO
                return_recv_hook=True,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if not get_moe_a2a_backend().is_deepep():
            return self.forward_normal(hidden_states, use_reduce_scatter)
        else:
            return self.forward_deepep(hidden_states, forward_batch)

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def _forward_shared_experts(self, hidden_states: torch.Tensor):
        shared_output = None
        if self.num_shared_experts > 0:
            shared_output = self.shared_experts(hidden_states)
        return shared_output
            
    def _forward_router_experts(self, hidden_states: torch.Tensor):
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        # self._save_record(topk_output.topk_ids)
        return self.experts(hidden_states, topk_output)

    @torch.compiler.disable
    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        router_output = self._forward_router_experts(hidden_states)

        with torch.cuda.stream(self.alt_stream):
            shared_output = self._forward_shared_experts(hidden_states)
        current_stream.wait_stream(self.alt_stream)

        return router_output, shared_output

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        bsz, num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        DUAL_STREAM_TOKEN_THRESHOLD = 1024
        if (
            self.alt_stream is not None
            and hidden_states.shape[0] > 0
            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
            and get_is_capture_mode()
        ):
            final_hidden_states, shared_output = self.forward_normal_dual_stream(
                hidden_states
            )
        else:
            shared_output = self._forward_shared_experts(hidden_states)
            final_hidden_states = self._forward_router_experts(hidden_states)
        # print('shared_output', shared_output.shape)
        # print('final_hidden_states', final_hidden_states.shape)

        if self.num_shared_experts > 0:
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1 and not use_reduce_scatter:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        # print('moe output', final_hidden_states.shape)
        return final_hidden_states.view(bsz, num_tokens, hidden_size)

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        shared_output = None
        forward_mode = forward_batch.forward_mode
        if is_non_idle_and_non_empty(forward_mode, hidden_states):
            router_logits = self.gate(hidden_states)
            if self.num_shared_experts > 0:
                shared_output = self.shared_experts(hidden_states)

            topk_weights, topk_idx, _ = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_idx = torch.full(
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            topk_weights = torch.empty(
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )

        if self.ep_size > 1:
            (
                hidden_states,
                topk_idx,
                topk_weights,
                reorder_topk_ids,
                num_recv_tokens_per_expert,
                seg_indptr,
                masked_m,
                expected_m,
            ) = self.deepep_dispatcher.dispatch(
                hidden_states,
                topk_idx,
                topk_weights,
                forward_batch=forward_batch,
            )

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            reorder_topk_ids=reorder_topk_ids,
            seg_indptr=seg_indptr,
            masked_m=masked_m,
            expected_m=expected_m,
            num_recv_tokens_per_expert=num_recv_tokens_per_expert,
            forward_batch=forward_batch,
        )
        if self.ep_size > 1:
            final_hidden_states = self.deepep_dispatcher.combine(
                final_hidden_states,
                topk_idx,
                topk_weights,
                forward_batch=forward_batch,
            )

        final_hidden_states *= self.routed_scaling_factor

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        return final_hidden_states

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LLaDA2Attention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_kv_heads = config.num_key_value_heads
        self.dp_size = get_attention_dp_size()
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        assert self.total_num_heads % attn_tp_size == 0
        assert self.total_num_heads >= self.total_kv_heads
        if attn_tp_size>self.total_kv_heads:
            assert attn_tp_size%self.total_kv_heads==0

        self.num_heads = self.total_num_heads // attn_tp_size
        self.head_dim = config.head_dim if hasattr(config, 'head_dim') else (self.hidden_size // self.total_num_heads)
        self.q_size = self.head_dim * self.num_heads

        self.num_kv_heads = max(1, self.total_kv_heads // attn_tp_size)
        self.total_kv_heads = self.num_kv_heads*attn_tp_size
        self.kv_size = max(1, self.num_kv_heads * self.head_dim)

        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.scale = self.head_dim**-0.5

        self.use_qk_norm = True

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("query_key_value", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # print('attn tp rank', attn_tp_rank, 'attn tp size', attn_tp_size)

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("dense", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        if hasattr(config, "partial_rotary_factor"):
            self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        elif hasattr(config, "rotary_dim"):
            self.rotary_dim = config.rotary_dim
        else:
            self.rotary_dim = self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
            dtype=torch.float32,
        )

        self.alt_stream = alt_stream

    def _apply_q_norm(self, q):
        q_by_head = q.reshape(-1, self.head_dim)
        q_by_head = self.query_layernorm(q_by_head)
        return q_by_head.view(q.shape)
    def _apply_k_norm(self, k):
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.key_layernorm(k_by_head)
        return k_by_head.view(k.shape)

    @torch.compiler.disable(recursive=False)
    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # overlap qk norm
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q = self._apply_q_norm(q)
            with torch.cuda.stream(self.alt_stream):
                k = self._apply_k_norm(k)
            current_stream.wait_stream(self.alt_stream)
        else:
            q = self._apply_q_norm(q)
            k = self._apply_k_norm(k)
        return q, k

    @torch.compiler.disable(recursive=False)
    def _apply_repeat(self, k, v):
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            k = repeat_kv(k, self.num_key_value_groups)
            with torch.cuda.stream(self.alt_stream):
                v = repeat_kv(v, self.num_key_value_groups)
            current_stream.wait_stream(self.alt_stream)   
        else:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)
        return k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values = None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        bsz, q_len, _ = hidden_states.size()
        qkv, _ = self.query_key_value(hidden_states)
        # print("qkv", qkv.shape, 'size', self.q_size, self.kv_size, self.kv_size)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # print(1, q.shape, k.shape, v.shape, positions.shape)
        if self.use_qk_norm:
            q, k = self._apply_qk_norm(q, k)
        
        # q, k = apply_rotary_pos_emb(q, k, *self.rotary_emb(positions), unsqueeze_dim=1)
        # print('rope:', positions.shape, q.shape, k.shape)
        q, k = self.rotary_emb(
            positions.flatten(),
            q.flatten(0, 1),
            k.flatten(0, 1),
            fused_set_kv_buffer_arg=None,
        )
        # print(2, q.shape, k.shape, v.shape, self.layer_id, replace_position)

        q = q.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        present_key_values = None
        if past_key_values is not None:
            cache_k = past_key_values[0]
            cache_v = past_key_values[1]
            cache_length = cache_k.shape[2]
            block_length = k.shape[2]
            k = cache_k.slice_scatter(k, dim=2, start=cache_length - block_length, end=cache_length)
            v = cache_v.slice_scatter(v, dim=2, start=cache_length - block_length, end=cache_length)
            # k, v = past_key_values.update(k, v, self.layer_id, replace_position)
        if use_cache:
            present_key_values = (k, v)

        k, v = self._apply_repeat(k, v)

        if attention_mask is not None:
            if len(attention_mask.shape)==3:
                attention_mask = attention_mask.unsqueeze(1)
        # This code is used to eliminate the impact of cache padding, but with vary cache length, the impact is 
        # limited and has little drop in score. We leave it here for potential future use if there is accuracy issue
        # if past_key_values is not None:
        #     key_padding_mask = (k.abs().sum(1, keepdim=True).sum(-1)>1e-9).unsqueeze(2).repeat(1, 1, q.shape[2], 1)
        #     if attention_mask is not None:
        #         attention_mask = attention_mask & key_padding_mask
        #     else:
        #         attention_mask = key_padding_mask
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        # print(5, attn_output.shape)

        attn_output, _ = self.dense(attn_output)
        # print(6, attn_output.shape)
        return attn_output, present_key_values


class LLaDA2Block(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.dp_size = get_attention_dp_size()
        self.attention = LLaDA2Attention(
            config,
            layer_id,
            quant_config,
            reduce_results=False,
            prefix=add_prefix("attention", prefix),
            alt_stream=alt_stream,
        )
        self.layer_id = layer_id
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.is_layer_sparse = self._is_layer_sparse(
            config, layer_id=layer_id, is_nextn=False
        )
        is_previous_layer_sparse = self._is_layer_sparse(
            config, layer_id=layer_id - 1, is_nextn=False
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
        )

        self.is_last_layer = self.layer_id == config.num_hidden_layers - 1

        if self.is_layer_sparse:
            self.mlp = LLaDA2SparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = LLaDA2MLP(
                intermediate_size=config.intermediate_size,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

    def _is_layer_sparse(
        self, config: PretrainedConfig, layer_id: int, is_nextn: bool
    ) -> bool:
        return is_nextn or (
            config.num_experts is not None and layer_id >= config.first_k_dense_replace
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        past_key_values = None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        bsz, q_len, h = hidden_states.size()
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=None,
        )
        # # residual = hidden_states

        # # hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_values = self.attention(
            positions=positions,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            replace_position=replace_position,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        # hidden_states = hidden_states.flatten(0, 1)
        # residual = residual.flatten(0, 1)
        # print(self.layer_communicator.prepare_mlp)
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=None,
        )
        # if self.tp_size>1:
        #     hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        # # hidden_states, residual = layernorm(hidden_states, residual)
        # hidden_states = residual + hidden_states

        # residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, False)

        # print(self.layer_communicator.postprocess_layer)
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=None,
        )
        # hidden_states = hidden_states + residual
        # hidden_states = hidden_states.view(bsz, q_len, -1)
        # residual = residual.view(bsz, q_len, -1)

        return hidden_states, residual, present_key_values


class LLaDA2Model(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = ".",
    ):
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.full_word_embeddings = nn.Embedding(
                self.vocab_size,
                self.embed_dim,
            )
        if self.pp_group.is_first_rank:
            self.word_embeddings = nn.Embedding(
                self.vocab_size,
                self.embed_dim,
            )
            # self.word_embeddings = VocabParallelEmbedding(
            #     self.vocab_size,
            #     self.embed_dim,
            #     quant_config=quant_config,
            #     prefix=add_prefix("word_embeddings", prefix),
            #     # enable_tp=False,
            #     enable_tp=not is_dp_attention_enabled(),
            # )
        else:
            self.word_embeddings = PPMissingLayer()

        # self.embedding_dropout = torch.nn.Dropout(config.embedding_dropout)

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: LLaDA2Block(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        past_key_values = None,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor]=None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.word_embeddings(input_ids.clone())
            else:
                hidden_states = input_embeds.clone()
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        all_present_key_values = []
        for i in range(self.start_layer, self.end_layer):
            # with get_global_expert_distribution_recorder().with_current_layer(i):
            layer = self.layers[i]
            hidden_states, residual, present_key_values = layer(
                positions,
                hidden_states,
                residual,
                past_key_values[i] if past_key_values is not None else None,
                replace_position=replace_position,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            if use_cache:
                all_present_key_values.extend(present_key_values)
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states = hidden_states + residual
                hidden_states = self.norm(hidden_states)
            return hidden_states, all_present_key_values


class LLaDA2SGLangLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        expert_map_path: str = "",
    ):
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        alt_stream = torch.cuda.Stream() if _is_cuda else None

        self.model = LLaDA2Model(
            config,
            quant_config,
            alt_stream=alt_stream,
            prefix=add_prefix("model", ""),
        )
        self.device = torch.device('cpu')
        self.expert_map_path=expert_map_path

        # tie_word_embeddings为true，复用tie_word_embeddings，反之是独立的
        if config.tie_word_embeddings:
            self.lm_head = self.model.word_embeddings
        else:
            # TODO something wrong with ParallelLMHead with DP attention enabled
            # self.lm_head = ParallelLMHead(
            #     config.vocab_size,
            #     config.hidden_size,
            #     quant_config=quant_config,
            #     prefix=add_prefix("lm_head", prefix),
            #     use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"],
            # )
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.logits_processor = LogitsProcessor(config)

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_embed_and_head(self):
        """Used by the eagle_worker."""
        return self.model.word_embeddings.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        """Used by the eagle_worker."""
        del self.model.word_embeddings.weight
        del self.lm_head.weight
        self.model.word_embeddings.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor=None,
        position_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        past_key_values = None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor]=None,
    ) -> MoeCausalLMOutputWithPast:
        self.device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            if replace_position is not None:
                position_ids = torch.arange(replace_position[0], replace_position[1], device=self.device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
            else:
                position_ids = torch.arange(length, device=self.device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

        # if input_ids is not None:
        #     input_ids = input_ids.flatten()
        
        hidden_states, present_key_values = self.model(
            input_ids,
            position_ids,
            past_key_values,
            inputs_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
            replace_position=replace_position,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states
        logits = self.lm_head(hidden_states)
        return MoeCausalLMOutputWithPast(
            logits=logits,
            past_key_values=present_key_values,
            hidden_states=hidden_states,
        )
        # if self.pp_group.is_last_rank:
        #     return self.logits_processor(
        #         input_ids, hidden_states, self.lm_head, forward_batch
        #     )
        # else:
        #     return hidden_states




    def apply_state_dicts(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
       
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

     
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        # print("====new_state_dict")
        # for key, value in weights.items():
        #     # if int(key.split(".")[3])<num_layers:
        #     print(key, value.shape, value.dtype)
        

        # print("====self.state_dict")
        # for key, value in params_dict.items():
        #     print(key, value.shape, value.dtype)

        new_state_dict_keys = weights.keys()
        self_state_dict_keys = params_dict.keys()
        unused_keys = []
        for key in new_state_dict_keys:
           if key not in self_state_dict_keys:
               unused_keys.append(key)

        not_inited_keys = []
        for key in self_state_dict_keys:
           if key not in new_state_dict_keys:
               not_inited_keys.append(key) 

        print("unused_keys", unused_keys)    
        print("not_inited_keys", not_inited_keys)    

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        for key, value in weights.items():
            if key not in params_dict:
                print(f"not match:{key}")
                continue
            if self.quant_config is not None:
                # unsqueeze to match sglang params shape
                if value.dim() == 0 and params_dict[key].dim() == 1:
                    if params_dict[key].shape[0] == 1:
                        # [] -> [1]
                        value = value.unsqueeze(0)    
                    elif params_dict[key].shape[0] == 2:
                        # [] -> [1] -> [2]
                        value = value.unsqueeze(0).repeat(2)    
                    elif params_dict[key].shape[0] == 3:
                        # [] -> [1] -> [2]
                        value = value.unsqueeze(0).repeat(3)    
                    weights[key] = value 
            if value.shape != params_dict[key].shape:
                # print('shape mismatch:', key, value.shape, params_dict[key].shape)
                if not re.search(r'query_key_value.weight$', key):
                    mismatch_dim = 0 if value.shape[0] != params_dict[key].shape[0] else 1
                    if mismatch_dim==0:
                        part_size = params_dict[key].shape[0]
                        weights[key] = (
                            value[tp_rank * part_size : (tp_rank + 1) * part_size] 
                            if self.quant_config is None 
                            else value[tp_rank * part_size : (tp_rank + 1) * part_size].contiguous()    # this fix stride issue
                        )    
                    else:
                        part_size = params_dict[key].shape[1]
                        weights[key] = (
                            value[:, tp_rank * part_size : (tp_rank + 1) * part_size] 
                            if self.quant_config is None 
                            else value[:, tp_rank * part_size : (tp_rank + 1) * part_size].contiguous()     # this fix stride issue
                        )  
                    if weights[key].shape != params_dict[key].shape:
                        print('shape mismatch fixed:', key, weights[key].shape, params_dict[key].shape)
                else:
                    hidden_size = self.config.hidden_size
                    total_num_heads = self.config.num_attention_heads
                    total_kv_heads = self.config.num_key_value_heads
                    q_dim = hidden_size
                    q_part = q_dim // tp_size
                    q_weight = value[tp_rank * q_part : (tp_rank + 1) * q_part]
                    if tp_size > total_kv_heads:
                        n_replica = tp_size//total_kv_heads
                        kv_dim = hidden_size * total_kv_heads // total_num_heads
                        kv_part = kv_dim // total_kv_heads
                        k_weight = value[q_dim + (tp_rank//n_replica) * kv_part : q_dim + ((tp_rank//n_replica) + 1) * kv_part]
                        v_weight = value[q_dim + kv_dim + (tp_rank//n_replica) * kv_part : q_dim + kv_dim + ((tp_rank//n_replica) + 1) * kv_part]
                    else:
                        kv_dim = hidden_size * total_kv_heads // total_num_heads
                        kv_part = kv_dim // tp_size
                        k_weight = value[q_dim + tp_rank * kv_part : q_dim + (tp_rank + 1) * kv_part]
                        v_weight = value[q_dim + kv_dim + tp_rank * kv_part : q_dim + kv_dim + (tp_rank + 1) * kv_part]
                    weights[key] = torch.cat([q_weight, k_weight, v_weight], dim=0)
                    assert weights[key].shape == params_dict[key].shape


        # for name, loaded_weight in weights.items():
        #     param = params_dict[name]
        #     param.data[:] = loaded_weight
        params_dict = dict(self.named_parameters())
        buffer_dict = dict(self.named_buffers())
        for name, loaded_weight in weights.items():
            if name in params_dict:
                param = params_dict[name]
                param.data = loaded_weight
            elif name in buffer_dict:
                buffer = buffer_dict[name]
                buffer.data = loaded_weight
            else:
                print('params not matching:', name)


        if not is_nextn:
            self.routed_experts_weights_of_layer = {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if not isinstance(layer, PPMissingLayer)
                and isinstance(layer.mlp, LLaDA2SparseMoeBlock)
            }
    
    def _update_state_dict_for_fusemoe_quant(self, state_dict, num_layers, dtype, per_gpu_expert_mapping, per_gpu_inverse_mapping, device):
        new_state_dict = {}
        gate_projs = [{} for _ in range(num_layers)]
        gate_input_scales = [{} for _ in range(num_layers)]
        gate_weight_scales = [{} for _ in range(num_layers)]
        up_projs = [{} for _ in range(num_layers)]
        up_weight_scales = [{} for _ in range(num_layers)]
        down_projs = [{} for _ in range(num_layers)]
        down_input_scales = [{} for _ in range(num_layers)]
        down_weight_scales = [{} for _ in range(num_layers)]
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        moe_tp_rank = get_moe_tensor_parallel_rank()
        moe_tp_size = get_moe_tensor_parallel_world_size()
        for key, value in tqdm.tqdm(state_dict.items()):
            if ".mlp.experts." in key:
                layer_id = int(key.split(".mlp.experts.")[0].split(".")[-1])
                expert_id = int(key.split(".mlp.experts.")[1].split(".")[0])
                
                if layer_id < num_layers:
                
                    if re.search(r'experts\.\d{1,4}\.gate_proj\.input_scale',key):
                        gate_input_scales[layer_id][expert_id] = value
                    elif re.search(r'experts\.\d{1,4}\.gate_proj\.weight_scale',key):
                        gate_weight_scales[layer_id][expert_id] = value
                    elif re.search(r'experts\.\d{1,4}\.up_proj\.weight_scale',key):
                        up_weight_scales[layer_id][expert_id] = value
                    elif re.search(r'experts\.\d{1,4}\.down_proj\.input_scale',key):
                        down_input_scales[layer_id][expert_id] = value
                    elif re.search(r'experts\.\d{1,4}\.down_proj\.weight_scale',key):
                        down_weight_scales[layer_id][expert_id] = value


                    elif re.search(r'experts\.\d{1,4}\.gate_proj\.weight$',key):
                        gate_projs[layer_id][expert_id] = value
                    elif re.search(r'experts\.\d{1,4}\.up_proj\.weight$',key):
                        up_projs[layer_id][expert_id] = value
                    elif re.search(r'experts\.\d{1,4}\.down_proj\.weight$',key):
                        down_projs[layer_id][expert_id] = value
            else:
                new_state_dict[key] = value

        for layer_id in tqdm.trange(num_layers):
            w13_weight = []
            w2_weight = []
            w13_input_scale = []
            w13_weight_scale = []
            w2_input_scale = []
            w2_weight_scale = []
            if f"model.layers.{layer_id}.mlp.w1" in state_dict.keys():
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight"] = self._tp_split(state_dict[f"model.layers.{layer_id}.mlp.w1"][per_gpu_expert_mapping[layer_id]], dim=1, rank=moe_tp_rank, world=moe_tp_size, is_w13=True).contiguous()
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight"] = self._tp_split(state_dict[f"model.layers.{layer_id}.mlp.w2"][per_gpu_expert_mapping[layer_id]], dim=2, rank=moe_tp_rank, world=moe_tp_size).contiguous()
                del new_state_dict[f"model.layers.{layer_id}.mlp.w1"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.w2"]
                self.model.layers[layer_id].mlp.experts.expert_map_cpu = per_gpu_inverse_mapping[layer_id]
                
            # if 0 in gate_projs[layer_id].keys():
            if len(gate_projs[layer_id]) > 0:
                for expert_id in per_gpu_expert_mapping[layer_id]:
                    expert_id = int(expert_id)
                    gate_proj = gate_projs[layer_id][expert_id].to(device)
                    up_proj = up_projs[layer_id][expert_id].to(device)
                    down_proj = down_projs[layer_id][expert_id].to(device)
                    gate_weight_scale = gate_weight_scales[layer_id][expert_id].to(device)
                    up_weight_scale = up_weight_scales[layer_id][expert_id].to(device)
                    down_weight_scale = down_weight_scales[layer_id][expert_id].to(device)
                    gate_input_scale = gate_input_scales[layer_id][expert_id].to(device)
                    down_input_scale = down_input_scales[layer_id][expert_id].to(device)

                    w13_weight.append(torch.cat([gate_proj, up_proj], dim=0))
                    w2_weight.append(down_proj)

                    w13_input_scale.append(gate_input_scale)
                    w13_weight_scale.append(torch.stack([gate_weight_scale, up_weight_scale], dim=0))
                    w2_input_scale.append(down_input_scale)
                    w2_weight_scale.append(down_weight_scale)

                w13_weight = self._tp_split(torch.stack(w13_weight, dim=0), dim=1, rank=moe_tp_rank, world=moe_tp_size, is_w13=True)
                w2_weight = self._tp_split(torch.stack(w2_weight, dim=0), dim=2, rank=moe_tp_rank, world=moe_tp_size)
                w13_input_scale = torch.stack(w13_input_scale, dim=0)
                w13_weight_scale = torch.stack(w13_weight_scale, dim=0)
                w2_input_scale = torch.stack(w2_input_scale, dim=0)
                w2_weight_scale = torch.stack(w2_weight_scale, dim=0)

                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight"] = w13_weight.contiguous()
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight"] = w2_weight.contiguous()
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_input_scale"] = w13_input_scale.contiguous()
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight_scale"] = w13_weight_scale.contiguous()
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_input_scale"] = w2_input_scale.contiguous()
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight_scale"] = w2_weight_scale.contiguous()
                    
                self.model.layers[layer_id].mlp.experts.expert_map_cpu = per_gpu_inverse_mapping[layer_id]
            
            if f"model.layers.{layer_id}.mlp.gate.expert_bias" in state_dict.keys():
                new_state_dict[f"model.layers.{layer_id}.mlp.correction_bias"] = state_dict[f"model.layers.{layer_id}.mlp.gate.expert_bias"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.gate.expert_bias"]

            if f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight" in state_dict.keys():
                part_size = state_dict[f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"].shape[0] // tp_size
                part_start = tp_rank * part_size
                part_end = (tp_rank + 1) * part_size
                
                new_state_dict[f"model.layers.{layer_id}.mlp.shared_experts.gate_up_proj.weight"] = torch.cat(
                    [state_dict[f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"][part_start:part_end], 
                    state_dict[f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"][part_start:part_end]], dim=0)
                new_state_dict[f"model.layers.{layer_id}.mlp.shared_experts.gate_up_proj.weight_scale"] = torch.stack(
                    [state_dict[f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight_scale"], 
                    state_dict[f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight_scale"]], dim=0)
                new_state_dict[f"model.layers.{layer_id}.mlp.shared_experts.gate_up_proj.input_scale"] = torch.stack(
                    [state_dict[f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.input_scale"], 
                    state_dict[f"model.layers.{layer_id}.mlp.shared_experts.up_proj.input_scale"]], dim=0)
                del new_state_dict[f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight_scale"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight_scale"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.input_scale"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.shared_experts.up_proj.input_scale"]

            if f"model.layers.{layer_id}.mlp.gate_proj.weight" in state_dict.keys():
                part_size = state_dict[f"model.layers.{layer_id}.mlp.gate_proj.weight"].shape[0] // tp_size
                part_start = tp_rank * part_size
                part_end = (tp_rank + 1) * part_size
                new_state_dict[f"model.layers.{layer_id}.mlp.gate_up_proj.weight"] = torch.cat(
                    [state_dict[f"model.layers.{layer_id}.mlp.gate_proj.weight"][part_start:part_end], 
                    state_dict[f"model.layers.{layer_id}.mlp.up_proj.weight"][part_start:part_end]], dim=0)
                new_state_dict[f"model.layers.{layer_id}.mlp.gate_up_proj.weight_scale"] = torch.stack(
                    [state_dict[f"model.layers.{layer_id}.mlp.gate_proj.weight_scale"], 
                    state_dict[f"model.layers.{layer_id}.mlp.up_proj.weight_scale"]], dim=0)
                new_state_dict[f"model.layers.{layer_id}.mlp.gate_up_proj.input_scale"] = torch.stack(
                    [state_dict[f"model.layers.{layer_id}.mlp.gate_proj.input_scale"], 
                    state_dict[f"model.layers.{layer_id}.mlp.up_proj.input_scale"]], dim=0)
                del new_state_dict[f"model.layers.{layer_id}.mlp.gate_proj.weight"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.up_proj.weight"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.gate_proj.weight_scale"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.up_proj.weight_scale"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.gate_proj.input_scale"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.up_proj.input_scale"]

        new_state_dict['model.full_word_embeddings.weight'] = state_dict['model.word_embeddings.weight']
        
        for key, value in tqdm.tqdm(new_state_dict.items()):
            new_state_dict[key] = value.to(device)
        self.apply_state_dicts(new_state_dict)
        for name, param in self.named_parameters():

            if 'norm' in name:
                param.data = param.data.to(dtype)
            elif 'embed_tokens' in name:
                param.data = param.data.to(dtype)
            elif 'lm_head' in name:
                param.data = param.data.to(dtype)
            elif '.mlp.correction_bias' in name :
                param.data = param.data.to(torch.float32)
            else:
                continue

        for name, buf in self.named_buffers():
            if "scale" in name:
                continue
            if buf.dtype != dtype:
                buf.data = buf.data.to(dtype)


    def _update_state_dict_for_fusemoe(self, state_dict, num_layers, dtype, per_gpu_expert_mapping, per_gpu_inverse_mapping, device):
        new_state_dict = {}
        gate_projs = [{} for _ in range(num_layers)]
        up_projs = [{} for _ in range(num_layers)]
        down_projs = [{} for _ in range(num_layers)]

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        moe_tp_rank = get_moe_tensor_parallel_rank()
        moe_tp_size = get_moe_tensor_parallel_world_size()
        for key, value in tqdm.tqdm(state_dict.items()):
            if ".mlp.experts." in key:
                layer_id = int(key.split(".mlp.experts.")[0].split(".")[-1])
                expert_id = int(key.split(".mlp.experts.")[1].split(".")[0])
                
                if layer_id < num_layers:
                    if "gate_proj" in key:
                        gate_projs[layer_id][expert_id] = value
                    elif "up_proj" in key:
                        up_projs[layer_id][expert_id] = value
                    elif "down_proj" in key:
                        down_projs[layer_id][expert_id] = value
            else:
                new_state_dict[key] = value

        for layer_id in tqdm.trange(num_layers):
            w13_weight = []
            w2_weight = []
            if f"model.layers.{layer_id}.mlp.w1" in state_dict.keys():
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight"] = self._tp_split(state_dict[f"model.layers.{layer_id}.mlp.w1"][per_gpu_expert_mapping[layer_id]], dim=1, rank=moe_tp_rank, world=moe_tp_size, is_w13=True).contiguous()
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight"] = self._tp_split(state_dict[f"model.layers.{layer_id}.mlp.w2"][per_gpu_expert_mapping[layer_id]], dim=2, rank=moe_tp_rank, world=moe_tp_size).contiguous()
                del new_state_dict[f"model.layers.{layer_id}.mlp.w1"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.w2"]
                self.model.layers[layer_id].mlp.experts.expert_map_cpu = per_gpu_inverse_mapping[layer_id]
                
            if 0 in gate_projs[layer_id].keys():
                for expert_id in per_gpu_expert_mapping[layer_id]:
                    expert_id = int(expert_id)
                    gate_proj = gate_projs[layer_id][expert_id].to(device)
                    up_proj = up_projs[layer_id][expert_id].to(device)
                    down_proj = down_projs[layer_id][expert_id].to(device)
                    w13_weight.append(torch.cat([gate_proj, up_proj], dim=0))
                    w2_weight.append(down_proj)
                w13_weight = self._tp_split(torch.stack(w13_weight, dim=0), dim=1, rank=moe_tp_rank, world=moe_tp_size, is_w13=True)
                w2_weight = self._tp_split(torch.stack(w2_weight, dim=0), dim=2, rank=moe_tp_rank, world=moe_tp_size)

                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight"] = w13_weight.contiguous()
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight"] = w2_weight.contiguous()
                self.model.layers[layer_id].mlp.experts.expert_map_cpu = per_gpu_inverse_mapping[layer_id]
            if f"model.layers.{layer_id}.mlp.gate.expert_bias" in state_dict.keys():
                new_state_dict[f"model.layers.{layer_id}.mlp.correction_bias"] = state_dict[f"model.layers.{layer_id}.mlp.gate.expert_bias"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.gate.expert_bias"]

            if f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight" in state_dict.keys():
                part_size = state_dict[f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"].shape[0] // tp_size
                part_start = tp_rank * part_size
                part_end = (tp_rank + 1) * part_size
                new_state_dict[f"model.layers.{layer_id}.mlp.shared_experts.gate_up_proj.weight"] = torch.cat([state_dict[f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"][part_start:part_end], state_dict[f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"][part_start:part_end]], dim=0)
                del new_state_dict[f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"]
            if f"model.layers.{layer_id}.mlp.gate_proj.weight" in state_dict.keys():
                part_size = state_dict[f"model.layers.{layer_id}.mlp.gate_proj.weight"].shape[0] // tp_size
                part_start = tp_rank * part_size
                part_end = (tp_rank + 1) * part_size
                new_state_dict[f"model.layers.{layer_id}.mlp.gate_up_proj.weight"] = torch.cat([state_dict[f"model.layers.{layer_id}.mlp.gate_proj.weight"][part_start:part_end], state_dict[f"model.layers.{layer_id}.mlp.up_proj.weight"][part_start:part_end]], dim=0)
                del new_state_dict[f"model.layers.{layer_id}.mlp.gate_proj.weight"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.up_proj.weight"]

        new_state_dict['model.full_word_embeddings.weight'] = state_dict['model.word_embeddings.weight']
        for key, value in tqdm.tqdm(new_state_dict.items()):
            new_state_dict[key] = value.to(device)
        self.apply_state_dicts(new_state_dict)
        for name, param in self.named_parameters():
            if '.mlp.correction_bias' in name or 'layernorm.weight' in name:
                param.data = param.data.to(torch.float32)
            else:
                param.data = param.data.to(dtype)

    def _tp_split(self, tensor: torch.Tensor, dim: int, rank: int, world: int, is_w13=False):
        """把 tensor 按 dim 切成 world 份，返回 rank 对应的那份"""
        if world == 1:
            return tensor
        if is_w13:
            shard_size = tensor.size(dim) // 2
            size = shard_size // world
            w1 = tensor.narrow(dim, rank * size, size)
            w3 = tensor.narrow(dim, shard_size + rank * size, size)
            return torch.cat([w1, w3], dim=dim)
        else:
            size = tensor.size(dim) // world
            return tensor.narrow(dim, rank * size, size)

    def load_state_dict(self, model_dir, strict=True, dtype=torch.bfloat16, device=None):
        num_experts = self.config.num_experts
        moe_intermediate_size = self.config.moe_intermediate_size
        num_layers = self.config.num_hidden_layers
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        ep_rank = get_moe_expert_parallel_rank()
        ep_size = get_moe_expert_parallel_world_size()

        index_path = Path(model_dir) / "model.safetensors.index.json"
        with open(index_path, "r") as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        shard_files = {v for v in weight_map.values()}

        state_dict = {}
        if num_layers == 20:
            expert_map_path = Path(self.expert_map_path+'/mini_expert_map_'+str(ep_size)+'.pt')
        else:
            expert_map_path = Path(self.expert_map_path+'/flash_expert_map_'+str(ep_size)+'.pt')
        if expert_map_path.exists():
            print("load expert_map from", expert_map_path)
            expert_map = torch.load(expert_map_path)
        else:
            print('no expert_map found in', expert_map_path)
            expert_map = torch.zeros(num_experts, dtype=torch.int32)  # expert_to_gpu[l, e] = gpu_id
            for e in range(num_experts):
                expert_map[e] = e // (num_experts//ep_size)
            expert_map = expert_map.unsqueeze(0).repeat(num_layers, 1)
        arange_256 = torch.arange(num_experts, dtype=torch.int64)
        per_gpu_expert_mapping = [arange_256[expert_map[i]==ep_rank] for i in range(num_layers)]
        per_gpu_inverse_mapping = [torch.ones(num_experts, dtype=torch.int64).mul(-1) for _ in range(num_layers)]
        for layer_id in range(num_layers):
            per_gpu_inverse_mapping[layer_id][per_gpu_expert_mapping[layer_id]] = torch.arange(per_gpu_expert_mapping[layer_id].shape[0])
              
        for shard in tqdm.tqdm(sorted(shard_files)):
            shard_path = Path(model_dir) / shard
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard: {shard_path}")
            
            with torch.inference_mode():
                file_state_dict = load_file(str(shard_path))
                filtered_file_state_dict = {}
                for key, value in file_state_dict.items():
                    if ".mlp.experts." in key:
                        layer_id = int(key.split(".mlp.experts.")[0].split(".")[-1])
                        expert_id = int(key.split(".mlp.experts.")[1].split(".")[0])
                        if expert_map[layer_id][expert_id] == ep_rank:
                            filtered_file_state_dict[key] = value
                    else:
                        filtered_file_state_dict[key] = value
                        
                state_dict.update(file_state_dict)

        if self.quant_config is not None:
            self._update_state_dict_for_fusemoe_quant(state_dict, num_layers, dtype, per_gpu_expert_mapping, per_gpu_inverse_mapping, device)
        else:
            self._update_state_dict_for_fusemoe(state_dict, num_layers, dtype, per_gpu_expert_mapping, per_gpu_inverse_mapping, device)



    def init_h2e_module(self):
        self.h2e = H2Embed(self.model.full_word_embeddings, tau=1.0)


    def load_weights(self, model_path, torch_dtype = torch.bfloat16, device=None):
        if device is None:
            device = self.device
        self.load_state_dict(model_path, strict=False, dtype=torch_dtype, device=device)
        self.init_h2e_module()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        num_groups = getattr(config, "n_group", 0)
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None if num_groups == 0 else num_groups,
        )
    def after_loading(self):
        for name, module in self.named_modules():
            if hasattr(module, "quant_method") and module.quant_method is not None and hasattr(module.quant_method, "process_weights_after_loading"):
                if hasattr(module, "weight_scale") and module.weight_scale is not None:
                    if module.weight_scale.dim() == 0:
                        print(f"Fixing scalar weight_scale for {name}")
                        module.weight_scale.data = module.weight_scale.data.unsqueeze(0)
                if hasattr(module, "input_scale") and module.input_scale is not None:
                    if module.input_scale.dim() == 0:
                        print(f"Fixing scalar input_scale for {name}")
                        module.input_scale.data = module.input_scale.data.unsqueeze_(0)
                module.quant_method.process_weights_after_loading(module)
    
    def after_processing(self):
        if self.quant_config is not None:
            self.after_loading()

EntryClass = [LLaDA2SGLangLM]
