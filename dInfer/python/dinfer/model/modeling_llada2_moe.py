# coding=utf-8
# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch BailingMoE model."""

import math
import warnings
from typing import List, Optional, Tuple, Union, Literal

import torch
import torch.nn.functional as F
from torch import nn
import tqdm

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_llada2_moe import LLaDA2MoeConfig
from torch.nn.modules.normalization import RMSNorm
import torch.distributed as dist
from ..decoding.utils import KVCache
from transformers.generation.utils import GenerationMixin
from dataclasses import dataclass
from transformers.utils import ModelOutput

from pathlib import Path
import json
from safetensors.torch import load_file
from functools import partial
from vllm.model_executor.layers.fused_moe import FusedMoE
import re
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                        ReplicatedLinear,
                        QKVParallelLinear,
                        RowParallelLinear)
def torch_all_reduce(tensor):
    torch.distributed.all_reduce(tensor)
    return tensor
import vllm.distributed as vllm_distributed
vllm_distributed.tensor_model_parallel_all_reduce = torch_all_reduce

from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LLaDA2MoeConfig"


def roll_tensor(tensor, shifts=-1, dims=-1, fill_value=0):
    """Roll the tensor input along the given dimension(s).
    Inserted elements are set to be 0.0.
    """
    rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
    rolled_tensor.select(dims, shifts).fill_(fill_value)
    return rolled_tensor, rolled_tensor.sum()

def replace_linear_class(
    linear: nn.Linear, style: Literal["colwise", "rowwise", "qkv"],
    quant_config, model_config
) -> Union[ColumnParallelLinear, RowParallelLinear]:
    """
    Replace nn.Linear with one of vLLM's tensor parallel linear classes.

    Args:
        linear (nn.Linear): `nn.Linear` to be replaced.
        style (str): Tensor parallel style of the new linear, e.g. "colwise".
        quant_config (QuantConfig): Quantization config for the new linear.
    Returns:
        Union[ColumnParallelLinear, RowParallelLinear]: The new linear.
    """

    if not isinstance(style, str):
        raise ValueError(
            f"Unsupported parallel style type {type(style)}, expected str")

    vllm_linear_cls = {
        "colwise": ColumnParallelLinear,
        "rowwise": RowParallelLinear,
        "qkv": QKVParallelLinear
    }.get(style, ReplicatedLinear)
    if style != "qkv":
        return vllm_linear_cls(
            input_size=linear.in_features,
            output_size=linear.out_features,
            bias=linear.bias is not None,
            quant_config=quant_config,
            return_bias=False,
        )
    else:
        return QKVParallelLinear(
            hidden_size = model_config.hidden_size,
            head_size=model_config.head_dim,
            total_num_heads=model_config.num_attention_heads,
            total_num_kv_heads=model_config.num_key_value_heads,
            bias=linear.bias is not None,
            quant_config=quant_config,
            return_bias=False,
        )     
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
        self.sp_size = 1  # no sequence parallel by default

    def __call__(
        self,
        x: torch.Tensor,
        mask_index: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        iter_cont_weight: float = 0.0
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

        return out


@dataclass
class MoEV2CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs as well as Mixture of Expert's router hidden
    states terms, to train a MoE model.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        z_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            z_loss for the sparse modules.
        aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            aux_loss for the sparse modules.
        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Router logits of the encoder model, useful to compute the auxiliary loss and the z_loss for the sparse
            modules.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    z_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None
    mtp_loss: Optional[torch.FloatTensor] = None
    mtp_logits: Optional[tuple[torch.FloatTensor, ...]] = None


class MoeV2ModelOutputWithPast(MoeModelOutputWithPast):

    def __init__(self, mtp_hidden_states=None, **kwargs):
        super().__init__(**kwargs)
        self.mtp_hidden_states = mtp_hidden_states


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.LLaDA2Moe.modeling_LLaDA2Moe._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.LLaDA2Moe.modeling_LLaDA2Moe._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.LLaDA2Moe.modeling_LLaDA2Moe.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


class LLaDA2MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LLaDA2MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LLaDA2MoeRMSNorm)


class LLaDA2MoeRotaryEmbedding(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class LLaDA2MoeMLP(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LLaDA2MoeGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts

        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.routed_scaling_factor = config.routed_scaling_factor

        self.register_buffer("expert_bias", torch.zeros((self.num_experts)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def group_limited_topk(
        self,
        scores: torch.Tensor,
    ):
        num_tokens, _ = scores.size()
        # Organize the experts into groups
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Mask the experts based on selection groups
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )

        masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
        probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)

        return probs, top_indices

    def forward(self, hidden_states):
        # compute gating score
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

        scores = torch.sigmoid(logits.float()).type_as(logits)

        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)

        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)

        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight, logits

    def get_logits(self, hidden_states):
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return logits

    def routing(self, hidden_states, gating_output, topk, renormalize):
        scores = torch.sigmoid(gating_output.float()).type_as(gating_output)

        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)

        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(gating_output)

        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_weight, topk_idx

def static_routing_function(gate, hidden_states, gating_output, topk, renormalize):
    return gate.routing(hidden_states, gating_output, topk, renormalize)
class LLaDA2MoeSparseMoeBlock(nn.Module):
    """A tensor-parallel MoE implementation for Olmoe that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(self,
                 config,
                 prefix: str = ""):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # Gate always runs at half / full precision for now.
        self.gate = LLaDA2MoeGate(config)
        # print('config.num_shared_experts', config.num_shared_experts)
        if config.num_shared_experts is not None:
            # print('config.num_shared_experts is not None!')
            self.shared_experts = LLaDA2MoeMLP(
                config=config, intermediate_size=config.moe_intermediate_size * config.num_shared_experts
            )
        # custom_routing = partial(custom_routing_function, gate=self.gate)
        self.experts = FusedMoE(num_experts=self.num_experts,
                                top_k=self.top_k,
                                hidden_size=self.hidden_size,
                                intermediate_size=config.moe_intermediate_size,
                                reduce_results=True,
                                quant_config=None,
                                tp_size=None,
                                custom_routing_function=partial(static_routing_function, self.gate),
                                prefix=f"{prefix}.experts")
        # This is a hack. expert_map in FusedMoE isn't moved to GPU by default.
        # We have to register it explicitly so that it can be moved to GPU with FusedMoE
        expert_map = self.experts.expert_map
        del self.experts.expert_map
        self.experts.register_buffer('expert_map', expert_map)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # print("    mlp", "input", hidden_states.flatten()[:10].cpu())
        res = self.shared_experts(hidden_states)
        # print("    mlp", "initial identity", identity.flatten()[:10].cpu())
        bsz, seq_len, h = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, h)
        router_logits = self.gate.get_logits(hidden_states_flat)
        # print("    mlp", "router_logits", router_logits.flatten()[:10].cpu())
        y = self.experts.forward_impl(hidden_states=hidden_states_flat,
                                           router_logits=router_logits)
        y = y.view(bsz, seq_len, h)
        # y = hidden_states

        # print("    mlp", "after experts", y.flatten()[:10].cpu())
        if self.config.num_shared_experts is not None:
            # print('config.num_shared_experts is not None!')
            # print("    mlp", "shared_experts identity", identity.flatten()[:10].cpu())
            y = y + res
            # print("    mlp", "after shared_experts", y.flatten()[:10].cpu())
        return y

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


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->LLaDA2Moe
class LLaDA2MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LLaDA2MoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or self.hidden_size // self.num_heads
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        self.rope_dim = int(self.head_dim * partial_rotary_factor)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = False
        self.tp_size = 1

        self.query_key_value = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )

        # if self.config.use_qk_norm:
        self.query_layernorm = LLaDA2MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = LLaDA2MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # if self.config.use_qk_norm:
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        kv_seq_len = key_states.shape[-2]
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->LLaDA2Moe
class LLaDA2MoeFlashAttention2(LLaDA2MoeAttention):
    """
    LLaDA2Moe flash attention module. This module inherits from `LLaDA2MoeAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LLaDA2MoeFlashAttention2 attention does not support output_attentions
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # if self.config.use_qk_norm:
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently cast in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slow down training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LLaDA2MoeRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = self.query_key_value.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            query_length (`int`):
                The length of the query sequence in terms of tokens. This represents the number of tokens in the
                `query_states` tensor along the sequence dimension. It is used to determine the effective sequence
                length for attention computations.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LLaDA2MoeFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention with Llama->LLaDA2Moe
class LLaDA2MoeSdpaAttention(LLaDA2MoeAttention):
    """
    LLaDA2Moe attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LLaDA2MoeAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LLaDA2MoeAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        cache_position: Optional[torch.LongTensor] = None,
        replace_position= None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LLaDA2MoeModel is using LLaDA2MoeSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        # vanilla version
        # qkv = self.query_key_value(hidden_states)
        # qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        # query_states, key_states, value_states = qkv.split(
        #     [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        # )

        #tp version
        qkv = self.query_key_value(hidden_states) 
        # print("in sdpa ", qkv.shape, self.num_heads, self.num_key_value_heads, self.tp_size, self.num_heads//self.tp_size, self.num_key_value_heads//self.tp_size)
        qkv = qkv.view(bsz, q_len, self.num_heads//self.tp_size + 2 * self.num_key_value_heads//self.tp_size, self.head_dim)
        query_states, key_states, value_states = qkv.split(
            [self.num_heads//self.tp_size, self.num_key_value_heads//self.tp_size, self.num_key_value_heads//self.tp_size], dim=-2
        )
        
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # if self.config.use_qk_norm:
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings
        # print('shape in sdpa:', query_states.shape, key_states.shape, cos.shape, sin.shape)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # if past_key_value is not None:
        #     cache_kwargs = {"sin": sin, "cos": cos}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, replace_position)
        
        if use_cache:
            past_key_value = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            kv_seq_len = key_states.shape[-2]
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
              attention_mask = attention_mask.unsqueeze(1)
                # raise ValueError(
                #     f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                # )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attention_mask = attention_mask.bool() if attention_mask is not None else None

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.dense(attn_output)

        return attn_output, None, past_key_value


ATTENTION_CLASSES = {
    "eager": LLaDA2MoeAttention,
    "flash_attention_2": LLaDA2MoeFlashAttention2,
    "sdpa": LLaDA2MoeSdpaAttention,
}


class LLaDA2MoeMTPLayer(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.enorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.post_attention_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = LLaDA2MoeSparseMoeBlock(config)

        self.hnorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embeds,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        input_embeds = self.enorm(input_embeds)
        hidden_states = self.hnorm(hidden_states)
        hidden_states = self.eh_proj(torch.cat([input_embeds, hidden_states], dim=-1))
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.to(residual.device)
        hidden_states = self.final_layernorm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class LLaDA2MoeDecoderLayer(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention = ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = (
            LLaDA2MoeSparseMoeBlock(config, prefix=f"model.layers.{layer_idx}.mlp")
            if (config.num_experts is not None and layer_idx >= config.first_k_dense_replace)
            else LLaDA2MoeMLP(config=config, intermediate_size=config.intermediate_size)
        )
        self.input_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Tuple[torch.Tensor]] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        replace_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*):
                cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # print(hidden_states.shape)
        # print("attn_mask")
        # print(attention_mask.shape)
        # print("position_ids")
        # print(position_ids.shape)
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            replace_position=replace_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.to(residual.device)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


LLaDA2Moe_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LLaDA2MoeConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaDA2Moe Model outputting raw hidden-states without any specific head on top.",
    LLaDA2Moe_START_DOCSTRING,
)
class LLaDA2MoePreTrainedModel(PreTrainedModel):
    config_class = LLaDA2MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LLaDA2MoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLaDA2Moe_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaDA2Moe Model outputting raw hidden-states without any specific head on top.",
    LLaDA2Moe_START_DOCSTRING,
)
class LLaDA2MoeModel(LLaDA2MoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LLaDA2MoeDecoderLayer`]

    Args:
        config: LLaDA2MoeConfig
    """

    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_nextn_predict_layers = 0

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = []
        for layer_idx in range(config.num_hidden_layers + self.num_nextn_predict_layers):
            layer_cls = LLaDA2MoeDecoderLayer if layer_idx < config.num_hidden_layers else LLaDA2MoeMTPLayer
            self.layers.append(layer_cls(config, layer_idx))

        self.layers = nn.ModuleList(self.layers)

        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LLaDA2MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    @add_start_docstrings_to_model_forward(LLaDA2Moe_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None, # add extra cache_position / replace position for dInfer kvcache managerment
        replace_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, MoeV2ModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        # if position_ids is None:
            # position_ids = torch.arange(
            #     past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            # )
            # position_ids = position_ids.unsqueeze(0)
        
        # 
        if position_ids is None:
            if replace_position is not None:
                position_ids = torch.arange(replace_position[0], replace_position[1], device=inputs_embeds.device, dtype=torch.long).unsqueeze(0)
            else:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long).unsqueeze(0)


        # if self._use_flash_attention_2:
        #     # 2d mask is passed through the layers
        #     attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        # elif self._use_sdpa and not output_attentions:
        #     # output_attentions=True can not be supported when using SDPA, and we fall back on
        #     # the manual implementation that requires a 4D causal mask in all cases.
        #     attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
        #         attention_mask,
        #         (batch_size, seq_length),
        #         inputs_embeds,
        #         past_seen_tokens,
        #     )
        # else:
        #     # 4d mask is passed through the layers
        #     attention_mask = _prepare_4d_causal_attention_mask(
        #         attention_mask, (batch_size, seq_length), inputs_embeds, past_seen_tokens
        #     )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        # next_decoder_cache = None
        next_decoder_cache = []
        layers = self.layers[: -self.num_nextn_predict_layers] if self.num_nextn_predict_layers > 0 else self.layers
        mtp_layers = self.layers[-self.num_nextn_predict_layers :] if self.num_nextn_predict_layers > 0 else None

        for decoder_layer in layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position, # add 2 extra cache args
                    replace_position=replace_position,

                )
            hidden_states = layer_outputs[0]

            if use_cache:
                # next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                next_decoder_cache.extend(layer_outputs[2 if output_attentions else 1])


            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)
        main_hidden_states = hidden_states

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (main_hidden_states,)


        mtp_hidden_states = None

        if mtp_layers:
            for decoder_layer in mtp_layers:
                input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1)
                inputs_embeds = self.word_embeddings(input_ids)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        inputs_embeds,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        output_router_logits,
                        use_cache,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        inputs_embeds,
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                    )
                if mtp_hidden_states is None:
                    mtp_hidden_states = []
                hidden_states = layer_outputs[0]
                mtp_hidden_states.append(hidden_states)

                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                if output_router_logits and layer_outputs[-1] is not None:
                    all_router_logits += (layer_outputs[-1],)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache

        if not return_dict:
            return tuple(
                v
                for v in [main_hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeV2ModelOutputWithPast(
            last_hidden_state=main_hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            mtp_hidden_states=mtp_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class LLaDA2MoeModelLM(LLaDA2MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__(config)
        self.model = LLaDA2MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_nextn_predict_layers = 0
        self.mtp_loss_scaling_factor = 0

        # # Initialize weights and apply final processing
        # self.post_init()
        # Initialize weights and apply final processing
        self._tp_size = 1
        self.post_init()
        self._tp_plan = {
            "layers.*.attention.query_key_value": "qkv",
            "layers.*.attention.dense": "rowwise",
        }
        self.init_h2e_module()

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
        
    def load_state_dict(self, model_dir, strict=True, dtype=torch.bfloat16, device=None):
        num_experts = self.config.num_experts
        moe_intermediate_size = self.config.moe_intermediate_size
        num_layers = self.config.num_hidden_layers
        ep_rank = get_tensor_model_parallel_rank()
        ep_size = get_tensor_model_parallel_world_size()
        expert_start = ep_rank * num_experts // ep_size
        expert_end = (ep_rank + 1) * num_experts // ep_size
        index_path = Path(model_dir) / "model.safetensors.index.json"
        with open(index_path, "r") as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        shard_files = {v for v in weight_map.values()}

        state_dict = {}
        # print(shard_files)
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
                        if expert_start <= expert_id < expert_end:
                            filtered_file_state_dict[key] = value
                    else:
                        filtered_file_state_dict[key] = value
                            
                        
                state_dict.update(filtered_file_state_dict)

        new_state_dict = {}
        gate_projs = [{} for _ in range(num_layers)]
        up_projs = [{} for _ in range(num_layers)]
        down_projs = [{} for _ in range(num_layers)]
        for key, value in tqdm.tqdm(state_dict.items()):
            if ".mlp.experts." in key:
                layer_id = int(key.split(".mlp.experts.")[0].split(".")[-1])
                expert_id = int(key.split(".mlp.experts.")[1].split(".")[0])
                if layer_id < num_layers:
                    if "gate_proj" in key:
                        gate_projs[layer_id][expert_id-expert_start] = value
                    elif "up_proj" in key:
                        up_projs[layer_id][expert_id-expert_start] = value
                    elif "down_proj" in key:
                        down_projs[layer_id][expert_id-expert_start] = value
            else:
                new_state_dict[key] = value

        del state_dict
        for layer_id in tqdm.trange(num_layers):
            if f"model.layers.{layer_id}.mlp.w1" in new_state_dict.keys():
                ep_rank = get_tensor_model_parallel_rank()
                ep_size = get_tensor_model_parallel_world_size()
                size = divide(new_state_dict[f"model.layers.{layer_id}.mlp.w1"].shape[0], ep_size)
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight"] = new_state_dict[f"model.layers.{layer_id}.mlp.w1"][ep_rank*size:(ep_rank+1)*size]
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight"] = new_state_dict[f"model.layers.{layer_id}.mlp.w2"][ep_rank*size:(ep_rank+1)*size]
                del new_state_dict[f"model.layers.{layer_id}.mlp.w1"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.w2"]
            else:
                w13_weight = []
                w2_weight = []
                if 0 in gate_projs[layer_id].keys():
                    for expert_id in range(num_experts//ep_size):
                        gate_proj = gate_projs[layer_id][expert_id].to(device)
                        up_proj = up_projs[layer_id][expert_id].to(device)
                        down_proj = down_projs[layer_id][expert_id].to(device)
                        w13_weight.append(torch.cat([gate_proj, up_proj], dim=0))
                        w2_weight.append(down_proj)
                    w13_weight = torch.stack(w13_weight, dim=0)
                    w2_weight = torch.stack(w2_weight, dim=0)
                    new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight"] = w13_weight.contiguous().to(device)
                    new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight"] = w2_weight.contiguous().to(device)
                    del w13_weight, w2_weight


        # print("====new_state_dict")
        # for key, value in new_state_dict.items():
        #     # if int(key.split(".")[3])<num_layers:
        #     print(key, value.shape, value.dtype)

        # print("====self.state_dict")
        # for key, value in self.state_dict().items():
        #     print(key, value.shape, value.dtype)

        new_state_dict_keys = new_state_dict.keys()
        self_state_dict_keys = self.state_dict().keys()
        
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
        
        

        # 调用父类方法加载
        # super().load_state_dict(new_state_dict, strict=strict)
        for key, value in tqdm.tqdm(new_state_dict.items()):
            new_state_dict[key] = value.to(device)
        params_dict = dict(self.named_parameters())
        buffer_dict = dict(self.named_buffers())
        for name, loaded_weight in new_state_dict.items():
            if name in params_dict:
                param = params_dict[name]
                param.data = loaded_weight
            elif name in buffer_dict:
                buffer = buffer_dict[name]
                buffer.data = loaded_weight
            else:
                print('params not matching:', name)
        # super().load_state_dict(new_state_dict, strict=strict)
        for name, param in self.named_parameters():
            if '.mlp.gate.expert_bias' in name:
                param.data = param.data.to(torch.float32)
            else:
                param.data = param.data.to(dtype)
    
            


    def load_sharded_safetensors(self, model_dir):
        index_path = Path(model_dir) / "model.safetensors.index.json"
        with open(index_path, "r") as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        shard_files = {v for v in weight_map.values()}

        state_dict = {}
        for shard in sorted(shard_files):  
            shard_path = Path(model_dir) / shard
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard: {shard_path}")
            
            with torch.inference_mode():
                state_dict.update(load_file(str(shard_path)))

        return state_dict

    def init_h2e_module(self):
        self.h2e = H2Embed(self.model.word_embeddings, tau=1.0)

    def load_weights(self, model_path, torch_dtype = torch.bfloat16, device=None):
        self.load_state_dict(model_path, strict=False, dtype=torch_dtype, device=device)
        self.init_h2e_module()


    @add_start_docstrings_to_model_forward(LLaDA2Moe_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoEV2CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        replace_position: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[KVCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, MoEV2CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer

        >>> model = LLaDA2MoeForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists`` of (dec_features, layer_state, dec_hidden, dec_attn)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            replace_position=replace_position,
            **kwargs,
        )

        loss = None
        all_mtp_loss = None
        aux_loss = None
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.vocab_size, **kwargs)

        all_mtp_logits = None
        if self.num_nextn_predict_layers > 0:
            mtp_hidden_states = outputs.mtp_hidden_states
            shift_labels_mtp = None
            for i in range(self.num_nextn_predict_layers):
                mtp_hidden_states = mtp_hidden_states[i]
                mtp_logits = self.lm_head(mtp_hidden_states).float()
                if all_mtp_logits is None:
                    all_mtp_logits = []
                all_mtp_logits.append(mtp_logits)
                if labels is not None:
                    if shift_labels_mtp is None:
                        shift_labels_mtp = labels.clone()
                    shift_labels_mtp, _ = roll_tensor(shift_labels_mtp, shifts=-1, dims=-1, fill_value=-100)
                    mtp_logits_ = mtp_logits.view(-1, self.config.vocab_size)
                    mtp_loss = self.loss_function(mtp_logits_, shift_labels_mtp.to(mtp_logits_.device).view(-1), self.config.vocab_size, **kwargs)
                    if loss is not None:
                        loss += self.mtp_loss_scaling_factor * mtp_loss
                    else:
                        loss = self.mtp_loss_scaling_factor * mtp_loss

                    if all_mtp_loss is None:
                        all_mtp_loss = []
                    all_mtp_loss.append(mtp_loss)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        past_key_values = KVCache(outputs.past_key_values) if outputs.past_key_values is not None else None

        return MoEV2CausalLMOutputWithPast(
            loss=loss,
            mtp_loss=all_mtp_loss,
            aux_loss=aux_loss,
            logits=logits,
            mtp_logits=all_mtp_logits,
            # past_key_values=outputs.past_key_values,
            past_key_values=past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
    def tensor_parallel(self, tp_size):
        """
        Apply the model's tensor parallelization plan.
        Currently only supports linear layers.
        """
        tp_plan = self._tp_plan
        self._tp_size = tp_size

        def _tensor_parallel(module: nn.Module, prefix: str = ""):
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)
                # print(qual_name)
                for pattern, style in tp_plan.items():
                    if re.match(pattern, qual_name) and isinstance(
                            child_module, nn.Linear):
                        new_module = replace_linear_class(
                            child_module, style, None, self.config)
                        dtype = child_module.weight.dtype
                        new_module.weight_loader(new_module.weight, child_module.weight)
                        new_module.weight.data = new_module.weight.data.to(dtype)
                        setattr(module, child_name, new_module)
                        break
                    else:
                        _tensor_parallel(child_module, prefix=qual_name)
                if '.attention' in qual_name and len(qual_name.split('.'))==3:
                    child_module.tp_size = tp_size
            self.h2e.sp_size = tp_size
                # if qual_name == "transformer.ff_out":
                #     new_module = ColumnParallelLinear(child_module.in_features, child_module.out_features, False, True, return_bias=False)
                #     new_module.weight_loader(new_module.weight, child_module.weight)
                #     setattr(module, child_name, new_module)
                    
        _tensor_parallel(self.model)
