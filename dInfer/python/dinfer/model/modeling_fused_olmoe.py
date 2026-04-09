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
"""PyTorch OLMoE model."""

import math
from typing import List, Optional, Tuple, Union, Literal

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
import re
from collections import OrderedDict
from pathlib import Path
import json
from safetensors.torch import load_file
import vllm.distributed as vllm_distributed

def torch_all_reduce(tensor):
    torch.distributed.all_reduce(tensor)
    return tensor
vllm_distributed.tensor_model_parallel_all_reduce = torch_all_reduce


from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)


from .configuration_olmoe import OlmoeConfig
from ..decoding.utils import KVCache
# from .fuse_moe import fused_moe
from vllm.model_executor.layers.fused_moe import FusedMoE, fused_moe

import re
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                        ReplicatedLinear,
                        RowParallelLinear)

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

from torch.nn.modules.normalization import RMSNorm
import torch.distributed as dist
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "OlmoeConfig"


def replace_linear_class(
    linear: nn.Linear, style: Literal["colwise", "rowwise"],
    quant_config,
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
    }.get(style, ReplicatedLinear)

    return vllm_linear_cls(
        input_size=linear.in_features,
        output_size=linear.out_features,
        bias=linear.bias is not None,
        quant_config=quant_config,
        return_bias=False,
    )

OlmoeRMSNorm = RMSNorm
ALL_LAYERNORM_LAYERS.append(OlmoeRMSNorm)

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


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Olmoe
class OlmoeRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[OlmoeConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`OlmoeRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    rotary_dim = cos.shape[-1]

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]

    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_rotated = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rotated = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_final = torch.cat((q_rotated, q_pass), dim=-1)
    k_final = torch.cat((k_rotated, k_pass), dim=-1)

    return q_final, k_final


# Copied from transformers.models.olmo.modeling_olmo.OlmoMLP with Olmo->Olmoe
class OlmoeMLP(nn.Module):
    def __init__(self, config, mlp_type):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        if mlp_type == 'dense':
            self.intermediate_size = config.dense_intermediate_size
        elif mlp_type == 'expert':
            self.intermediate_size = config.expert_intermediate_size
        elif mlp_type == 'shared_expert':
            self.intermediate_size = config.shared_expert_intermediate_size
        else:
            assert False, "unknown mlp type"
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


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


class OlmoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: OlmoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # dLLM
        # self.is_causal = True
        self.is_causal = False
        self.tp_size = 1
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        if config.qk_layernorm:
            # print("qk layernorm set")
            self.q_norm = OlmoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = OlmoeRMSNorm(
                self.head_dim, eps=config.rms_norm_eps
            )
        self.has_qnorm = config.qk_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        if self.has_qnorm:
            query_states = self.q_norm(query_states.reshape(-1, self.head_dim)).reshape(bsz, q_len, -1)
            key_states = self.k_norm(key_states.reshape(-1, self.head_dim)).reshape(bsz, q_len, -1)
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # dLLM
        attention_mask = None

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

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

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class OlmoeSdpaAttention(OlmoeAttention):
    """
    OLMoE attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `OlmoeAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from OlmoeAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        replace_position= None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "OlmoeModel is using OlmoeSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
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

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        if self.has_qnorm:
            query_states = self.q_norm(query_states.reshape(-1, self.head_dim)).reshape(bsz, q_len, -1)
            key_states = self.k_norm(key_states.reshape(-1, self.head_dim)).reshape(bsz, q_len, -1)
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        query_states = query_states.view(bsz, q_len, self.num_heads//self.tp_size, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads//self.tp_size, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads//self.tp_size, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, replace_position)
        
        if use_cache:
            past_key_value = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        # if attention_mask is not None and cache_position is not None:
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # dLLM
        is_causal = False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size//self.tp_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


OLMOE_ATTENTION_CLASSES = {

    "eager": OlmoeAttention,
    # "flash_attention_2": OlmoeFlashAttention2,
    "sdpa": OlmoeSdpaAttention,
}    


class OlmoeMoE(nn.Module):
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
        self.hidden_size = config.hidden_size

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(self.hidden_size,
                                     self.num_experts,
                                     bias=False,
                                     quant_config=None)

        self.experts = FusedMoE(num_experts=self.num_experts,
                                top_k=self.top_k,
                                hidden_size=self.hidden_size,
                                intermediate_size=config.expert_intermediate_size,
                                reduce_results=True,
                                renormalize=False,
                                quant_config=None,
                                tp_size=None,
                                # enable_eplb=True,
                                prefix=f"{prefix}.experts")
        # This is a hack. expert_map in FusedMoE isn't moved to GPU by default.
        # We have to register it explicitly so that it can be moved to GPU with FusedMoE
        expert_map = self.experts.expert_map
        del self.experts.expert_map
        self.experts.register_buffer('expert_map', expert_map)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (num_tokens, n_experts)
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        router_logits, _ = self.gate(hidden_states)
        hidden_states = hidden_states.to(original_dtype)
        final_hidden_states = self.experts.forward_impl(hidden_states=hidden_states,
                                           router_logits=router_logits)
        return final_hidden_states.view(orig_shape)


class OlmoeDecoderLayer(nn.Module):
    def __init__(self, config: OlmoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mlp_type = 'dense' if config.moe_layer_freq[layer_idx] == 0 else 'moe'

        self.self_attn = OLMOE_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = OlmoeMoE(config, prefix=f"model.layers.{layer_idx}.mlp")
        # self.mlp = FusedOlmoeSparseMoeBlock(config) if self.mlp_type == 'moe' else OlmoeMLP(config, 'dense')
        self.input_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.shared_expert_intermediate_size is not None and self.mlp_type == 'moe':
            self.shared_expert = OlmoeMLP(config, 'shared_expert')

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        replace_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # For dLLM
        # use_cache = False
        attention_mask = None

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            replace_position=replace_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        shared_expert_states = hidden_states

        hidden_states = self.mlp(hidden_states.clone())

        if hasattr(self, "shared_expert"):
            hidden_states = hidden_states + self.shared_expert(shared_expert_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


OLMOE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OlmoeConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Olmoe Model outputting raw hidden-states without any specific head on top.",
    OLMOE_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel with Llama->Olmoe
class OlmoePreTrainedModel(PreTrainedModel):
    config_class = OlmoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OlmoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

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


OLMOE_INPUTS_DOCSTRING = r"""
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
        output_router_logits (`bool`, *optional*):
            Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
            should not be returned during inference.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Olmoe Model outputting raw hidden-states without any specific head on top.",
    OLMOE_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaModel with Llama->Olmoe
class OlmoeModel(OlmoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OlmoeDecoderLayer`]

    Args:
        config: OlmoeConfig
    """

    def __init__(self, config: OlmoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [OlmoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OlmoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(OLMOE_INPUTS_DOCSTRING)
    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        replace_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)



        if position_ids is None:
            if replace_position is not None:
                position_ids = torch.arange(replace_position[0], replace_position[1], device=inputs_embeds.device).unsqueeze(0)
            else:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        causal_mask = attention_mask

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = []

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                replace_position=replace_position,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache.extend(layer_outputs[2 if output_attentions else 1])


            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if use_cache:
            next_cache = next_decoder_cache
        else:
            next_cache=None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


def show_modules(module: nn.Module, prefix: str = ""):
    for child_name, child_module in module.named_children():
        qual_name = child_name if prefix == "" else f"{prefix}.{child_name}"
        print(qual_name)
        show_modules(child_module, prefix=qual_name)
    
class FusedOlmoeForCausalLM(OlmoePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OlmoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self._tp_size = 1
        self.post_init()
        self._tp_plan = {
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
        }
        self.init_h2e_module()

    def _keys_to_rename_on_load_unexpected(self):
        return [
            (
                r"^model\.layers\.(\d+)\.mlp\.w1$",
                r"model.layers.\1.mlp.experts.w13_weight"
            ),
            (
                r"^model\.layers\.(\d+)\.mlp\.w2$", 
                r"model.layers.\1.mlp.experts.w2_weight"
            )
        ]
    def load_state_dict(self, state_dict, strict=True, dtype=torch.bfloat16):
        rename_rules = self._keys_to_rename_on_load_unexpected()
        
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for pattern, replacement in rename_rules:
                if re.match(pattern, key):
                    new_key = re.sub(pattern, replacement, key)
                    break
            if new_key.endswith('w13_weight') or new_key.endswith('w2_weight'):
                ep_rank = get_tensor_model_parallel_rank()
                ep_size = get_tensor_model_parallel_world_size()
                size = divide(value.shape[0], ep_size)
                new_state_dict[new_key] = value[ep_rank*size:(ep_rank+1)*size]
            else:
                new_state_dict[new_key] = value
        
        super().load_state_dict(new_state_dict, strict=strict)
        for name, param in self.named_parameters():
            if 'gate' in name:
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


    def load_weights(self, model_path, torch_dtype = torch.bfloat16):
        state_dict = self.load_sharded_safetensors(model_path)
        self.load_state_dict(state_dict, strict=False, dtype=torch_dtype)
        self.init_h2e_module()


    def show_modules(self):
        show_modules(self.model)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def init_h2e_module(self):
        self.h2e = H2Embed(self.model.embed_tokens, tau=1.0)

    @add_start_docstrings_to_model_forward(OLMOE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[KVCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        replace_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OlmoeForCausalLM

        >>> model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0824")
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0824")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'Hey, are you conscious? Can you talk to me?\nI’m not sure if you’re conscious of this, but I’m'
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
            cache_position=cache_position,
            replace_position=replace_position,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        past_key_values = KVCache(outputs.past_key_values) if outputs.past_key_values is not None else None

        return MoeCausalLMOutputWithPast(
            logits=logits,
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
                for pattern, style in tp_plan.items():
                    if re.match(pattern, qual_name) and isinstance(
                            child_module, nn.Linear):
                        new_module = replace_linear_class(
                            child_module, style, None)
                        dtype = child_module.weight.dtype
                        new_module.weight_loader(new_module.weight, child_module.weight)
                        new_module.weight.data = new_module.weight.data.to(dtype)
                        setattr(module, child_name, new_module)
                        break
                    else:
                        _tensor_parallel(child_module, prefix=qual_name)
                if '.self_attn' in qual_name and len(qual_name.split('.'))==3:
                    child_module.tp_size = tp_size
            self.h2e.sp_size = tp_size

                    
        _tensor_parallel(self.model)
