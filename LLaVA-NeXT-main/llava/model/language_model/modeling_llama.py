# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
import os
from functools import partial
from typing import Callable, List, Optional, Tuple, Union
import torch.nn.functional as F
import torch
import torch.utils.checkpoint
from torch import nn
import math
try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice as triton_libdevice

    _TRITON_ROUTER_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    triton_libdevice = None
    _TRITON_ROUTER_AVAILABLE = False

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.llama.configuration_llama import LlamaConfig


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
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

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
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

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
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
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]
        # Plain, eval-only attributes: keeping these outside registered buffers
        # avoids broadcasting several GB of derived weights under DDP.
        self._skip_gate_up_weight = None
        self._skip_gate_up_bias = None
        self._skip_gate_up_weight_fp8 = None
        self._skip_gate_up_weight_scale = None
        self._skip_down_weight_fp8 = None
        self._skip_down_weight_scale = None

    def _shared_gate_up_weight_view(self) -> Optional[torch.Tensor]:
        gate_weight = self.gate_proj.weight
        up_weight = self.up_proj.weight
        if (
            gate_weight.ndim != 2
            or gate_weight.shape != up_weight.shape
            or gate_weight.device != up_weight.device
            or gate_weight.dtype != up_weight.dtype
            or gate_weight.untyped_storage().data_ptr()
            != up_weight.untyped_storage().data_ptr()
            or gate_weight.storage_offset() != 0
            or up_weight.storage_offset() != gate_weight.numel()
            or not gate_weight.is_contiguous()
            or not up_weight.is_contiguous()
        ):
            return None
        rows, columns = gate_weight.shape
        return torch.as_strided(
            gate_weight,
            (2 * rows, columns),
            (columns, 1),
            storage_offset=0,
        )

    def _clear_skip_cache(self):
        self._skip_gate_up_weight = None
        self._skip_gate_up_bias = None
        self._skip_gate_up_weight_fp8 = None
        self._skip_gate_up_weight_scale = None
        self._skip_down_weight_fp8 = None
        self._skip_down_weight_scale = None

    def _build_skip_cache(self):
        if not _skip_fusions_requested() or self.training:
            return
        use_fp8 = _skip_fp8_requested() and self.gate_proj.weight.is_cuda
        if use_fp8 and self._skip_gate_up_weight_fp8 is not None:
            return
        if not use_fp8 and self._skip_gate_up_weight is not None:
            return
        with torch.no_grad():
            gate_up_weight = None
            if not use_fp8 and _shared_packed_weights_requested():
                gate_up_weight = self._shared_gate_up_weight_view()
                if gate_up_weight is None:
                    gate_weight = self.gate_proj.weight
                    up_weight = self.up_proj.weight
                    gate_rows = gate_weight.shape[0]
                    packed_weight = torch.cat(
                        [gate_weight.detach(), up_weight.detach()], dim=0
                    ).contiguous()
                    self.gate_proj.weight = nn.Parameter(
                        packed_weight[:gate_rows],
                        requires_grad=gate_weight.requires_grad,
                    )
                    self.up_proj.weight = nn.Parameter(
                        packed_weight[gate_rows:],
                        requires_grad=up_weight.requires_grad,
                    )
                    gate_up_weight = packed_weight
            if gate_up_weight is None:
                gate_up_weight = torch.cat(
                    [self.gate_proj.weight, self.up_proj.weight], dim=0
                ).contiguous()
            if self.gate_proj.bias is not None:
                self._skip_gate_up_bias = torch.cat(
                    [self.gate_proj.bias, self.up_proj.bias], dim=0
                ).contiguous()
            if use_fp8:
                (
                    self._skip_gate_up_weight_fp8,
                    self._skip_gate_up_weight_scale,
                ) = _fp8_quantize_weight(gate_up_weight)
                (
                    self._skip_down_weight_fp8,
                    self._skip_down_weight_scale,
                ) = _fp8_quantize_weight(self.down_proj.weight)
            else:
                self._skip_gate_up_weight = gate_up_weight

    def skip_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only packed SwiGLU used after a visual-token skip."""
        global _SKIP_FP8_FAILED
        if self.training or not _skip_fusions_requested():
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        self._build_skip_cache()
        use_fp8 = _skip_fp8_requested() and self._skip_gate_up_weight_fp8 is not None
        try:
            if use_fp8:
                gate_up = _fp8_linear_dynamic(
                    x,
                    self._skip_gate_up_weight_fp8,
                    self._skip_gate_up_weight_scale,
                    self._skip_gate_up_bias,
                )
            else:
                gate_up = F.linear(x, self._skip_gate_up_weight, self._skip_gate_up_bias)
            gate, up = gate_up.split(self.intermediate_size, dim=-1)
            hidden = (
                _skip_swiglu(gate, up)
                if self.config.hidden_act in {"silu", "swish"}
                else self.act_fn(gate) * up
            )
            if use_fp8:
                return _fp8_linear_dynamic(
                    hidden,
                    self._skip_down_weight_fp8,
                    self._skip_down_weight_scale,
                    self.down_proj.bias,
                )
            return self.down_proj(hidden)
        except Exception as exc:
            if not use_fp8:
                raise
            _SKIP_FP8_FAILED = True
            logger.warning_once(f"Disabling FP8 skip-path projections after MLP failure: {exc}")
            return self.skip_forward(x)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

    def train(self, mode: bool = True):
        if mode:
            self._clear_skip_cache()
        return super().train(mode)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self._clear_skip_cache()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0 #config.attention_dropout
        #self.attention_bias = False
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self._skip_q_dim = config.num_attention_heads * self.head_dim
        self._skip_kv_dim = config.num_key_value_heads * self.head_dim
        self._skip_qkv_weight = None
        self._skip_qkv_bias = None
        self._skip_qkv_weight_fp8 = None
        self._skip_qkv_weight_scale = None
        self._skip_o_weight_fp8 = None
        self._skip_o_weight_scale = None

    def _shared_qkv_weight_view(self) -> Optional[torch.Tensor]:
        q_weight = self.q_proj.weight
        k_weight = self.k_proj.weight
        v_weight = self.v_proj.weight
        shared_ptr = q_weight.untyped_storage().data_ptr()
        if (
            q_weight.ndim != 2
            or k_weight.ndim != 2
            or v_weight.ndim != 2
            or q_weight.shape[1] != k_weight.shape[1]
            or q_weight.shape[1] != v_weight.shape[1]
            or q_weight.device != k_weight.device
            or q_weight.device != v_weight.device
            or q_weight.dtype != k_weight.dtype
            or q_weight.dtype != v_weight.dtype
            or k_weight.untyped_storage().data_ptr() != shared_ptr
            or v_weight.untyped_storage().data_ptr() != shared_ptr
            or q_weight.storage_offset() != 0
            or k_weight.storage_offset() != q_weight.numel()
            or v_weight.storage_offset() != q_weight.numel() + k_weight.numel()
            or not q_weight.is_contiguous()
            or not k_weight.is_contiguous()
            or not v_weight.is_contiguous()
        ):
            return None
        total_rows = q_weight.shape[0] + k_weight.shape[0] + v_weight.shape[0]
        columns = q_weight.shape[1]
        return torch.as_strided(
            q_weight,
            (total_rows, columns),
            (columns, 1),
            storage_offset=0,
        )

    def _clear_skip_cache(self):
        self._skip_qkv_weight = None
        self._skip_qkv_bias = None
        self._skip_qkv_weight_fp8 = None
        self._skip_qkv_weight_scale = None
        self._skip_o_weight_fp8 = None
        self._skip_o_weight_scale = None

    def _build_skip_cache(self):
        if not _skip_fusions_requested() or self.training:
            return
        use_fp8 = _skip_fp8_requested() and self.q_proj.weight.is_cuda
        if use_fp8 and self._skip_qkv_weight_fp8 is not None:
            return
        if not use_fp8 and self._skip_qkv_weight is not None:
            return
        with torch.no_grad():
            qkv_weight = None
            if not use_fp8 and _shared_packed_weights_requested():
                qkv_weight = self._shared_qkv_weight_view()
                if qkv_weight is None:
                    q_weight = self.q_proj.weight
                    k_weight = self.k_proj.weight
                    v_weight = self.v_proj.weight
                    q_rows = q_weight.shape[0]
                    k_rows = k_weight.shape[0]
                    packed_weight = torch.cat(
                        [q_weight.detach(), k_weight.detach(), v_weight.detach()],
                        dim=0,
                    ).contiguous()
                    self.q_proj.weight = nn.Parameter(
                        packed_weight[:q_rows],
                        requires_grad=q_weight.requires_grad,
                    )
                    self.k_proj.weight = nn.Parameter(
                        packed_weight[q_rows : q_rows + k_rows],
                        requires_grad=k_weight.requires_grad,
                    )
                    self.v_proj.weight = nn.Parameter(
                        packed_weight[q_rows + k_rows :],
                        requires_grad=v_weight.requires_grad,
                    )
                    qkv_weight = packed_weight
            if qkv_weight is None:
                qkv_weight = torch.cat(
                    [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
                    dim=0,
                ).contiguous()
            if self.q_proj.bias is not None:
                self._skip_qkv_bias = torch.cat(
                    [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0
                ).contiguous()
            if use_fp8:
                (
                    self._skip_qkv_weight_fp8,
                    self._skip_qkv_weight_scale,
                ) = _fp8_quantize_weight(qkv_weight)
                (
                    self._skip_o_weight_fp8,
                    self._skip_o_weight_scale,
                ) = _fp8_quantize_weight(self.o_proj.weight)
            else:
                self._skip_qkv_weight = qkv_weight

    def skip_qkv_projection(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One packed eval projection for text-only Q/K/V."""
        global _SKIP_FP8_FAILED
        if self.training or not _skip_fusions_requested():
            return (
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            )
        self._build_skip_cache()
        use_fp8 = _skip_fp8_requested() and self._skip_qkv_weight_fp8 is not None
        try:
            if use_fp8:
                qkv = _fp8_linear_dynamic(
                    hidden_states,
                    self._skip_qkv_weight_fp8,
                    self._skip_qkv_weight_scale,
                    self._skip_qkv_bias,
                )
            else:
                qkv = F.linear(hidden_states, self._skip_qkv_weight, self._skip_qkv_bias)
            return qkv.split(
                [self._skip_q_dim, self._skip_kv_dim, self._skip_kv_dim], dim=-1
            )
        except Exception as exc:
            if not use_fp8:
                raise
            _SKIP_FP8_FAILED = True
            logger.warning_once(f"Disabling FP8 skip-path projections after QKV failure: {exc}")
            return self.skip_qkv_projection(hidden_states)

    def skip_kv_projection(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One packed eval projection for branch-independent K/V.

        The exact immediate Router path computes K/V for every logical token
        regardless of whether Attention later keeps or skips visual queries.
        Reuse the contiguous K/V rows of the existing packed QKV weight so this
        common work does not also compute route-dependent visual Q.
        """
        global _SKIP_FP8_FAILED
        if self.training or not _skip_fusions_requested():
            return self.k_proj(hidden_states), self.v_proj(hidden_states)
        self._build_skip_cache()
        use_fp8 = _skip_fp8_requested() and self._skip_qkv_weight_fp8 is not None
        kv_bias = (
            None
            if self._skip_qkv_bias is None
            else self._skip_qkv_bias[self._skip_q_dim :]
        )
        try:
            if use_fp8:
                kv = _fp8_linear_dynamic(
                    hidden_states,
                    self._skip_qkv_weight_fp8[self._skip_q_dim :],
                    self._skip_qkv_weight_scale,
                    kv_bias,
                )
            else:
                kv = F.linear(
                    hidden_states,
                    self._skip_qkv_weight[self._skip_q_dim :],
                    kv_bias,
                )
            return kv.split([self._skip_kv_dim, self._skip_kv_dim], dim=-1)
        except Exception as exc:
            if not use_fp8:
                raise
            _SKIP_FP8_FAILED = True
            logger.warning_once(
                f"Disabling FP8 skip-path projections after KV failure: {exc}"
            )
            return self.skip_kv_projection(hidden_states)

    def skip_o_projection(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Optional FP8 output projection for the text-only attention path."""
        global _SKIP_FP8_FAILED
        use_fp8 = (
            not self.training
            and _skip_fp8_requested()
            and self._skip_o_weight_fp8 is not None
        )
        if not use_fp8:
            return self.o_proj(hidden_states)
        try:
            return _fp8_linear_dynamic(
                hidden_states,
                self._skip_o_weight_fp8,
                self._skip_o_weight_scale,
                self.o_proj.bias,
            )
        except Exception as exc:
            _SKIP_FP8_FAILED = True
            logger.warning_once(f"Disabling FP8 skip-path projections after O projection failure: {exc}")
            return self.o_proj(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        precomputed_rotated_key_value_states: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        common_kv = precomputed_rotated_key_value_states
        if common_kv is not None:
            if self.training or past_key_value is None:
                raise RuntimeError(
                    "precomputed rotated K/V is eval-only and requires an "
                    "already-updated cache"
                )
            if len(common_kv) != 2:
                raise RuntimeError("precomputed rotated K/V must contain two tensors")
            expected_shape = (
                hidden_states.shape[0],
                self.config.num_key_value_heads,
                hidden_states.shape[1],
                self.head_dim,
            )
            for name, state in zip(("key", "value"), common_kv):
                if (
                    tuple(state.shape) != expected_shape
                    or state.device != hidden_states.device
                    or state.dtype != hidden_states.dtype
                ):
                    raise RuntimeError(
                        f"precomputed rotated {name} must match {expected_shape}, "
                        "device, and dtype of the full Attention input"
                    )

        # A host-side Router decision drains the CUDA queue before this call.
        # On the keep path, three separate Q/K/V launches plus the elementwise
        # RoPE launches then become visibly CPU-launch-bound. Reuse the packed
        # projection and single-kernel RoPE already exercised by the visual-
        # skip path. Limit this to Router prefill: training and decode retain
        # the checkpoint's original execution order, while Router-off remains
        # the matched CDPruner baseline.
        router_prefill_fusions = (
            common_kv is None
            and not self.training
            and hidden_states.shape[1] > 1
            and bool(getattr(self.config, "router_enabled", False))
            and _router_full_attn_fusions_requested()
            and _skip_fusions_requested()
        )
        if common_kv is not None:
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states, value_states = common_kv
        elif router_prefill_fusions:
            query_linear, key_linear, value_linear = self.skip_qkv_projection(hidden_states)
            query_states = query_linear.view(hidden_shape).transpose(1, 2)
            key_states = key_linear.view(hidden_shape).transpose(1, 2)
            value_states = value_linear.view(hidden_shape).transpose(1, 2)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if common_kv is not None:
            query_states = (query_states * cos.unsqueeze(1)) + (
                rotate_half(query_states) * sin.unsqueeze(1)
            )
        elif router_prefill_fusions:
            query_states, key_states = _skip_rope_qk(query_states, key_states, cos, sin)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None and common_kv is None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def train(self, mode: bool = True):
        if mode:
            self._clear_skip_cache()
        return super().train(mode)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self._clear_skip_cache()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


_COMPILED_ROUTER_PAIR_KERNEL = None
_ROUTER_PAIR_COMPILE_FAILED = False
_TRITON_ROUTER_FAILED = False
_TRITON_ROUTER_WARMED_CONFIGS = set()
_SKIP_FUSIONS_FAILED = False
_SKIP_FP8_FAILED = False
_ROUTER_FP8_FAILED = False
def _triton_router_requested() -> bool:
    value = os.environ.get("TRANSFORMERS_LLAMA_TRITON_ROUTER", "0").strip().lower()
    requested = _TRITON_ROUTER_AVAILABLE and value not in {"0", "false", "no", "off", ""}
    skip_norm_mid = (
        os.environ.get("TRANSFORMERS_LLAMA_ROUTER_SKIP_NORM_MID", "0").strip().lower()
        not in {"0", "false", "no", "off", ""}
    )
    if requested and skip_norm_mid:
        logger.warning_once(
            "Using the timing-only Triton Router without Router.norm_mid; "
            "gate values and routing decisions are not checkpoint-equivalent"
        )
    return requested


def _triton_router_text_length_supported(num_text: int) -> bool:
    # The serial-in-text pooling kernel wins for VQA-style short prompts. For
    # long text, PyTorch's parallel reduction is faster, so use it as fallback.
    try:
        max_text_tokens = int(os.environ.get("TRANSFORMERS_LLAMA_TRITON_ROUTER_MAX_TEXT_TOKENS", "64"))
    except ValueError:
        max_text_tokens = 64
    return max_text_tokens <= 0 or num_text <= max_text_tokens


def _skip_fusions_requested() -> bool:
    value = os.environ.get("TRANSFORMERS_LLAMA_SKIP_FUSIONS", "1").strip().lower()
    return value not in {"0", "false", "no", "off", ""}


def _shared_packed_weights_requested() -> bool:
    value = os.environ.get(
        "TRANSFORMERS_LLAMA_SHARED_PACKED_WEIGHTS", "0"
    ).strip().lower()
    return value not in {"0", "false", "no", "off", ""}


def _router_full_attn_fusions_requested() -> bool:
    """Use launch-efficient, FP16-equivalent prefill projections."""
    value = os.environ.get("TRANSFORMERS_LLAMA_ROUTER_FULL_ATTN_FUSIONS", "1").strip().lower()
    return value not in {"0", "false", "no", "off", ""}


def _skip_fp8_requested() -> bool:
    value = os.environ.get("TRANSFORMERS_LLAMA_SKIP_FP8", "0").strip().lower()
    return (
        not _SKIP_FP8_FAILED
        and hasattr(torch, "float8_e4m3fn")
        and value not in {"0", "false", "no", "off", ""}
    )


def _router_fp8_requested() -> bool:
    """Use a per-output-channel E4M3 cache for eval Router down weights."""
    value = os.environ.get("TRANSFORMERS_LLAMA_ROUTER_FP8", "0").strip().lower()
    return (
        not _ROUTER_FP8_FAILED
        and hasattr(torch, "float8_e4m3fn")
        and value not in {"0", "false", "no", "off", ""}
    )


def _fp8_quantize_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tensor-wise E4M3 cache used only by the optional eval skip path."""
    scale = (weight.abs().amax().float() / 448.0).clamp_min(1e-12)
    return (weight / scale).to(torch.float8_e4m3fn).contiguous(), scale


def _fp8_linear_dynamic(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dynamically quantized activation with a pre-quantized row-major weight."""
    output_shape = (*x.shape[:-1], weight_fp8.shape[0])
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    input_scale = (x_2d.abs().amax().float() / 448.0).clamp_min(1e-12)
    x_fp8 = (x_2d / input_scale).to(torch.float8_e4m3fn)
    output = torch._scaled_mm(
        x_fp8,
        weight_fp8.t(),
        input_scale,
        weight_scale,
        bias=bias,
        out_dtype=x.dtype,
        use_fast_accum=True,
    )
    return output.reshape(output_shape)


if _TRITON_ROUTER_AVAILABLE:

    @triton.jit
    def _triton_router_pool_text_kernel(
        text,
        router_input,
        num_text,
        hidden_size: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """Build [mean(text[:-1]), text[-1]] without a separate cat kernel."""
        offsets = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offsets < hidden_size
        acc = tl.zeros((BLOCK_H,), tl.float32)
        text_idx = 0
        while text_idx < num_text - 1:
            values = tl.load(text + text_idx * hidden_size + offsets, mask=mask, other=0.0)
            acc += values.to(tl.float32)
            text_idx += 1
        average = acc / (num_text - 1)
        last = tl.load(text + (num_text - 1) * hidden_size + offsets, mask=mask, other=0.0)
        tl.store(router_input + offsets, average, mask=mask)
        tl.store(router_input + hidden_size + offsets, last, mask=mask)


    @triton.jit
    def _triton_router_input_norm_kernel(
        router_input,
        normalized_input,
        input_size: tl.constexpr,
        rms_eps: tl.constexpr,
        BLOCK_NORM: tl.constexpr,
    ):
        """Normalize one shared Router input once at the model-dtype boundary."""
        offsets = tl.arange(0, BLOCK_NORM)
        mask = offsets < input_size
        values = tl.load(
            router_input + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        inv_rms = tl.rsqrt(
            tl.sum(values * values, axis=0) / input_size + rms_eps
        )
        # Preserve the checkpoint Router's first RMSNorm boundary: the shared
        # normalized activation is rounded to the model dtype before either
        # Router applies its independently trained norm weight.
        values = (values * inv_rms).to(router_input.dtype.element_ty)
        tl.store(normalized_input + offsets, values, mask=mask)


    @triton.jit
    def _triton_router_pair_hidden_kernel(
        router_input,
        normalized_input,
        norm_in_weight,
        down_weight,
        down_weight_scale,
        hidden_output,
        input_size: tl.constexpr,
        router_hidden_size: tl.constexpr,
        num_hidden_blocks: tl.constexpr,
        rms_eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_NORM: tl.constexpr,
        input_is_pre_normalized: tl.constexpr,
        down_weight_is_fp8: tl.constexpr,
    ):
        """Dual normalized GEMV and tanh, with a fail-closed legacy RMS path."""
        pid = tl.program_id(0)
        router_idx = pid // num_hidden_blocks
        hidden_block = pid - router_idx * num_hidden_blocks

        if not input_is_pre_normalized:
            norm_offsets = tl.arange(0, BLOCK_NORM)
            norm_values = tl.load(
                router_input + norm_offsets,
                mask=norm_offsets < input_size,
                other=0.0,
            ).to(tl.float32)
            inv_rms = tl.rsqrt(
                tl.sum(norm_values * norm_values, axis=0) / input_size + rms_eps
            )

        hidden_offsets = hidden_block * BLOCK_D + tl.arange(0, BLOCK_D)
        accumulator = tl.zeros((BLOCK_D,), tl.float32)
        for input_start in range(0, input_size, BLOCK_I):
            input_offsets = input_start + tl.arange(0, BLOCK_I)
            if input_is_pre_normalized:
                normalized = tl.load(
                    normalized_input + input_offsets,
                    mask=input_offsets < input_size,
                    other=0.0,
                )
            else:
                input_values = tl.load(
                    router_input + input_offsets,
                    mask=input_offsets < input_size,
                    other=0.0,
                ).to(tl.float32)
                normalized = (input_values * inv_rms).to(
                    router_input.dtype.element_ty
                )
            norm_weight = tl.load(
                norm_in_weight + router_idx * input_size + input_offsets,
                mask=input_offsets < input_size,
                other=0.0,
            )
            normalized = (normalized * norm_weight).to(router_input.dtype.element_ty)
            weights = tl.load(
                down_weight
                + router_idx * router_hidden_size * input_size
                + hidden_offsets[:, None] * input_size
                + input_offsets[None, :],
                mask=(hidden_offsets[:, None] < router_hidden_size)
                & (input_offsets[None, :] < input_size),
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(
                weights * normalized[None, :].to(tl.float32), axis=1
            )

        if down_weight_is_fp8:
            weight_scale = tl.load(
                down_weight_scale
                + router_idx * router_hidden_size
                + hidden_offsets,
                mask=hidden_offsets < router_hidden_size,
                other=0.0,
            ).to(tl.float32)
            accumulator *= weight_scale

        # Match the eager FP16/BF16 linear boundary before tanh.
        hidden = triton_libdevice.tanh(
            accumulator.to(router_input.dtype.element_ty).to(tl.float32)
        )
        tl.store(
            hidden_output + router_idx * router_hidden_size + hidden_offsets,
            hidden,
            mask=hidden_offsets < router_hidden_size,
        )


    @triton.jit
    def _triton_router_pair_final_kernel(
        hidden,
        norm_mid_weight,
        head_weight,
        head_bias,
        output,
        router_hidden_size: tl.constexpr,
        norm_mid_eps: tl.constexpr,
        logit_clip: tl.constexpr,
        attn_temperature: tl.constexpr,
        mlp_temperature: tl.constexpr,
        skip_norm_mid: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """Fuse checkpoint Router norm_mid, collapsed head, and sigmoid."""
        router_idx = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_H)
        mask = offsets < router_hidden_size
        values = tl.load(
            hidden + router_idx * router_hidden_size + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if not skip_norm_mid:
            inv_rms = tl.rsqrt(
                tl.sum(values * values, axis=0) / router_hidden_size
                + norm_mid_eps
            )
            values = (values * inv_rms).to(hidden.dtype.element_ty)
            mid_weight = tl.load(
                norm_mid_weight + router_idx * router_hidden_size + offsets,
                mask=mask,
                other=0.0,
            )
            values = (values * mid_weight).to(hidden.dtype.element_ty).to(tl.float32)
        head = tl.load(
            head_weight + router_idx * router_hidden_size + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(values * head, axis=0) + tl.load(head_bias + router_idx)
        logits = tl.maximum(-logit_clip, tl.minimum(logit_clip, logits))
        temperature = tl.where(router_idx == 0, attn_temperature, mlp_temperature)
        tl.store(output + router_idx, tl.sigmoid(logits / temperature))


    @triton.jit
    def _triton_skip_rms_norm_kernel(
        x,
        weight,
        output,
        width: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < width
        values = tl.load(x + row * width + offsets, mask=mask, other=0.0).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(values * values, axis=0) / width + eps)
        scales = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(output + row * width + offsets, values * inverse_rms * scales, mask=mask)


    @triton.jit
    def _triton_skip_rope_qk_kernel(
        query,
        key,
        cos,
        sin,
        q_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        q_stride_head: tl.constexpr,
        q_stride_token: tl.constexpr,
        k_stride_head: tl.constexpr,
        k_stride_token: tl.constexpr,
        cos_stride_token: tl.constexpr,
        sin_stride_token: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        head = pid % (q_heads + kv_heads)
        token = pid // (q_heads + kv_heads)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < head_dim
        is_query = head < q_heads
        local_head = tl.where(is_query, head, head - q_heads)
        query_ptr = query + token * q_stride_token + local_head * q_stride_head
        key_ptr = key + token * k_stride_token + local_head * k_stride_head
        query_values = tl.load(query_ptr + offsets, mask=mask & is_query, other=0.0)
        key_values = tl.load(key_ptr + offsets, mask=mask & ~is_query, other=0.0)
        paired_offsets = tl.where(
            offsets < head_dim // 2,
            offsets + head_dim // 2,
            offsets - head_dim // 2,
        )
        paired_query = tl.load(query_ptr + paired_offsets, mask=mask & is_query, other=0.0)
        paired_key = tl.load(key_ptr + paired_offsets, mask=mask & ~is_query, other=0.0)
        rotated_query = tl.where(offsets < head_dim // 2, -paired_query, paired_query)
        rotated_key = tl.where(offsets < head_dim // 2, -paired_key, paired_key)
        cosine = tl.load(cos + token * cos_stride_token + offsets, mask=mask, other=0.0)
        sine = tl.load(sin + token * sin_stride_token + offsets, mask=mask, other=0.0)
        tl.store(
            query_ptr + offsets,
            query_values * cosine + rotated_query * sine,
            mask=mask & is_query,
        )
        tl.store(
            key_ptr + offsets,
            key_values * cosine + rotated_key * sine,
            mask=mask & ~is_query,
        )


    @triton.jit
    def _triton_skip_swiglu_kernel(
        gate,
        up,
        output,
        width: tl.constexpr,
        gate_stride_row: tl.constexpr,
        up_stride_row: tl.constexpr,
        blocks_per_row: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid // blocks_per_row
        offsets = (pid % blocks_per_row) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < width
        gate_values = tl.load(
            gate + row * gate_stride_row + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        up_values = tl.load(
            up + row * up_stride_row + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        result = gate_values * tl.sigmoid(gate_values) * up_values
        tl.store(output + row * width + offsets, result, mask=mask)


def _skip_rms_norm(module: "LlamaRMSNorm", x: torch.Tensor) -> torch.Tensor:
    global _SKIP_FUSIONS_FAILED
    width = x.shape[-1]
    if (
        not _skip_fusions_requested()
        or _SKIP_FUSIONS_FAILED
        or not _TRITON_ROUTER_AVAILABLE
        or module.training
        or not x.is_cuda
        or x.dtype not in (torch.float16, torch.bfloat16)
        or not x.is_contiguous()
        or width > 65536
    ):
        return module(x)
    try:
        output = torch.empty_like(x)
        rows = x.numel() // width
        _triton_skip_rms_norm_kernel[(rows,)](
            x,
            module.weight,
            output,
            width=width,
            eps=module.variance_epsilon,
            BLOCK=triton.next_power_of_2(width),
            num_warps=8,
        )
        return output
    except Exception as exc:
        _SKIP_FUSIONS_FAILED = True
        logger.warning_once(f"Disabling Triton skip-path fusions after RMSNorm failure: {exc}")
        return module(x)


def _skip_rope_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    global _SKIP_FUSIONS_FAILED
    if (
        not _skip_fusions_requested()
        or _SKIP_FUSIONS_FAILED
        or not _TRITON_ROUTER_AVAILABLE
        or not query.is_cuda
        or query.size(0) != 1
        or key.size(0) != 1
        or query.stride(-1) != 1
        or key.stride(-1) != 1
        or cos.stride(-1) != 1
        or sin.stride(-1) != 1
    ):
        return apply_rotary_pos_emb(query, key, cos, sin)
    try:
        num_tokens = query.shape[2]
        q_heads = query.shape[1]
        kv_heads = key.shape[1]
        head_dim = query.shape[-1]
        _triton_skip_rope_qk_kernel[(num_tokens * (q_heads + kv_heads),)](
            query,
            key,
            cos,
            sin,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            q_stride_head=query.stride(1),
            q_stride_token=query.stride(2),
            k_stride_head=key.stride(1),
            k_stride_token=key.stride(2),
            cos_stride_token=cos.stride(1),
            sin_stride_token=sin.stride(1),
            BLOCK=triton.next_power_of_2(head_dim),
            num_warps=1,
        )
        return query, key
    except Exception as exc:
        _SKIP_FUSIONS_FAILED = True
        logger.warning_once(f"Disabling Triton skip-path fusions after RoPE failure: {exc}")
        return apply_rotary_pos_emb(query, key, cos, sin)


def _skip_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    global _SKIP_FUSIONS_FAILED
    width = gate.shape[-1]
    if (
        not _skip_fusions_requested()
        or _SKIP_FUSIONS_FAILED
        or not _TRITON_ROUTER_AVAILABLE
        or not gate.is_cuda
        or gate.dtype not in (torch.float16, torch.bfloat16)
        or gate.stride(-1) != 1
        or up.stride(-1) != 1
    ):
        return F.silu(gate) * up
    try:
        output = torch.empty_like(gate)
        rows = gate.numel() // width
        block = 256
        blocks_per_row = triton.cdiv(width, block)
        _triton_skip_swiglu_kernel[(rows * blocks_per_row,)](
            gate,
            up,
            output,
            width=width,
            gate_stride_row=gate.stride(-2),
            up_stride_row=up.stride(-2),
            blocks_per_row=blocks_per_row,
            BLOCK=block,
            num_warps=4,
        )
        return output
    except Exception as exc:
        _SKIP_FUSIONS_FAILED = True
        logger.warning_once(f"Disabling Triton skip-path fusions after SwiGLU failure: {exc}")
        return F.silu(gate) * up


def _router_pair_kernel(
    x: torch.Tensor,
    norm_in_weight: torch.Tensor,
    down_weight: torch.Tensor,
    norm_mid_weight: torch.Tensor,
    head_weight: torch.Tensor,
    head_bias: torch.Tensor,
    norm_in_eps: float,
    norm_mid_eps: float,
    logit_clip: float,
    temperature: Union[float, torch.Tensor],
    skip_norm_mid: bool,
) -> torch.Tensor:
    """Functional dual-router kernel shared by every decoder layer."""
    input_dtype = x.dtype
    z = x.float()
    z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + norm_in_eps)
    z = z.to(input_dtype).unsqueeze(-2) * norm_in_weight
    h = torch.tanh(torch.einsum("...ri,rdi->...rd", z, down_weight))
    if not skip_norm_mid:
        h_float = h.float()
        h = h_float * torch.rsqrt(h_float.pow(2).mean(-1, keepdim=True) + norm_mid_eps)
        h = h.to(input_dtype) * norm_mid_weight
    logits = (h * head_weight).sum(dim=-1) + head_bias
    logits = logits.clamp(-logit_clip, logit_clip)
    return torch.sigmoid(logits / temperature)


def _get_compiled_router_pair_kernel():
    """Compile once per process; tensor arguments let all layers reuse the graph."""
    global _COMPILED_ROUTER_PAIR_KERNEL
    if _COMPILED_ROUTER_PAIR_KERNEL is None:
        _COMPILED_ROUTER_PAIR_KERNEL = torch.compile(
            _router_pair_kernel,
            mode="reduce-overhead",
            fullgraph=True,
        )
    return _COMPILED_ROUTER_PAIR_KERNEL


class InferenceRouterPair(nn.Module):
    """Packed eval-only execution for two independently trained routers."""

    def __init__(self, attn_router: "Router", mlp_router: "Router"):
        super().__init__()
        # Source routers stay registered only under attn_routers/mlp_routers so
        # the training parameters and checkpoint keys remain unchanged.
        object.__setattr__(self, "_attn_router", attn_router)
        object.__setattr__(self, "_mlp_router", mlp_router)
        self.register_buffer("_norm_in_weight", None, persistent=False)
        self.register_buffer("_down_weight", None, persistent=False)
        self.register_buffer("_down_weight_scale", None, persistent=False)
        self.register_buffer("_norm_mid_weight", None, persistent=False)
        self.register_buffer("_head_weight", None, persistent=False)
        self.register_buffer("_head_bias", None, persistent=False)
        # Batch-1 Triton inference cache. The hidden buffer deliberately uses
        # the model dtype so the down -> tanh -> norm_mid boundaries match the
        # checkpoint Router's eager FP16/BF16 execution.
        self.register_buffer("_triton_router_input", None, persistent=False)
        self.register_buffer(
            "_triton_normalized_input", None, persistent=False
        )
        self.register_buffer("_triton_hidden", None, persistent=False)
        self.register_buffer("_triton_output", None, persistent=False)
        # Optional graph of the fixed-shape Router compute kernels.  Text
        # pooling remains outside the graph and refreshes _triton_router_input
        # from the current sample before every replay.
        object.__setattr__(self, "_triton_compute_graph", None)
        object.__setattr__(self, "_triton_compute_graph_key", None)
        object.__setattr__(self, "_triton_compute_graph_stream", None)
        object.__setattr__(self, "_triton_compute_graph_failed", False)
        # Runtime-only host cache for the CPU speculative Router.  Keep these
        # tensors out of the module buffer registry: a later ``model.to(cuda)``
        # must not move the deliberately host-resident weights or pinned input.
        object.__setattr__(self, "_cpu_norm_in_weight", None)
        object.__setattr__(self, "_cpu_down_weight", None)
        object.__setattr__(self, "_cpu_norm_mid_weight", None)
        object.__setattr__(self, "_cpu_head_weight", None)
        object.__setattr__(self, "_cpu_head_bias", None)
        object.__setattr__(self, "_cpu_router_input", None)
        object.__setattr__(self, "_cpu_compute_mode", None)
        object.__setattr__(self, "_router_fp8_active", False)

    @staticmethod
    def _legacy_down_parameter(router: "Router") -> nn.Parameter:
        parameter = router.down.weight
        if parameter is None:
            parameter = router._compacted_down_weight
        if parameter is None:
            raise RuntimeError("Router legacy down Parameter is unavailable")
        return parameter

    @staticmethod
    def _packed_parameter_view(
        first: nn.Parameter, second: nn.Parameter
    ) -> torch.Tensor:
        """Pack two legacy Parameters without retaining duplicate storage.

        The Parameters remain registered on their original Router modules, so
        old checkpoint keys and optimizers continue to see the same objects.
        The returned two-route tensor and both Parameters are views of one
        contiguous storage allocation.
        """
        if (
            first.shape != second.shape
            or first.dtype != second.dtype
            or first.device != second.device
        ):
            raise RuntimeError(
                "Cannot pack Router parameters with different shape, dtype, "
                f"or device: {tuple(first.shape)}/{first.dtype}/{first.device} "
                f"vs {tuple(second.shape)}/{second.dtype}/{second.device}"
            )

        compact_parameters = os.environ.get(
            "TRANSFORMERS_LLAMA_COMPACT_ROUTER_PARAMETERS", "1"
        ).strip().lower() not in {"", "0", "false", "no", "off"}
        if not compact_parameters:
            # Exact legacy allocation path retained for compatibility and
            # controlled A/B memory measurements.
            return torch.stack([first, second]).contiguous()

        # A train -> eval transition clears the derived buffers but deliberately
        # leaves the two legacy Parameters as non-overlapping views. Recreate
        # the packed buffer view directly instead of copying them again.
        if not first.is_meta and not second.is_meta:
            first_storage = first.untyped_storage()
            second_storage = second.untyped_storage()
            if (
                first_storage.data_ptr() == second_storage.data_ptr()
                and first.is_contiguous()
                and second.is_contiguous()
                and second.storage_offset()
                == first.storage_offset() + first.numel()
            ):
                return first.as_strided(
                    (2, *first.shape),
                    (first.numel(), *first.stride()),
                )

        packed = torch.stack([first.detach(), second.detach()]).contiguous()
        # Repoint the existing Parameter objects rather than replacing them.
        # This preserves references held by the legacy ModuleLists and keeps
        # their state-dict names unchanged.
        first.data = packed[0]
        second.data = packed[1]
        return packed

    def _clear_cache(self):
        object.__setattr__(self, "_triton_compute_graph", None)
        object.__setattr__(self, "_triton_compute_graph_key", None)
        object.__setattr__(self, "_triton_compute_graph_stream", None)
        object.__setattr__(self, "_triton_compute_graph_failed", False)
        self._norm_in_weight = None
        self._down_weight = None
        self._down_weight_scale = None
        self._norm_mid_weight = None
        self._head_weight = None
        self._head_bias = None
        self._triton_router_input = None
        self._triton_normalized_input = None
        self._triton_hidden = None
        self._triton_output = None
        object.__setattr__(self, "_cpu_norm_in_weight", None)
        object.__setattr__(self, "_cpu_down_weight", None)
        object.__setattr__(self, "_cpu_norm_mid_weight", None)
        object.__setattr__(self, "_cpu_head_weight", None)
        object.__setattr__(self, "_cpu_head_bias", None)
        object.__setattr__(self, "_cpu_router_input", None)
        object.__setattr__(self, "_cpu_compute_mode", None)
        object.__setattr__(self, "_router_fp8_active", False)

    def _build_head_cache(self):
        if self._head_weight is not None:
            return
        attn_router = self._attn_router
        mlp_router = self._mlp_router
        attn_down = self._legacy_down_parameter(attn_router)
        with torch.no_grad():
            attn_head = attn_router.head.weight.squeeze(0)
            mlp_head = mlp_router.head.weight.squeeze(0)
            head_weight = torch.stack([attn_head, mlp_head], dim=0)
            head_bias = torch.stack([attn_router.head.bias[0], mlp_router.head.bias[0]])

            dtype = attn_down.dtype
            self._head_weight = head_weight.to(dtype=dtype)
            self._head_bias = head_bias.to(dtype=dtype)

    def _build_compiled_cache(self):
        self._build_head_cache()
        if self._down_weight is not None:
            return
        attn_router = self._attn_router
        mlp_router = self._mlp_router
        with torch.no_grad():
            # The torch.compile fallback preserves the original FP16 ordering.
            self._norm_in_weight = self._packed_parameter_view(
                attn_router.norm_in.weight, mlp_router.norm_in.weight
            )
            self._down_weight = self._packed_parameter_view(
                attn_router.down.weight, mlp_router.down.weight
            )
            self._norm_mid_weight = self._packed_parameter_view(
                attn_router.norm_mid.weight, mlp_router.norm_mid.weight
            )

    def compact_parameters_for_inference(self) -> int:
        """Prepare packed inference weights and optionally compact FP8 sources.

        Returns the number of accelerator-resident bytes moved to the Router's
        unregistered CPU compatibility backup. Compacted down parameters are
        restored automatically before training, direct Router use, or loading
        another state dict. Fused heads stay registered.
        """
        if self.training or self._attn_router.training or self._mlp_router.training:
            raise RuntimeError("Router parameter compaction is eval-only")
        self._build_compiled_cache()
        if _router_fp8_requested():
            if (
                not _triton_router_requested()
                or not self._down_weight.is_cuda
                or self._down_weight.dtype
                not in (torch.float16, torch.bfloat16)
            ):
                raise RuntimeError(
                    "Router FP8 requires the CUDA Triton Router with FP16/BF16 "
                    "source weights"
                )
            # Allocate the persistent model-dtype scratch before moving the
            # legacy down Parameters to their exact CPU compatibility copy.
            self._build_triton_cache()
            with torch.no_grad():
                # Per-output-channel scaling materially reduces gate error over
                # tensor-wise scaling while adding only 2 KiB per layer pair.
                down_scale = (
                    self._down_weight.detach()
                    .abs()
                    .amax(dim=-1)
                    .float()
                    .div(448.0)
                    .clamp_min(1e-12)
                )
                self._down_weight = (
                    self._down_weight.detach().float()
                    / down_scale.unsqueeze(-1)
                ).to(torch.float8_e4m3fn).contiguous()
                self._down_weight_scale = down_scale.contiguous()
            object.__setattr__(self, "_router_fp8_active", True)
            # A graph captured during BF16 warmup points at the displaced
            # weight storage and must be recaptured lazily for the FP8 cache.
            object.__setattr__(self, "_triton_compute_graph", None)
            object.__setattr__(self, "_triton_compute_graph_key", None)
            object.__setattr__(self, "_triton_compute_graph_stream", None)
        released = 0
        for router in (self._attn_router, self._mlp_router):
            if self._router_fp8_active:
                released += router._compact_down_parameter_for_inference()
        return released

    def _restore_fp8_down_parameters(self) -> None:
        """Restore the exact BF16/FP16 down path after FP8 is no longer valid."""
        if not self._router_fp8_active:
            return
        self._attn_router._restore_compacted_down_parameter()
        self._mlp_router._restore_compacted_down_parameter()
        self._down_weight = self._packed_parameter_view(
            self._attn_router.down.weight, self._mlp_router.down.weight
        )
        self._down_weight_scale = None
        object.__setattr__(self, "_router_fp8_active", False)
        object.__setattr__(self, "_triton_compute_graph", None)
        object.__setattr__(self, "_triton_compute_graph_key", None)
        object.__setattr__(self, "_triton_compute_graph_stream", None)

    def _build_triton_cache(self):
        self._build_compiled_cache()
        if self._triton_hidden is not None:
            return
        attn_router = self._attn_router
        with torch.no_grad():
            dtype = attn_router.down.weight.dtype
            device = attn_router.down.weight.device
            input_size = attn_router.down.in_features
            router_hidden_size = attn_router.down.out_features
            self._triton_router_input = torch.empty(input_size, device=device, dtype=dtype)
            # This path is intentionally restricted to the frozen formal-run
            # shape/dtype. Other Router configurations retain the original
            # per-hidden-program RMS implementation below.
            if (
                dtype == torch.float16
                and input_size == 8192
                and router_hidden_size == 256
            ):
                self._triton_normalized_input = torch.empty(
                    input_size, device=device, dtype=dtype
                )
            self._triton_hidden = torch.empty(
                (2, router_hidden_size), device=device, dtype=dtype
            )
            self._triton_output = torch.empty((1, 2), device=device, dtype=dtype)

    def _build_cpu_cache(self):
        """Pack one layer's dual Router weights on the host once.

        The CPU path evaluates the checkpoint architecture in FP32 or BF16. It
        is a routing path, not a value-parity oracle for the FP16 Triton kernel;
        its contract is to produce the two gates used by this execution mode.
        """
        if self._cpu_down_weight is not None:
            return
        self._build_head_cache()
        attn_router = self._attn_router
        mlp_router = self._mlp_router
        attn_down = self._legacy_down_parameter(attn_router)
        mlp_down = self._legacy_down_parameter(mlp_router)
        compute_mode = os.environ.get(
            "TRANSFORMERS_LLAMA_CPU_ROUTER_DTYPE", "float32"
        ).strip().lower()
        if compute_mode not in {"float32", "bfloat16"}:
            raise ValueError(
                "TRANSFORMERS_LLAMA_CPU_ROUTER_DTYPE must be 'float32', "
                f"or 'bfloat16', got {compute_mode!r}"
            )
        compute_dtype = (
            torch.bfloat16 if compute_mode == "bfloat16" else torch.float32
        )
        with torch.no_grad():
            object.__setattr__(
                self,
                "_cpu_norm_in_weight",
                torch.stack(
                    [attn_router.norm_in.weight, mlp_router.norm_in.weight]
                )
                .detach()
                .to(device="cpu", dtype=compute_dtype)
                .contiguous(),
            )
            object.__setattr__(
                self,
                "_cpu_down_weight",
                torch.stack(
                    [attn_down, mlp_down]
                )
                .detach()
                .to(device="cpu", dtype=compute_dtype)
                .contiguous(),
            )
            object.__setattr__(
                self,
                "_cpu_norm_mid_weight",
                torch.stack(
                    [attn_router.norm_mid.weight, mlp_router.norm_mid.weight]
                )
                .detach()
                .to(device="cpu", dtype=compute_dtype)
                .contiguous(),
            )
            object.__setattr__(
                self,
                "_cpu_head_weight",
                self._head_weight.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous(),
            )
            object.__setattr__(
                self,
                "_cpu_head_bias",
                self._head_bias.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous(),
            )
            object.__setattr__(
                self,
                "_cpu_router_input",
                torch.empty(
                    attn_router.down.in_features,
                    device="cpu",
                    dtype=attn_down.dtype,
                    pin_memory=torch.cuda.is_available(),
                ),
            )
            object.__setattr__(self, "_cpu_compute_mode", compute_mode)

    def cpu_router_input_buffer(self) -> torch.Tensor:
        self._build_cpu_cache()
        return self._cpu_router_input

    def pool_hidden_states_for_cpu(
        self,
        hidden_states: torch.Tensor,
        text_start: int,
    ) -> Optional[torch.Tensor]:
        """Produce ``[mean(text[:-1]), last(text)]`` on the current stream."""
        attn_router = self._attn_router
        if (
            hidden_states.ndim != 3
            or hidden_states.size(0) != 1
            or 2 * hidden_states.size(-1) != attn_router.down.in_features
            or text_start < 0
            or hidden_states.size(1) - text_start < 2
        ):
            return None

        num_text = hidden_states.size(1) - text_start
        if (
            hidden_states.is_cuda
            and _TRITON_ROUTER_AVAILABLE
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and hidden_states.stride(-1) == 1
            and hidden_states.stride(-2) == hidden_states.size(-1)
            and _triton_router_text_length_supported(num_text)
        ):
            if self._triton_router_input is None:
                self._build_triton_cache()
            text = hidden_states[0, text_start:, :]
            _triton_router_pool_text_kernel[(triton.cdiv(text.size(-1), 256),)](
                text,
                self._triton_router_input,
                num_text,
                hidden_size=text.size(-1),
                BLOCK_H=256,
                num_warps=4,
            )
            return self._triton_router_input

        text = hidden_states[0, text_start:, :]
        return torch.cat([text[:-1].mean(dim=0), text[-1]], dim=0)

    def forward_from_cpu_input(
        self,
        router_input: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Evaluate the packed dual Router synchronously on CPU."""
        self._build_cpu_cache()
        if router_input.device.type != "cpu":
            raise RuntimeError("CPU Router input must reside on the host")
        expected_input_size = self._attn_router.down.in_features
        if router_input.numel() != expected_input_size:
            raise RuntimeError(
                "CPU Router input width mismatch: "
                f"expected {expected_input_size}, got "
                f"{router_input.numel()}"
            )

        attn_router = self._attn_router
        mlp_router = self._mlp_router
        compute_dtype = (
            torch.bfloat16
            if self._cpu_compute_mode == "bfloat16"
            else torch.float32
        )
        x = router_input.reshape(1, -1).to(dtype=compute_dtype).expand(2, -1)
        z = x * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True)
            + attn_router.norm_in.variance_epsilon
        ).to(dtype=compute_dtype)
        z = z * self._cpu_norm_in_weight
        hidden = torch.bmm(
            self._cpu_down_weight, z.unsqueeze(-1)
        ).squeeze(-1)
        hidden = torch.tanh(hidden)
        if os.environ.get(
            "TRANSFORMERS_LLAMA_ROUTER_SKIP_NORM_MID", "0"
        ).strip().lower() in {"0", "false", "no", "off", ""}:
            hidden = hidden * torch.rsqrt(
                hidden.float().square().mean(dim=-1, keepdim=True)
                + attn_router.norm_mid.variance_epsilon
            ).to(dtype=hidden.dtype)
            hidden = hidden * self._cpu_norm_mid_weight
        logits = (
            (hidden * self._cpu_head_weight).sum(dim=-1)
            + self._cpu_head_bias
        ).float()
        logits = logits.clamp(-attn_router.logit_clip, attn_router.logit_clip)
        if temperature is None:
            temperatures = logits.new_tensor(
                [attn_router.gate_temperature, mlp_router.gate_temperature]
            )
        else:
            temperatures = logits.new_full((2,), float(temperature))
        gates = torch.sigmoid(logits / temperatures)
        return float(gates[0]), float(gates[1])

    def _build_cache(self):
        """Warm the preferred inference cache before DDP buffer discovery."""
        global _TRITON_ROUTER_FAILED
        attn_router = self._attn_router
        if (
            _triton_router_requested()
            and not _TRITON_ROUTER_FAILED
            and attn_router.down.weight.is_cuda
            and attn_router.down.weight.dtype in (torch.float16, torch.bfloat16)
        ):
            try:
                self._build_triton_cache()
                self._warmup_triton_cache()
            except Exception as exc:
                _TRITON_ROUTER_FAILED = True
                self._triton_router_input = None
                self._triton_normalized_input = None
                self._triton_hidden = None
                self._triton_output = None
                logger.warning_once(
                    f"Falling back to torch.compile dual-router inference because the Triton kernel failed: {exc}"
                )
                self._build_compiled_cache()
        else:
            self._build_compiled_cache()

    def _launch_triton_router_input_eager(
        self,
        router_input: torch.Tensor,
        attn_temperature: float,
        mlp_temperature: float,
        use_pre_normalized_input: Optional[bool] = None,
    ) -> torch.Tensor:
        attn_router = self._attn_router
        input_size = attn_router.down.in_features
        router_hidden_size = attn_router.down.out_features
        pre_normalize_supported = bool(
            self._triton_normalized_input is not None
            and router_input.is_cuda
            and router_input.dtype == torch.float16
            and router_input.is_contiguous()
            and router_input.numel() == 8192
            and input_size == 8192
            and router_hidden_size == 256
        )
        if use_pre_normalized_input is None:
            use_pre_normalized_input = pre_normalize_supported
        elif use_pre_normalized_input and not pre_normalize_supported:
            raise RuntimeError(
                "single-pass Router input normalization is restricted to "
                "contiguous CUDA FP16 input=8192 and hidden=256"
            )
        if use_pre_normalized_input:
            _triton_router_input_norm_kernel[(1,)](
                router_input,
                self._triton_normalized_input,
                input_size=input_size,
                rms_eps=attn_router.norm_in.variance_epsilon,
                BLOCK_NORM=triton.next_power_of_2(input_size),
                num_warps=4,
            )
        normalized_input = (
            self._triton_normalized_input
            if use_pre_normalized_input
            else router_input
        )
        hidden_block = 8
        num_hidden_blocks = triton.cdiv(router_hidden_size, hidden_block)
        _triton_router_pair_hidden_kernel[(2 * num_hidden_blocks,)](
            router_input,
            normalized_input,
            self._norm_in_weight,
            self._down_weight,
            (
                self._down_weight_scale
                if self._down_weight_scale is not None
                else self._head_weight
            ),
            self._triton_hidden,
            input_size=input_size,
            router_hidden_size=router_hidden_size,
            num_hidden_blocks=num_hidden_blocks,
            rms_eps=attn_router.norm_in.variance_epsilon,
            BLOCK_D=hidden_block,
            BLOCK_I=128,
            BLOCK_NORM=triton.next_power_of_2(input_size),
            input_is_pre_normalized=bool(use_pre_normalized_input),
            down_weight_is_fp8=bool(self._router_fp8_active),
            num_warps=4,
        )
        skip_norm_mid = (
            os.environ.get("TRANSFORMERS_LLAMA_ROUTER_SKIP_NORM_MID", "0").strip().lower()
            not in {"0", "false", "no", "off", ""}
        )
        _triton_router_pair_final_kernel[(2,)](
            self._triton_hidden,
            self._norm_mid_weight,
            self._head_weight,
            self._head_bias,
            self._triton_output,
            router_hidden_size=router_hidden_size,
            norm_mid_eps=attn_router.norm_mid.variance_epsilon,
            logit_clip=attn_router.logit_clip,
            attn_temperature=attn_temperature,
            mlp_temperature=mlp_temperature,
            skip_norm_mid=skip_norm_mid,
            BLOCK_H=triton.next_power_of_2(router_hidden_size),
            num_warps=4,
        )
        return self._triton_output

    def _triton_compute_graph_requested(self) -> bool:
        value = os.environ.get(
            "TRANSFORMERS_LLAMA_TRITON_ROUTER_COMPUTE_GRAPH", "0"
        ).strip().lower()
        return value not in {"0", "false", "no", "off", ""}

    def _capture_triton_compute_graph(
        self,
        attn_temperature: float,
        mlp_temperature: float,
        graph_key: Tuple[float, float, bool, bool],
    ) -> None:
        """Capture normalize+hidden+final Router work over fixed buffers."""
        device = self._triton_router_input.device
        current_stream = torch.cuda.current_stream(device)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            # CUDA Graph capture requires kernels and runtime state to be warm
            # on a side stream first.
            for _ in range(3):
                self._launch_triton_router_input_eager(
                    self._triton_router_input,
                    attn_temperature,
                    mlp_temperature,
                    use_pre_normalized_input=graph_key[3],
                )
        capture_stream.synchronize()

        compute_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(compute_graph, stream=capture_stream):
            self._launch_triton_router_input_eager(
                self._triton_router_input,
                attn_temperature,
                mlp_temperature,
                use_pre_normalized_input=graph_key[3],
            )
        current_stream.wait_stream(capture_stream)
        object.__setattr__(self, "_triton_compute_graph", compute_graph)
        object.__setattr__(self, "_triton_compute_graph_key", graph_key)
        object.__setattr__(self, "_triton_compute_graph_stream", capture_stream)

    def _launch_triton_router_input(
        self,
        router_input: torch.Tensor,
        attn_temperature: float,
        mlp_temperature: float,
    ) -> torch.Tensor:
        # A graph is safe only for the persistent input buffer whose address
        # was captured.  The general ``forward(x)`` path may pass a temporary
        # tensor and therefore stays on ordinary kernel launches.
        graph_eligible = (
            self._triton_compute_graph_requested()
            and not self._triton_compute_graph_failed
            and self._triton_router_input is not None
            and router_input.data_ptr() == self._triton_router_input.data_ptr()
            and not torch.cuda.is_current_stream_capturing()
        )
        use_pre_normalized_input = bool(
            self._triton_normalized_input is not None
            and router_input.is_cuda
            and router_input.dtype == torch.float16
            and router_input.is_contiguous()
            and router_input.numel() == 8192
            and self._attn_router.down.in_features == 8192
            and self._attn_router.down.out_features == 256
        )
        if graph_eligible:
            skip_norm_mid = (
                os.environ.get(
                    "TRANSFORMERS_LLAMA_ROUTER_SKIP_NORM_MID", "0"
                ).strip().lower()
                not in {"0", "false", "no", "off", ""}
            )
            graph_key = (
                float(attn_temperature),
                float(mlp_temperature),
                bool(skip_norm_mid),
                use_pre_normalized_input,
            )
            if self._triton_compute_graph is None:
                try:
                    self._capture_triton_compute_graph(
                        attn_temperature,
                        mlp_temperature,
                        graph_key,
                    )
                except Exception as exc:
                    object.__setattr__(
                        self, "_triton_compute_graph_failed", True
                    )
                    logger.warning_once(
                        "Disabling the exact Triton Router compute graph after "
                        f"capture failure: {exc}"
                    )
            if (
                self._triton_compute_graph is not None
                and self._triton_compute_graph_key == graph_key
            ):
                self._triton_compute_graph.replay()
                return self._triton_output

        return self._launch_triton_router_input_eager(
            router_input,
            attn_temperature,
            mlp_temperature,
            use_pre_normalized_input=use_pre_normalized_input,
        )

    def _launch_triton_router(
        self,
        text: torch.Tensor,
        num_text: int,
        attn_temperature: float,
        mlp_temperature: float,
    ) -> torch.Tensor:
        hidden_size = text.size(-1)
        _triton_router_pool_text_kernel[(triton.cdiv(hidden_size, 256),)](
            text,
            self._triton_router_input,
            num_text,
            hidden_size=hidden_size,
            BLOCK_H=256,
            num_warps=4,
        )
        return self._launch_triton_router_input(
            self._triton_router_input,
            attn_temperature=attn_temperature,
            mlp_temperature=mlp_temperature,
        )

    def _warmup_triton_cache(self):
        """Compile/load the kernels at model setup, outside the first request."""
        global _TRITON_ROUTER_WARMED_CONFIGS
        attn_router = self._attn_router
        input_size = attn_router.down.in_features
        warmup_key = (
            attn_router.down.weight.device,
            attn_router.down.weight.dtype,
            input_size,
            attn_router.down.out_features,
        )
        if warmup_key in _TRITON_ROUTER_WARMED_CONFIGS:
            return
        self._triton_router_input.zero_()
        self._launch_triton_router(
            self._triton_router_input.view(2, input_size // 2),
            num_text=2,
            attn_temperature=attn_router.gate_temperature,
            mlp_temperature=self._mlp_router.gate_temperature,
        )
        torch.cuda.synchronize(attn_router.down.weight.device)
        _TRITON_ROUTER_WARMED_CONFIGS.add(warmup_key)

    @staticmethod
    def _synchronize_triton_failure_stream(device: torch.device) -> None:
        """Drain failed Triton work before releasing its persistent buffers.

        This intentionally does not catch synchronization failures.  Such an
        error propagates with the original kernel exception as its context, and
        the caller therefore never reaches the buffer-clearing statements.
        """
        torch.cuda.current_stream(device).synchronize()

    def forward_from_hidden_states(
        self,
        hidden_states: torch.Tensor,
        text_start: int,
        temperature: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        """Fused batch-1 text pooling and dual-router inference.

        Returns ``None`` when the specialized CUDA path is not applicable so
        the caller can transparently use the general torch.compile path.
        """
        global _TRITON_ROUTER_FAILED, _ROUTER_FP8_FAILED
        attn_router = self._attn_router
        mlp_router = self._mlp_router
        if (
            self.training
            or not _triton_router_requested()
            or _TRITON_ROUTER_FAILED
            or not hidden_states.is_cuda
            or hidden_states.dtype not in (torch.float16, torch.bfloat16)
            or hidden_states.ndim != 3
            or hidden_states.size(0) != 1
            or hidden_states.stride(-1) != 1
            or hidden_states.stride(-2) != hidden_states.size(-1)
            or 2 * hidden_states.size(-1) != attn_router.down.in_features
            or text_start < 0
            or hidden_states.size(1) - text_start < 2
            or not _triton_router_text_length_supported(hidden_states.size(1) - text_start)
            or attn_router.down.out_features != mlp_router.down.out_features
            or attn_router.norm_in.variance_epsilon != mlp_router.norm_in.variance_epsilon
            or attn_router.norm_mid.variance_epsilon != mlp_router.norm_mid.variance_epsilon
            or attn_router.logit_clip != mlp_router.logit_clip
            or torch.is_tensor(temperature)
        ):
            return None

        if temperature is None:
            attn_temperature = attn_router.gate_temperature
            mlp_temperature = mlp_router.gate_temperature
        else:
            attn_temperature = float(temperature)
            mlp_temperature = float(temperature)

        try:
            if self._triton_hidden is None:
                self._build_triton_cache()

            num_text = hidden_states.size(1) - text_start
            text = hidden_states[0, text_start:, :]
            return self._launch_triton_router(
                text,
                num_text=num_text,
                attn_temperature=attn_temperature,
                mlp_temperature=mlp_temperature,
            )
        except Exception as exc:
            _TRITON_ROUTER_FAILED = True
            restore_fp8 = self._router_fp8_active
            if restore_fp8:
                _ROUTER_FP8_FAILED = True
            self._synchronize_triton_failure_stream(hidden_states.device)
            if restore_fp8:
                self._restore_fp8_down_parameters()
            self._triton_router_input = None
            self._triton_normalized_input = None
            self._triton_hidden = None
            self._triton_output = None
            logger.warning_once(
                f"Falling back to torch.compile dual-router inference because the Triton kernel failed: {exc}"
            )
            return None

    def forward(self, x: torch.Tensor, temperature: Optional[float] = None) -> torch.Tensor:
        global _ROUTER_PAIR_COMPILE_FAILED, _TRITON_ROUTER_FAILED, _ROUTER_FP8_FAILED
        if self._down_weight is None:
            self._build_compiled_cache()

        attn_router = self._attn_router
        mlp_router = self._mlp_router
        triton_supported = (
            not self.training
            and _triton_router_requested()
            and not _TRITON_ROUTER_FAILED
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and x.ndim == 2
            and x.size(0) == 1
            and x.size(1) == attn_router.down.in_features
            and x.is_contiguous()
            and attn_router.down.out_features == mlp_router.down.out_features
            and attn_router.norm_in.variance_epsilon == mlp_router.norm_in.variance_epsilon
            and attn_router.norm_mid.variance_epsilon == mlp_router.norm_mid.variance_epsilon
            and attn_router.logit_clip == mlp_router.logit_clip
            and not torch.is_tensor(temperature)
        )
        if triton_supported:
            if temperature is None:
                attn_temperature = attn_router.gate_temperature
                mlp_temperature = mlp_router.gate_temperature
            else:
                attn_temperature = float(temperature)
                mlp_temperature = float(temperature)
            try:
                if self._triton_hidden is None:
                    self._build_triton_cache()
                return self._launch_triton_router_input(
                    x,
                    attn_temperature=attn_temperature,
                    mlp_temperature=mlp_temperature,
                )
            except Exception as exc:
                _TRITON_ROUTER_FAILED = True
                restore_fp8 = self._router_fp8_active
                if restore_fp8:
                    _ROUTER_FP8_FAILED = True
                self._synchronize_triton_failure_stream(x.device)
                if restore_fp8:
                    self._restore_fp8_down_parameters()
                self._triton_router_input = None
                self._triton_normalized_input = None
                self._triton_hidden = None
                self._triton_output = None
                logger.warning_once(
                    f"Falling back to torch.compile dual-router inference because the exact Triton kernel failed: {exc}"
                )

        # The specialized FP8 cache is intentionally batch-1 Triton-only. Any
        # unsupported direct call restores the exact legacy path rather than
        # silently evaluating mixed-dtype weights in the compiled fallback.
        if self._router_fp8_active:
            self._restore_fp8_down_parameters()

        if temperature is None:
            temperature = attn_router.gate_temperature
            if mlp_router.gate_temperature != temperature:
                temperature = x.new_tensor([temperature, mlp_router.gate_temperature])

        kernel_args = (
            x,
            self._norm_in_weight,
            self._down_weight,
            self._norm_mid_weight,
            self._head_weight,
            self._head_bias,
            attn_router.norm_in.variance_epsilon,
            attn_router.norm_mid.variance_epsilon,
            attn_router.logit_clip,
            temperature,
            os.environ.get("TRANSFORMERS_LLAMA_ROUTER_SKIP_NORM_MID", "0").strip().lower()
            not in {"0", "false", "no", "off", ""},
        )
        compile_router = os.environ.get("TRANSFORMERS_LLAMA_COMPILE_ROUTER", "1").strip().lower()
        if x.is_cuda and compile_router not in {"0", "false", "no", "off", ""} and not _ROUTER_PAIR_COMPILE_FAILED:
            try:
                return _get_compiled_router_pair_kernel()(*kernel_args)
            except Exception as exc:
                _ROUTER_PAIR_COMPILE_FAILED = True
                logger.warning_once(
                    f"Falling back to eager dual-router inference because torch.compile failed: {exc}"
                )
        return _router_pair_kernel(*kernel_args)

    def train(self, mode: bool = True):
        if mode:
            self._restore_fp8_down_parameters()
            self._clear_cache()
        return super().train(mode)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self._restore_fp8_down_parameters()
        self._clear_cache()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )



@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

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


LLAMA_INPUTS_DOCSTRING = r"""
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
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
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
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""

import math
import torch
import torch.nn as nn

class Router(nn.Module):
    """Router with the consecutive up/head projections fused into one head."""

    def __init__(
        self,
        hidden_size: int,
        bottleneck_ratio: int,
        rms_eps: float,
        gate_temperature: float = 1,
        logit_clip: float = 8.0,
    ):
        super().__init__()
        input_size = 2 * hidden_size
        dr = max(1, hidden_size // bottleneck_ratio)
        self.norm_in = LlamaRMSNorm(input_size, eps=rms_eps)
        self.down = nn.Linear(input_size, dr, bias=False)
        self.norm_mid = LlamaRMSNorm(dr, eps=rms_eps)
        self.head = nn.Linear(dr, 1, bias=True)
        object.__setattr__(self, "_compacted_down_weight", None)
        self.gate_temperature = float(gate_temperature)
        self.logit_clip = float(logit_clip)

    def forward(self, x: torch.Tensor, temperature: Optional[float] = None) -> torch.Tensor:
        self._restore_compacted_down_parameter()
        temp = temperature if temperature is not None else self.gate_temperature
        h = self.norm_mid(torch.tanh(self.down(self.norm_in(x))))
        logits = self.head(h).squeeze(-1)
        return torch.sigmoid(logits.clamp(-self.logit_clip, self.logit_clip) / temp)

    def _compact_down_parameter_for_inference(self) -> int:
        """Move the exact legacy down Parameter behind an FP8 eval cache."""
        if self.training:
            raise RuntimeError("Router parameter compaction is eval-only")
        if self._compacted_down_weight is not None:
            return 0
        if self.down.weight is None:
            raise RuntimeError("Router down Parameter is unavailable for compaction")
        parameter = self.down.weight
        released = (
            parameter.numel() * parameter.element_size()
            if parameter.device.type != "cpu"
            else 0
        )
        parameter.data = parameter.detach().to(
            device="cpu", copy=True
        ).contiguous()
        object.__setattr__(self, "_compacted_down_weight", parameter)
        self.down.weight = None
        return released

    def _restore_compacted_down_parameter(self) -> None:
        """Restore the exact down Parameter for training/load/direct use."""
        if self._compacted_down_weight is None:
            return
        parameter = self._compacted_down_weight
        parameter.data = parameter.detach().to(
            device=self.norm_in.weight.device,
            dtype=self.norm_in.weight.dtype,
        )
        self.down.weight = parameter
        object.__setattr__(self, "_compacted_down_weight", None)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self._compacted_down_weight is not None:
            destination[prefix + "down.weight"] = (
                self._compacted_down_weight
                if keep_vars
                else self._compacted_down_weight.detach()
            )

    def train(self, mode: bool = True):
        if mode:
            self._restore_compacted_down_parameter()
        return super().train(mode)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        self._restore_compacted_down_parameter()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Visual token gating routers
        self.router_bottleneck_ratio = 16
        # Separate routers for self-attention and MLP
        self.attn_routers = nn.ModuleList([
            Router(
                hidden_size=config.hidden_size,
                bottleneck_ratio=self.router_bottleneck_ratio,
                rms_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        self.mlp_routers = nn.ModuleList([
            Router(
                hidden_size=config.hidden_size,
                bottleneck_ratio=self.router_bottleneck_ratio,
                rms_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        self.inference_router_pairs = nn.ModuleList([
            InferenceRouterPair(attn, mlp)
            for attn, mlp in zip(self.attn_routers, self.mlp_routers)
        ])
        self.router_visual_kv_cache_on_skip = bool(
            getattr(config, "router_visual_kv_cache_on_skip", False)
        )
        self.config.router_enabled = bool(getattr(config, "router_enabled", True))
        self.visual_token_index = int(getattr(config, "image_token_index", -200))
        object.__setattr__(self, "_router_immediate_gate_host", None)
        object.__setattr__(self, "_router_immediate_gate_copy_done", None)
        object.__setattr__(self, "_router_immediate_gate_device", None)
        self.last_gating_scores_attn: Optional[List[torch.Tensor]] = None
        self.last_gating_scores_mlp: Optional[List[torch.Tensor]] = None
        self.tau = getattr(config, 'router_tau', 0.5)  # threshold for skipping during inference

        # Legacy helper fallback; forward infers the actual image span from input_ids.
        self.visual_start = 35
        self.visual_len_fallback = getattr(config, 'visual_len_fallback', 576)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _compute_visual_end(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: torch.Tensor,
    ) -> int:
        """Backward-compatible wrapper for callers that only need the end."""
        return self._compute_visual_span(input_ids, inputs_embeds)[1]

    def _build_keep_index(self, seq_length: int, device: torch.device, visual_start: int, visual_end: int) -> torch.Tensor:
        """
        Build indices of tokens to keep (excluding visual tokens in the middle).
        Returns a 1D LongTensor of indices.
        """
        vs = visual_start
        ve = visual_end
        if seq_length <= vs:
            return torch.arange(seq_length, device=device, dtype=torch.long)
        ve = min(ve, seq_length)
        if ve <= vs:
            return torch.arange(seq_length, device=device, dtype=torch.long)

        left = torch.arange(0, vs, device=device, dtype=torch.long)
        right = torch.arange(ve, seq_length, device=device, dtype=torch.long)
        return torch.cat([left, right], dim=0)

    def _build_visual_index(self, seq_length: int, device: torch.device, visual_start: int, visual_end: int) -> torch.Tensor:
        """Build indices of visual tokens. Returns 1D LongTensor."""
        vs = visual_start
        ve = visual_end
        if seq_length <= vs:
            return torch.tensor([], device=device, dtype=torch.long)
        ve = min(ve, seq_length)
        if ve <= vs:
            return torch.tensor([], device=device, dtype=torch.long)
        return torch.arange(vs, ve, device=device, dtype=torch.long)

    def _slice_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        keep_idx: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Slice attention mask to remove visual tokens.
        Handles both 2D (flash-attn2 padding mask) and 4D (causal mask) cases.
        """
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            # 4D: [bs, 1, q_len, kv_len]
            # Slice both query and key/value dimensions
            m = attention_mask.index_select(dim=2, index=keep_idx)
            m = m.index_select(dim=3, index=keep_idx)
            return m
        elif attention_mask.dim() == 2:
            # 2D: [bs, seq_len]
            return attention_mask.index_select(dim=1, index=keep_idx)
        else:
            # Shouldn't happen, but return as-is
            return attention_mask

    def _slice_position_embeddings(
        self,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        keep_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Slice position embeddings (cos, sin) to match the kept tokens.
        """
        cos, sin = position_embeddings
        # cos, sin shape: [batch_size, seq_len, head_dim]
        cos_sliced = cos.index_select(dim=1, index=keep_idx)
        sin_sliced = sin.index_select(dim=1, index=keep_idx)
        return cos_sliced, sin_sliced

    def _text_gate_scalar(
        self,
        hidden_states: torch.Tensor,
        router: nn.Module,
        seq_length: int,
        visual_start: int,
        visual_end: int,
        router_temperature: Optional[float] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Training-compatible single-router path."""
        router_input = self._text_router_input(
            hidden_states,
            seq_length,
            visual_start,
            visual_end,
            labels=labels,
        )
        if router_input is None:
            return torch.ones(
                hidden_states.size(0),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        return router(router_input, temperature=router_temperature)

    def _efficient_skip_forward(
        self,
        decoder_layer: LlamaDecoderLayer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        visual_start: Optional[int] = None,
        visual_end: Optional[int] = None,
        **kwargs,
    ) -> Tuple:
        """
        Complete skip forward: visual tokens are entirely bypassed.
        - K, V: computed for all tokens but only stored in KV cache for decoding alignment
        - Q, O_proj: text tokens only
        - Attention: text Q attends to text K/V ONLY (visual K/V completely excluded)
        - MLP: text tokens only
        - Visual tokens: pass through via residual unchanged, zero influence on text
        """
        vs = visual_start if visual_start is not None else self.visual_start
        ve = visual_end if visual_end is not None else vs + self.visual_len_fallback
        #print(self.visual_len_fallback)
        bs, seq_len, _ = hidden_states.shape
        ve = min(ve, seq_len)
        n_visual = ve - vs
        nv_len = seq_len - n_visual
        attn = decoder_layer.self_attn
        device = hidden_states.device

        # ==================== Self-Attention ====================
        residual = hidden_states
        normed = decoder_layer.input_layernorm(hidden_states)

        # K, V: ALL tokens (stored in cache for decoding alignment)
        full_shape = (bs, seq_len, -1, attn.head_dim)
        key_states = attn.k_proj(normed).view(full_shape).transpose(1, 2)
        value_states = attn.v_proj(normed).view(full_shape).transpose(1, 2)

        # Q: text tokens only
        normed_nv = torch.cat([normed[:, :vs], normed[:, ve:]], dim=1)
        nv_shape = (bs, nv_len, -1, attn.head_dim)
        query_states = attn.q_proj(normed_nv).view(nv_shape).transpose(1, 2)

        # RoPE
        cos, sin = position_embeddings
        cos_nv = torch.cat([cos[:, :vs], cos[:, ve:]], dim=1)
        sin_nv = torch.cat([sin[:, :vs], sin[:, ve:]], dim=1)
        query_states = (query_states * cos_nv.unsqueeze(1)) + (
            rotate_half(query_states) * sin_nv.unsqueeze(1)
        )
        key_states = (key_states * cos.unsqueeze(1)) + (
            rotate_half(key_states) * sin.unsqueeze(1)
        )

        # KV cache: store full K/V (including visual) for decoding alignment
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, attn.layer_idx, cache_kwargs
            )

        # Extract text-only K/V for attention (visual K/V excluded)
        k_text = torch.cat([key_states[:, :, :vs], key_states[:, :, ve:]], dim=2)
        v_text = torch.cat([value_states[:, :, :vs], value_states[:, :, ve:]], dim=2)

        # GQA
        k_text = repeat_kv(k_text, attn.num_key_value_groups)
        v_text = repeat_kv(v_text, attn.num_key_value_groups)

        # Attention mask: text Q × text K/V (both dims exclude visual)
        if attention_mask is not None and attention_mask.dim() == 4:
            attn_mask = torch.cat(
                [attention_mask[:, :, :vs, :], attention_mask[:, :, ve:, :]],
                dim=2,
            )
            attn_mask = torch.cat(
                [attn_mask[:, :, :, :vs], attn_mask[:, :, :, ve:]],
                dim=3,
            )
        else:
            dtype = query_states.dtype
            q_pos = torch.cat([
                torch.arange(vs, device=device),
                torch.arange(ve, seq_len, device=device),
            ])
            k_pos = q_pos
            causal = q_pos[:, None] >= k_pos[None, :]
            attn_mask = torch.zeros(nv_len, nv_len, dtype=dtype, device=device)
            attn_mask.masked_fill_(~causal, torch.finfo(dtype).min)
            attn_mask = attn_mask[None, None, :, :]
            if attention_mask is not None and attention_mask.dim() == 2:
                pad_text = torch.cat(
                    [attention_mask[:, :vs], attention_mask[:, ve:]], dim=1
                )
                attn_mask = attn_mask.masked_fill(
                    pad_text[:, None, None, :] == 0, torch.finfo(dtype).min
                )

        query_states = query_states.contiguous()
        k_text = k_text.contiguous()
        v_text = v_text.contiguous()

        attn_output = F.scaled_dot_product_attention(
            query_states, k_text, v_text,
            attn_mask=attn_mask,
            dropout_p=attn.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bs, nv_len, -1)
        attn_output = attn.o_proj(attn_output)

        # Reassemble: text = residual + attn; visual = residual (unchanged)
        residual_nv = torch.cat([residual[:, :vs], residual[:, ve:]], dim=1)
        nv_after_attn = residual_nv + attn_output
        hidden_states = torch.cat(
            [nv_after_attn[:, :vs], residual[:, vs:ve], nv_after_attn[:, vs:]],
            dim=1,
        )

        # ==================== MLP (text tokens only) ====================
        residual = hidden_states
        nv_for_mlp = torch.cat(
            [hidden_states[:, :vs], hidden_states[:, ve:]], dim=1
        )
        nv_for_mlp = decoder_layer.post_attention_layernorm(nv_for_mlp)
        nv_for_mlp = decoder_layer.mlp(nv_for_mlp)

        residual_nv = torch.cat([residual[:, :vs], residual[:, ve:]], dim=1)
        nv_after_mlp = residual_nv + nv_for_mlp
        hidden_states = torch.cat(
            [nv_after_mlp[:, :vs], residual[:, vs:ve], nv_after_mlp[:, vs:]],
            dim=1,
        )

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        return outputs

    def _efficient_skip_self_attn(
        self,
        decoder_layer: LlamaDecoderLayer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_start: Optional[int] = None,
        visual_end: Optional[int] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        text_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        text_cache_position: Optional[torch.LongTensor] = None,
        visual_hidden_states: Optional[torch.Tensor] = None,
        hidden_states_are_compact: bool = False,
        return_compact: bool = False,
        precomputed_input_layernorm: Optional[torch.Tensor] = None,
        precomputed_visual_input_layernorm: Optional[torch.Tensor] = None,
        precomputed_rotated_key_value_states: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        仅对 Self-Attention 部分做「跳过视觉 token」的高效前向，MLP 保持不变。
        返回的是经过 Self-Attention（文本仅与文本交互）后的 hidden_states。
        """
        vs = visual_start if visual_start is not None else self.visual_start
        if hidden_states_are_compact:
            if self.training:
                raise RuntimeError("Compact visual-skip attention is inference-only")
            requested_ve = visual_end if visual_end is not None else vs + self.visual_len_fallback
            n_visual = max(0, requested_ve - vs)
            bs, nv_len, _ = hidden_states.shape
            seq_len = nv_len + n_visual
            ve = min(requested_ve, seq_len)
            n_visual = ve - vs
        else:
            bs, seq_len, _ = hidden_states.shape
            ve = visual_end if visual_end is not None else min(vs + self.visual_len_fallback, seq_len)
            ve = min(ve, seq_len)
            n_visual = ve - vs
            nv_len = seq_len - n_visual
        attn = decoder_layer.self_attn
        device = hidden_states.device
        cache_visual_kv = bool(
            not self.training
            and self.router_visual_kv_cache_on_skip
            and past_key_value is not None
            and n_visual > 0
        )
        common_kv = precomputed_rotated_key_value_states
        if common_kv is not None:
            if (
                self.training
                or not cache_visual_kv
                or not isinstance(past_key_value, DynamicCache)
                or len(common_kv) != 2
                or past_key_value.get_seq_length(attn.layer_idx) != seq_len
            ):
                raise RuntimeError(
                    "precomputed rotated K/V requires eval visual-KV cache "
                    "preservation and one completed DynamicCache prefill write"
                )
            expected_kv_shape = (
                bs,
                self.config.num_key_value_heads,
                seq_len,
                attn.head_dim,
            )
            for name, state in zip(("key", "value"), common_kv):
                if (
                    tuple(state.shape) != expected_kv_shape
                    or state.device != hidden_states.device
                    or state.dtype != hidden_states.dtype
                ):
                    raise RuntimeError(
                        f"precomputed rotated {name} must match the full logical "
                        "skip-Attention sequence, device, and dtype"
                    )
        if precomputed_input_layernorm is not None:
            if self.training:
                raise RuntimeError(
                    "precomputed input layernorm is eval-only"
                )
            if (
                precomputed_input_layernorm.shape != hidden_states.shape
                or precomputed_input_layernorm.device != hidden_states.device
                or precomputed_input_layernorm.dtype != hidden_states.dtype
            ):
                raise RuntimeError(
                    "precomputed input layernorm must match skip-attention "
                    "hidden states"
                )
            if hidden_states_are_compact and cache_visual_kv:
                if (
                    precomputed_visual_input_layernorm is None
                    or visual_hidden_states is None
                    or precomputed_visual_input_layernorm.shape
                    != visual_hidden_states.shape
                    or precomputed_visual_input_layernorm.device
                    != visual_hidden_states.device
                    or precomputed_visual_input_layernorm.dtype
                    != visual_hidden_states.dtype
                ):
                    raise RuntimeError(
                        "compact precomputed input layernorm requires a "
                        "matching visual partition"
                    )
        elif precomputed_visual_input_layernorm is not None:
            raise RuntimeError(
                "visual precomputed input layernorm requires text input layernorm"
            )
        if cache_visual_kv and hidden_states_are_compact:
            expected_visual_shape = (bs, n_visual, hidden_states.shape[-1])
            if (
                visual_hidden_states is None
                or tuple(visual_hidden_states.shape) != expected_visual_shape
                or visual_hidden_states.device != hidden_states.device
                or visual_hidden_states.dtype != hidden_states.dtype
            ):
                actual_shape = (
                    None
                    if visual_hidden_states is None
                    else tuple(visual_hidden_states.shape)
                )
                raise RuntimeError(
                    "Compact visual K/V caching requires visual_hidden_states "
                    f"with shape {expected_visual_shape}, got {actual_shape}"
                )

        residual = hidden_states
        cos, sin = position_embeddings
        if text_position_embeddings is None:
            cos_nv = torch.cat([cos[:, :vs], cos[:, ve:]], dim=1)
            sin_nv = torch.cat([sin[:, :vs], sin[:, ve:]], dim=1)
        else:
            cos_nv, sin_nv = text_position_embeddings

        if self.training:
            # Preserve the original training graph and FP ordering.
            normed = decoder_layer.input_layernorm(hidden_states)
        elif hidden_states_are_compact:
            residual_nv = hidden_states
            normed_nv = (
                precomputed_input_layernorm
                if precomputed_input_layernorm is not None
                else _skip_rms_norm(decoder_layer.input_layernorm, residual_nv)
            )
            normed_visual = None
            if cache_visual_kv:
                normed_visual = (
                    precomputed_visual_input_layernorm
                    if precomputed_visual_input_layernorm is not None
                    else _skip_rms_norm(
                        decoder_layer.input_layernorm, visual_hidden_states
                    )
                )
        elif precomputed_input_layernorm is not None:
            residual_nv = torch.cat(
                [hidden_states[:, :vs], hidden_states[:, ve:]], dim=1
            )
            # RMSNorm is token-wise.  Slicing one full-shape result is
            # mathematically identical to normalizing the text and visual
            # partitions separately, while allowing this mandatory work to be
            # queued before the exact host branch decision is available.
            normed_nv = torch.cat(
                [
                    precomputed_input_layernorm[:, :vs],
                    precomputed_input_layernorm[:, ve:],
                ],
                dim=1,
            )
            normed_visual = None
            if cache_visual_kv:
                normed_visual = precomputed_input_layernorm[:, vs:ve]
        else:
            residual_nv = torch.cat([hidden_states[:, :vs], hidden_states[:, ve:]], dim=1)
            # RMSNorm and all projections are token-wise. Keep the legacy
            # text path compact; the opt-in cache path normalizes the visual
            # block separately and computes only its K/V projections.
            normed_nv = _skip_rms_norm(decoder_layer.input_layernorm, residual_nv)
            normed_visual = None
            if cache_visual_kv:
                normed_visual = _skip_rms_norm(
                    decoder_layer.input_layernorm, hidden_states[:, vs:ve]
                )


        nv_shape = (bs, nv_len, -1, attn.head_dim)
        if self.training:
            full_shape = (bs, seq_len, -1, attn.head_dim)
            key_states = attn.k_proj(normed).view(full_shape).transpose(1, 2)
            value_states = attn.v_proj(normed).view(full_shape).transpose(1, 2)
            normed_nv = torch.cat([normed[:, :vs], normed[:, ve:]], dim=1)
            query_states = attn.q_proj(normed_nv).view(nv_shape).transpose(1, 2)
            key_states = (key_states * cos.unsqueeze(1)) + (
                rotate_half(key_states) * sin.unsqueeze(1)
            )
            k_text = torch.cat([key_states[:, :, :vs], key_states[:, :, ve:]], dim=2)
            v_text = torch.cat([value_states[:, :, :vs], value_states[:, :, ve:]], dim=2)
        elif common_kv is not None:
            query_linear = attn.q_proj(normed_nv)
            query_states = query_linear.view(nv_shape).transpose(1, 2)
            key_states, value_states = common_kv
            k_text = torch.cat(
                [key_states[:, :, :vs], key_states[:, :, ve:]], dim=2
            )
            v_text = torch.cat(
                [value_states[:, :, :vs], value_states[:, :, ve:]], dim=2
            )
        else:
            # Q, K, V: one packed projection over text tokens only. Keeping
            # this eval-only leaves checkpoint and training parameters intact.
            query_linear, key_linear, value_linear = attn.skip_qkv_projection(normed_nv)
            query_states = query_linear.view(nv_shape).transpose(1, 2)
            k_text = key_linear.view(nv_shape).transpose(1, 2)
            v_text = value_linear.view(nv_shape).transpose(1, 2)

            if cache_visual_kv:
                visual_shape = (bs, n_visual, -1, attn.head_dim)
                k_visual = attn.k_proj(normed_visual).view(visual_shape).transpose(1, 2)
                v_visual = attn.v_proj(normed_visual).view(visual_shape).transpose(1, 2)

        if self.training:
            # Preserve the original training graph and operation order.
            k_text = (k_text * cos_nv.unsqueeze(1)) + (
                rotate_half(k_text) * sin_nv.unsqueeze(1)
            )
            query_states = (query_states * cos_nv.unsqueeze(1)) + (
                rotate_half(query_states) * sin_nv.unsqueeze(1)
            )
        elif common_kv is not None:
            query_states = (query_states * cos_nv.unsqueeze(1)) + (
                rotate_half(query_states) * sin_nv.unsqueeze(1)
            )
        else:
            query_states, k_text = _skip_rope_qk(
                query_states, k_text, cos_nv, sin_nv
            )

        # The actual skipped attention stays text-only in both modes. The
        # opt-in path additionally writes visual K/V to the cache so decode can
        # attend to the CDPruner-retained visual tokens without changing the
        # prefill hidden states produced by this layer.
        if past_key_value is not None and common_kv is None:
            if cache_visual_kv:
                cos_visual = cos[:, vs:ve]
                sin_visual = sin[:, vs:ve]
                k_visual = (k_visual * cos_visual.unsqueeze(1)) + (
                    rotate_half(k_visual) * sin_visual.unsqueeze(1)
                )
                key_states = torch.cat(
                    [k_text[:, :, :vs], k_visual, k_text[:, :, vs:]], dim=2
                )
                value_states = torch.cat(
                    [v_text[:, :, :vs], v_visual, v_text[:, :, vs:]], dim=2
                )
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                # Router visual skipping is prefill-only, so the local text K/V
                # below already represents the complete attention sequence for
                # this call. Cache.update is intentionally side-effect-only:
                # using its full return here would expose visual K/V to prefill.
                past_key_value.update(
                    key_states, value_states, attn.layer_idx, cache_kwargs
                )
            else:
                cache_position_text = text_cache_position
                if cache_position_text is None and cache_position is not None:
                    cache_position_text = torch.cat(
                        [cache_position[:vs], cache_position[ve:]], dim=0
                    )
                cache_kwargs = {
                    "sin": sin_nv,
                    "cos": cos_nv,
                    "cache_position": cache_position_text,
                }
                k_text, v_text = past_key_value.update(
                    k_text, v_text, attn.layer_idx, cache_kwargs
                )

        k_text = repeat_kv(k_text, attn.num_key_value_groups)
        v_text = repeat_kv(v_text, attn.num_key_value_groups)

        # Attention mask: text × text. In eval with no padding, removing
        # visual tokens preserves the causal order, so let SDPA use its causal
        # fast path instead of rebuilding an explicit dense mask in every
        # skipped layer. Keep the original explicit-mask path for training and
        # for inputs that actually carry a padding/4D mask.
        use_causal_sdpa = (
            not self.training
            and text_attention_mask is None
            and attention_mask is None
        )
        if use_causal_sdpa:
            attn_mask = None
        elif text_attention_mask is not None:
            attn_mask = text_attention_mask
        elif attention_mask is not None and attention_mask.dim() == 4:
            attn_mask = torch.cat(
                [attention_mask[:, :, :vs, :], attention_mask[:, :, ve:, :]],
                dim=2,
            )
            attn_mask = torch.cat(
                [attn_mask[:, :, :, :vs], attn_mask[:, :, :, ve:]],
                dim=3,
            )
        else:
            dtype = query_states.dtype
            q_pos = torch.cat(
                [
                    torch.arange(vs, device=device),
                    torch.arange(ve, seq_len, device=device),
                ]
            )
            k_pos = q_pos
            causal = q_pos[:, None] >= k_pos[None, :]
            attn_mask = torch.zeros(nv_len, nv_len, dtype=dtype, device=device)
            attn_mask.masked_fill_(~causal, torch.finfo(dtype).min)
            attn_mask = attn_mask[None, None, :, :]
            if attention_mask is not None and attention_mask.dim() == 2:
                pad_text = torch.cat(
                    [attention_mask[:, :vs], attention_mask[:, ve:]], dim=1
                )
                attn_mask = attn_mask.masked_fill(
                    pad_text[:, None, None, :] == 0, torch.finfo(dtype).min
                )

        query_states = query_states.contiguous()
        k_text = k_text.contiguous()
        v_text = v_text.contiguous()

        attn_output = F.scaled_dot_product_attention(
            query_states,
            k_text,
            v_text,
            attn_mask=attn_mask,
            dropout_p=attn.attention_dropout if self.training else 0.0,
            is_causal=use_causal_sdpa and query_states.shape[-2] > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bs, nv_len, -1)
        attn_output = attn.skip_o_projection(attn_output)


        # 文本 token 应用注意力，视觉 token 仅残差直通
        if self.training:
            residual_nv = torch.cat([residual[:, :vs], residual[:, ve:]], dim=1)
        nv_after_attn = residual_nv + attn_output
        if return_compact:
            return nv_after_attn
        if hidden_states_are_compact:
            raise RuntimeError("Compact attention state must be returned as compact")
        hidden_states = torch.cat(
            [nv_after_attn[:, :vs], residual[:, vs:ve], nv_after_attn[:, vs:]],
            dim=1,
        )
        return hidden_states

    def _efficient_skip_mlp(
        self,
        decoder_layer: LlamaDecoderLayer,
        hidden_states: torch.Tensor,
        visual_start: Optional[int] = None,
        visual_end: Optional[int] = None,
        return_compact: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        仅对 MLP 部分做「跳过视觉 token」的高效前向（文本走 MLP，视觉仅残差）。
        输入应为已经过 Self-Attention 后的 hidden_states。
        """
        bs, seq_len, _ = hidden_states.shape
        vs = visual_start if visual_start is not None else self.visual_start
        ve = visual_end if visual_end is not None else min(vs + self.visual_len_fallback, seq_len)
        ve = min(ve, seq_len)

        residual = hidden_states
        residual_nv = torch.cat(
            [hidden_states[:, :vs], hidden_states[:, ve:]], dim=1
        )
        nv_for_mlp = _skip_rms_norm(
            decoder_layer.post_attention_layernorm, residual_nv
        )
        nv_for_mlp = decoder_layer.mlp.skip_forward(nv_for_mlp)

        nv_after_mlp = residual_nv + nv_for_mlp
        if return_compact:
            return nv_after_mlp, residual[:, vs:ve]
        hidden_states = torch.cat(
            [nv_after_mlp[:, :vs], residual[:, vs:ve], nv_after_mlp[:, vs:]],
            dim=1,
        )
        return hidden_states

    def _compute_visual_span(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: torch.Tensor,
    ) -> Tuple[int, int]:
        """
        Infer the expanded visual interval from the actual image placeholder.

        LLaVA passes either the original ids (one IMAGE_TOKEN_INDEX expanded to
        many embeddings) or reconstructed expanded ids (one IMAGE_TOKEN_INDEX
        per retained visual token).  The latter is used by physical Top-K
        compaction; treating its entire prompt suffix as visual was the source
        of severe K=64 router/LCCache drift.
        """
        seq_length = int(inputs_embeds.shape[1])
        fallback_start = min(int(self.visual_start), seq_length)
        fallback_end = min(
            fallback_start + int(self.visual_len_fallback), seq_length
        )
        if input_ids is None or input_ids.ndim != 2:
            return 0, 0

        ids_length = int(input_ids.shape[1])
        pad_id = getattr(self.config, "pad_token_id", None)
        if ids_length != seq_length:
            recorded_spans = getattr(self, "_prefill_visual_spans", None)
            if recorded_spans is not None:
                flat_spans = [
                    tuple(int(value) for value in span)
                    for row_spans in recorded_spans
                    for span in row_spans
                ]
                if len(flat_spans) == input_ids.shape[0] and all(
                    span == flat_spans[0] for span in flat_spans[1:]
                ):
                    start, end = flat_spans[0]
                    if 0 <= start < end <= seq_length:
                        return start, end
        spans: List[Tuple[int, int]] = []
        for row in input_ids:
            image_positions = row.eq(self.visual_token_index).nonzero(
                as_tuple=False
            ).flatten()
            if image_positions.numel() == 0:
                return 0, 0

            if ids_length == seq_length:
                # Expanded ids contain one contiguous placeholder for every
                # retained visual token.
                if not bool(
                    (image_positions[1:] == image_positions[:-1] + 1).all()
                ):
                    raise ValueError(
                        "expanded multimodal input_ids must contain one contiguous "
                        "IMAGE_TOKEN_INDEX span"
                    )
                start = int(image_positions[0].item())
                end = int(image_positions[-1].item()) + 1
            else:
                # Raw ids contain exactly one placeholder.  Multimodal prepare
                # removes padding before constructing the expanded prompt.
                if image_positions.numel() != 1:
                    raise ValueError(
                        "raw multimodal prefill currently requires exactly one "
                        "IMAGE_TOKEN_INDEX per sequence"
                    )
                if pad_id is None:
                    valid_row = row
                else:
                    # Padding is contiguous at a sequence edge.  Slice the
                    # padded edges instead of deleting every pad-id occurrence;
                    # for LLaMA, id 0 is also the valid unknown-token id.
                    non_padding = row.ne(pad_id).nonzero(
                        as_tuple=False
                    ).flatten()
                    if non_padding.numel() == 0:
                        raise ValueError("multimodal input_ids row contains only padding")
                    valid_row = row[
                        int(non_padding[0].item()) : int(non_padding[-1].item()) + 1
                    ]
                valid_image_positions = valid_row.eq(
                    self.visual_token_index
                ).nonzero(as_tuple=False).flatten()
                if valid_image_positions.numel() != 1:
                    raise ValueError(
                        "could not locate a unique image placeholder after removing padding"
                    )
                start = int(valid_image_positions.item())
                visual_length = seq_length - int(valid_row.numel()) + 1
                if visual_length <= 0:
                    raise ValueError(
                        f"invalid expanded visual length {visual_length}: "
                        f"embeddings={seq_length}, raw_valid_ids={int(valid_row.numel())}"
                    )
                end = start + visual_length

            if not (0 <= start < end <= seq_length):
                raise ValueError(
                    f"invalid inferred visual span [{start}, {end}) for "
                    f"sequence length {seq_length}"
                )
            spans.append((start, end))

        if any(span != spans[0] for span in spans[1:]):
            raise ValueError(
                "batched router inference requires identical visual spans; "
                f"got {spans}"
            )
        return spans[0]

    def _text_router_input(
        self,
        hidden_states: torch.Tensor,
        seq_length: int,
        visual_start: int,
        visual_end: int,
        labels: Optional[torch.LongTensor] = None,
        hidden_states_are_compact: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        Build the shared router input using text tokens (after image tokens):
        1. Extract text tokens (after visual_end position)
        2. When labels provided (training): use only QUESTION tokens (labels=-100 部分)
           When labels is None (inference): use all text tokens
        3. Average of (question tokens except last) concat with last question token

        hidden_states: [bs, T, d]
        labels: [bs, seq_len]，-100 表示 prompt/问题，非 -100 表示答案。用于训练时只取问题 token。
        return: [bs, 2*d], or None when fewer than two text tokens are available
        """
        ve = min(visual_end, seq_length)
        bs = hidden_states.size(0)
        device = hidden_states.device
        
        # Text tokens start after visual tokens
        # In compact inference state the visual span [visual_start:visual_end]
        # has already been removed, so the post-image text starts at
        # visual_start instead of visual_end.
        text_start = visual_start if hidden_states_are_compact else ve
        text_tokens = hidden_states[:, text_start:, :]  # [bs, num_text, d]
        num_text = text_tokens.size(1)
        
        if num_text < 2:
            # Not enough text tokens (need at least 2: one for avg, one for last).
            # The caller keeps both paths in this case, matching the old behavior.
            return None
        
        # 训练时有 labels：只取问题 token（labels[i]=-100 的最后一个位置之前）
        # 推理时 labels=None：用全部 text token
        if labels is not None:
            # first_ans_pos[b] = 第一个 labels[b,i] != -100 的位置（答案开始）
            masks = (labels != -100)
            first_ans_pos = torch.where(
                masks.any(dim=1),
                masks.long().argmax(dim=1),
                torch.full((bs,), seq_length, device=device, dtype=torch.long),
            )
            # question 结束位置 = first_ans_pos - 1，问题文本 = [ve, first_ans_pos-1]
            question_end = (first_ans_pos - 1).clamp(min=ve - 1)
            question_text_len = (question_end - ve + 1).clamp(min=0, max=num_text)
            
            # 对每个 batch 计算：avg(问题 token 除最后一个) 和 最后一个问题 token
            # avg 范围：ve 到 first_ans_pos-2（当 question_text_len>=2）
            # question_text_len=1 时：只有一个问题 token，avg 和 last 都用它
            pos = torch.arange(seq_length, device=device).unsqueeze(0)  # [1, seq]
            avg_end = (first_ans_pos - 2).clamp(min=ve - 1)  # 参与 avg 的最后一个位置
            avg_mask = (pos >= ve) & (pos <= avg_end.unsqueeze(1))
            sum_avg = (hidden_states * avg_mask.unsqueeze(-1).to(hidden_states.dtype)).sum(dim=1)
            count_avg = avg_mask.sum(dim=1).clamp(min=1).unsqueeze(-1).to(hidden_states.dtype)
            text_avg = sum_avg / count_avg  # [bs, d], same dtype as hidden_states
            
            last_q_pos = (first_ans_pos - 1).clamp(min=ve, max=seq_length - 1)
            text_last = hidden_states[torch.arange(bs, device=device), last_q_pos, :]  # [bs, d]
            
            # question_text_len=1 时 count_avg 可能为 0（被 clamp 成 1），此时 sum_avg=0 会得到全 0。用 last 替代 avg
            need_single_token = (question_text_len == 1)
            if need_single_token.any():
                text_avg = torch.where(need_single_token.unsqueeze(-1), text_last, text_avg)
            
            # 若某样本没有问题 text（question_text_len[b]=0），回退到用全部 text
            use_full_text = (question_text_len == 0)
            if use_full_text.any():
                fallback_avg = text_tokens[:, :-1, :].mean(dim=1)
                fallback_last = text_tokens[:, -1, :]
                text_avg = torch.where(use_full_text.unsqueeze(-1), fallback_avg, text_avg)
                text_last = torch.where(use_full_text.unsqueeze(-1), fallback_last, text_last)
        else:
            # 推理：用全部 text token
            text_avg = text_tokens[:, :-1, :].mean(dim=1)  # [bs, d]
            text_last = text_tokens[:, -1, :]  # [bs, d]
        
        return torch.cat([text_avg, text_last], dim=-1).to(hidden_states.dtype)  # [bs, 2*d]

    @staticmethod
    def _slice_attention_mask_by_visual_span(
        attention_mask: Optional[torch.Tensor],
        visual_start: int,
        visual_end: int,
    ) -> Optional[torch.Tensor]:
        """Remove one validated contiguous visual span without an index tensor."""
        if attention_mask is None:
            return None
        visual_start = int(visual_start)
        visual_end = int(visual_end)
        if attention_mask.dim() == 4:
            query_length = int(attention_mask.shape[2])
            key_value_length = int(attention_mask.shape[3])
            if not (
                0
                <= visual_start
                < visual_end
                <= query_length
                and visual_end <= key_value_length
            ):
                raise ValueError(
                    "visual span must fit both query and key/value mask axes"
                )
            sliced = torch.cat(
                [
                    attention_mask[:, :, :visual_start, :],
                    attention_mask[:, :, visual_end:, :],
                ],
                dim=2,
            )
            return torch.cat(
                [
                    sliced[:, :, :, :visual_start],
                    sliced[:, :, :, visual_end:],
                ],
                dim=3,
            )
        if attention_mask.dim() == 2:
            sequence_length = int(attention_mask.shape[1])
            if not 0 <= visual_start < visual_end <= sequence_length:
                raise ValueError("visual span must fit the 2D attention mask")
            return torch.cat(
                [
                    attention_mask[:, :visual_start],
                    attention_mask[:, visual_end:],
                ],
                dim=1,
            )
        return attention_mask

    @staticmethod
    def _slice_position_embeddings_by_visual_span(
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        visual_start: int,
        visual_end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack text RoPE rows around one validated contiguous visual span."""
        cos, sin = position_embeddings
        if cos.ndim != 3 or sin.ndim != 3 or cos.shape != sin.shape:
            raise ValueError("cos and sin must have the same rank-three shape")
        visual_start = int(visual_start)
        visual_end = int(visual_end)
        sequence_length = int(cos.shape[1])
        if not 0 <= visual_start < visual_end <= sequence_length:
            raise ValueError("visual span must fit the position embeddings")
        return (
            torch.cat([cos[:, :visual_start], cos[:, visual_end:]], dim=1),
            torch.cat([sin[:, :visual_start], sin[:, visual_end:]], dim=1),
        )

    def _efficient_skip_attn_mlp_compact(
        self,
        decoder_layer: LlamaDecoderLayer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_start: Optional[int] = None,
        visual_end: Optional[int] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        text_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        text_cache_position: Optional[torch.LongTensor] = None,
        visual_hidden_states: Optional[torch.Tensor] = None,
        hidden_states_are_compact: bool = False,
        precomputed_input_layernorm: Optional[torch.Tensor] = None,
        precomputed_visual_input_layernorm: Optional[torch.Tensor] = None,
        precomputed_rotated_key_value_states: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Inference-only combined path when both attention and MLP skip visual tokens.

        The visual block is unchanged by either sublayer. Return only the
        compact non-visual sequence so consecutive double-skip layers do not
        repeatedly rebuild and re-slice the full sequence.
        """
        text_hidden_states = self._efficient_skip_self_attn(
            decoder_layer,
            hidden_states,
            attention_mask,
            position_embeddings,
            past_key_value=past_key_value,
            cache_position=cache_position,
            visual_start=visual_start,
            visual_end=visual_end,
            text_attention_mask=text_attention_mask,
            text_position_embeddings=text_position_embeddings,
            text_cache_position=text_cache_position,
            visual_hidden_states=visual_hidden_states,
            hidden_states_are_compact=hidden_states_are_compact,
            return_compact=True,
            precomputed_input_layernorm=precomputed_input_layernorm,
            precomputed_visual_input_layernorm=(
                precomputed_visual_input_layernorm
            ),
            precomputed_rotated_key_value_states=(
                precomputed_rotated_key_value_states
            ),
            **kwargs,
        )

        residual = text_hidden_states
        text_hidden_states = _skip_rms_norm(
            decoder_layer.post_attention_layernorm, text_hidden_states
        )
        text_hidden_states = decoder_layer.mlp.skip_forward(text_hidden_states)
        return residual + text_hidden_states

    @staticmethod
    def _restore_compact_hidden_states(
        text_hidden_states: torch.Tensor,
        visual_hidden_states: torch.Tensor,
        visual_start: int,
    ) -> torch.Tensor:
        """Restore [text-prefix, visual, text-suffix] from compact inference state."""
        return torch.cat(
            [
                text_hidden_states[:, :visual_start],
                visual_hidden_states,
                text_hidden_states[:, visual_start:],
            ],
            dim=1,
        )

    def _exact_router_gates_to_host(
        self, gate_values: torch.Tensor
    ) -> Tuple[float, float]:
        """Read the current layer's exact gates on the host.

        This helper remains the synchronous portable contract.  The ordinary
        immediate decoder path uses ``_enqueue_immediate_router_gate_copy``
        directly so it can schedule branch-independent work before waiting.
        """
        if gate_values.numel() != 2:
            raise RuntimeError(
                "immediate Router inference requires exactly two gate values"
            )
        pinned_requested = os.environ.get(
            "TRANSFORMERS_LLAMA_PINNED_ROUTER_DECISION", "1"
        ).strip().lower() not in {"0", "false", "no", "off", ""}
        if not pinned_requested or not gate_values.is_cuda:
            attn_gate, mlp_gate = gate_values.reshape(-1).tolist()
            return float(attn_gate), float(mlp_gate)

        device = gate_values.device
        device_key = (device.type, device.index)
        host_gate = self._router_immediate_gate_host
        if host_gate is None or host_gate.dtype != gate_values.dtype:
            host_gate = torch.empty(
                2,
                dtype=gate_values.dtype,
                device="cpu",
                pin_memory=True,
            )
            object.__setattr__(self, "_router_immediate_gate_host", host_gate)
        copy_done = self._router_immediate_gate_copy_done
        if copy_done is None or self._router_immediate_gate_device != device_key:
            copy_done = torch.cuda.Event()
            object.__setattr__(
                self, "_router_immediate_gate_copy_done", copy_done
            )
            object.__setattr__(self, "_router_immediate_gate_device", device_key)

        compute_stream = torch.cuda.current_stream(device)
        host_gate.copy_(gate_values.reshape(-1), non_blocking=True)
        copy_done.record(compute_stream)
        copy_done.synchronize()
        # The pinned CPU buffer is ready. Read both values in one conversion,
        # avoiding two scalar tensor views/item calls on every decoder layer.
        return tuple(host_gate.tolist())

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        router_temperature = flash_attn_kwargs.pop("router_temperature", None)
        labels = flash_attn_kwargs.pop("labels", None)
        final_norm_only_last_token = flash_attn_kwargs.pop("_final_norm_only_last_token", False)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #if (input_ids is None) ^ (inputs_embeds is not None):
            #raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        visual_start, visual_end = self._compute_visual_span(input_ids, inputs_embeds)
        #print(visual_end)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        #print(f"past_key_values: {past_key_values}")

        # Determine if we are in prefill stage (only skip during prefill)
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        is_prefill = (past_seen_tokens == 0)

        # Pre-compute keep indices for visual token skipping (only once, shared across layers)
        seq_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        can_drop_visual = (
            0 <= visual_start < min(visual_end, seq_length)
            and is_prefill and self.config.router_enabled
        )
        compact_text_states = None
        compact_visual_states = None
        compact_double_skip = (
            not self.training
            and os.environ.get("TRANSFORMERS_LLAMA_COMPACT_DOUBLE_SKIP", "1").lower()
            not in {"0", "false", "no", "off"}
        )
        text_attention_mask = None
        text_position_embeddings = None
        text_cache_position = None
        if can_drop_visual:
            text_position_embeddings = self._slice_position_embeddings_by_visual_span(
                position_embeddings, visual_start, visual_end
            )
            text_attention_mask = self._slice_attention_mask_by_visual_span(
                causal_mask, visual_start, visual_end
            )
            text_cache_position = torch.cat(
                [cache_position[:visual_start], cache_position[visual_end:]]
            )
        # --- custom decoder (router / skip paths) ---
        gating_scores_all_layers_attn: List[torch.Tensor] = []
        gating_scores_all_layers_mlp: List[torch.Tensor] = []
        layer_skip_count = 0  
        skipped_layer_indices: List[int] = []

        attn_skip_count = 0
        attn_skipped_layer_indices: List[int] = []
        
        mlp_skip_count = 0
        mlp_skipped_layer_indices: List[int] = []

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                full_states = (hidden_states if compact_text_states is None else
                               self._restore_compact_hidden_states(
                                   compact_text_states, compact_visual_states, visual_start))
                all_hidden_states += (full_states,)

            # Pool once and execute the two independently trained routers together.
            active_states = compact_text_states if compact_text_states is not None else hidden_states
            if can_drop_visual:
                if self.training:
                    g_attn_soft = self._text_gate_scalar(
                        active_states, self.attn_routers[layer_idx], seq_length,
                        visual_start, visual_end, router_temperature, labels,
                    )
                    g_mlp_soft = self._text_gate_scalar(
                        active_states, self.mlp_routers[layer_idx], seq_length,
                        visual_start, visual_end, router_temperature, labels,
                    )
                    gate_values = torch.stack([g_attn_soft.mean(), g_mlp_soft.mean()])
                else:
                    pair = self.inference_router_pairs[layer_idx]
                    text_start = visual_start if compact_text_states is not None else visual_end
                    gates = pair.forward_from_hidden_states(active_states, text_start, router_temperature)
                    if gates is None:
                        router_input = self._text_router_input(
                            active_states, seq_length, visual_start, visual_end,
                            labels=labels, hidden_states_are_compact=compact_text_states is not None,
                        )
                        gates = (active_states.new_ones((batch_size, 2)) if router_input is None
                                 else pair(router_input, temperature=router_temperature))
                    gates = gates.reshape(batch_size, 2)
                    # Triton reuses its output buffer at the next prefill.
                    g_attn_soft, g_mlp_soft = gates[:, 0].clone(), gates[:, 1].clone()
                    gate_values = gates[0] if batch_size == 1 else gates.mean(dim=0)
                gating_scores_all_layers_attn.append(g_attn_soft)
                gating_scores_all_layers_mlp.append(g_mlp_soft)
                attn_gate, mlp_gate = self._exact_router_gates_to_host(gate_values)
                do_skip_attn = attn_gate <= self.tau
                do_skip_mlp = mlp_gate <= self.tau

                if do_skip_attn or do_skip_mlp:
                    layer_skip_count += 1
                    skipped_layer_indices.append(layer_idx)

                if do_skip_attn:
                    attn_skip_count += 1
                    attn_skipped_layer_indices.append(layer_idx)
                
                if do_skip_mlp:
                    mlp_skip_count += 1
                    mlp_skipped_layer_indices.append(layer_idx)
                    
                if do_skip_attn and do_skip_mlp and compact_double_skip:
                    if compact_text_states is None:
                        compact_visual_states = hidden_states[:, visual_start:visual_end]
                    compact_text_states = self._efficient_skip_attn_mlp_compact(
                        decoder_layer, active_states, causal_mask, position_embeddings,
                        past_key_value=past_key_values, cache_position=cache_position,
                        visual_start=visual_start, visual_end=visual_end,
                        text_attention_mask=text_attention_mask,
                        text_position_embeddings=text_position_embeddings,
                        text_cache_position=text_cache_position,
                        visual_hidden_states=compact_visual_states,
                        hidden_states_are_compact=compact_text_states is not None,
                        **flash_attn_kwargs,
                    )
                    if output_attentions:
                        all_self_attns += (None,)
                    continue
                if compact_text_states is not None:
                    hidden_states = self._restore_compact_hidden_states(
                        compact_text_states, compact_visual_states, visual_start
                    )
                    compact_text_states = compact_visual_states = None

                # ---------- Self-Attention ----------
                if do_skip_attn:
                    h_after_attn = self._efficient_skip_self_attn(
                        decoder_layer,
                        hidden_states,
                        causal_mask,
                        position_embeddings,
                        past_key_value=past_key_values,
                        cache_position=cache_position,
                        visual_start=visual_start,
                        visual_end=visual_end,
                        **flash_attn_kwargs,
                    )
                    self_attn_weights = None
                else:
                    residual_sa = hidden_states
                    normed_sa = decoder_layer.input_layernorm(hidden_states)
                    attn_outputs_full = decoder_layer.self_attn(
                        hidden_states=normed_sa,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )
                    h_after_attn = residual_sa + attn_outputs_full[0]
                    self_attn_weights = attn_outputs_full[1] if output_attentions else None

                # ---------- MLP ----------
                if do_skip_mlp:
                    hidden_states = self._efficient_skip_mlp(
                        decoder_layer,
                        h_after_attn,
                        visual_start=visual_start,
                        visual_end=visual_end,
                    )
                else:
                    residual_mlp = h_after_attn
                    h_norm_mlp = decoder_layer.post_attention_layernorm(h_after_attn)
                    h_mlp_full = decoder_layer.mlp(h_norm_mlp)
                    hidden_states = residual_mlp + h_mlp_full

                layer_outputs = (hidden_states, self_attn_weights)
            else:
                # Default path 
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__, hidden_states, causal_mask, position_ids, past_key_values,
                        output_attentions, use_cache, cache_position, position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                        past_key_value=past_key_values, output_attentions=output_attentions,
                        use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if compact_text_states is not None:
            hidden_states = self._restore_compact_hidden_states(
                compact_text_states, compact_visual_states, visual_start
            )
        if final_norm_only_last_token:
            hidden_states = hidden_states[:, -1:, :]
        hidden_states = self.norm(hidden_states)

        # Overwrite gating caches on prefill only; keep prefill values across decode
        if is_prefill and len(gating_scores_all_layers_attn) > 0:
            if self.training:
                self.last_gating_scores_attn = list(gating_scores_all_layers_attn)
                self.last_gating_scores_mlp = list(gating_scores_all_layers_mlp)
            else:
                self.last_gating_scores_attn = [g.detach() for g in gating_scores_all_layers_attn]
                self.last_gating_scores_mlp = [g.detach() for g in gating_scores_all_layers_mlp]
                    
        # (hook: store gating scores for optional loss)
        # --- end custom decoder ---
        # Store layer skip count for evaluation (only when visual skipping was applicable)
        # Use object.__setattr__ to avoid DDP/FSDP sync; no init in __init__ to avoid training hang
        if not self.training and can_drop_visual:
            object.__setattr__(self, "_last_prefill_layer_skip_count", layer_skip_count)
            object.__setattr__(self, "_last_prefill_skipped_layer_indices", skipped_layer_indices)

            object.__setattr__(self, "_last_prefill_attn_skip_count", attn_skip_count)
            object.__setattr__(self, "_last_prefill_attn_skipped_layer_indices", attn_skipped_layer_indices)
        
            object.__setattr__(self, "_last_prefill_mlp_skip_count", mlp_skip_count)
            object.__setattr__(self, "_last_prefill_mlp_skipped_layer_indices", mlp_skipped_layer_indices)
        elif not self.training and is_prefill:
            object.__setattr__(self, "_last_prefill_layer_skip_count", 0)
            object.__setattr__(self, "_last_prefill_skipped_layer_indices", [])

            object.__setattr__(self, "_last_prefill_attn_skip_count", 0)
            object.__setattr__(self, "_last_prefill_attn_skipped_layer_indices", [])
        
            object.__setattr__(self, "_last_prefill_mlp_skip_count", 0)
            object.__setattr__(self, "_last_prefill_mlp_skipped_layer_indices", [])
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            if isinstance(attention_mask, BlockMask):
                return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

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

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #torch.set_printoptions(profile="full")
        #print(input_ids)
        def _run_core_forward():
            final_norm_only_last_token = (
                not self.training and not torch.is_grad_enabled()
                and labels is None
                and not (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
                and type(logits_to_keep) is int and logits_to_keep == 1
            )
            outputs_ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                router_temperature=None,  # Router default gate_temperature
                labels=labels,
                _final_norm_only_last_token=final_norm_only_last_token,
                **kwargs,
            )
            hidden_states_ = outputs_[0]
            slice_indices_ = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits_ = self.lm_head(hidden_states_[:, slice_indices_, :])
            return outputs_, logits_

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs, logits = _run_core_forward()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #print(labels)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
