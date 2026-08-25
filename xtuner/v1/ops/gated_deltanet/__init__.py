"""GatedDeltaNet operator dispatchers.

`XTUNER_HF_IMPL` controls which implementations XTuner's `GatedDeltaNet` module uses,
mirroring how `xtuner/v1/ops/attn_imp.py::get_attn_impl_fn` and the rms_norm selector
switch between fast / fused paths and HF-exact paths. Under `XTUNER_HF_IMPL=true`:

* `chunk_gated_delta_rule` is the canonical `fla.ops.gated_delta_rule.chunk_gated_delta_rule`
  (same callable HF's `Qwen3_5GatedDeltaNet` uses), bypassing XTuner's
  `torch.library.custom_op` wrap.
* causal-conv accepts XTuner's ``[B, T, H, K]`` public layout, flattens it to the channel-last
  layout, then calls the high-level `causal_conv1d.causal_conv1d_fn` with ``seq_idx=None``
  (HF's non-packed convention).

These switches are only meant for the bitwise-parity tests. Production / training stays on the
XTuner path (compile-friendly custom_op wraps + seq_idx-aware kernel dispatch).
"""

import os

import torch

from ...data_proto.sequence_context import GatedDeltaNetMetadata
from ...utils import get_device
from .gen_seq_idx import gen_seq_idx


_TRUTHY = {"true", "1", "yes", "on"}


def _hf_impl_enabled() -> bool:
    return os.getenv("XTUNER_HF_IMPL", "").strip().lower() in _TRUTHY


def _hf_causal_conv1d(
    x,
    weight,
    bias,
    activation,
    cu_seqlens,
    cu_seqlens_list=None,
    seq_idx=None,
    cu_seqlens_int64=None,
    chunk_indices=None,
):
    from causal_conv1d import causal_conv1d_fn as _hf_causal_conv1d_fn

    batch_size, seq_len, num_heads, head_dim = x.shape
    x_cf = x.reshape(batch_size, seq_len, num_heads * head_dim).transpose(1, 2)
    out = _hf_causal_conv1d_fn(x=x_cf, weight=weight, bias=bias, activation=activation, seq_idx=None)
    return out.transpose(1, 2).reshape(batch_size, seq_len, num_heads, head_dim)


def _hf_chunk_gated_delta_rule(
    *args,
    cu_seqlens_int64=None,
    chunk_indices=None,
    chunk_indices_list=None,
    **kwargs,
):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule(*args, **kwargs)


def get_chunk_gated_delta_rule_fn():
    if _hf_impl_enabled():
        return _hf_chunk_gated_delta_rule
    if get_device() == "npu":
        from .npu.flash_gated_delta_rule import flash_gated_delta_rule as _npu_chunk_gated_delta_rule

        return _npu_chunk_gated_delta_rule
    from .chunk_gated_delta_rule import chunk_gated_delta_rule as _xtuner_chunk_gated_delta_rule

    return _xtuner_chunk_gated_delta_rule


def get_causal_conv1d_fn():
    if _hf_impl_enabled():
        return _hf_causal_conv1d
    if get_device() == "npu":
        from .npu.causal_conv1d import causal_conv1d_triton as _npu_causal_conv1d_fn

        return _npu_causal_conv1d_fn
    from .causal_conv1d import causal_conv1d as _xtuner_causal_conv1d_fn

    return _xtuner_causal_conv1d_fn


def prepare_gated_deltanet_metadata(
    *,
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: list[int],
    num_heads: int,
) -> GatedDeltaNetMetadata | None:
    """Prepare the current device's GatedDeltaNet metadata before layer
    execution."""
    if _hf_impl_enabled():
        return None

    total_tokens = cu_seqlens_list[-1]
    if get_device() == "npu":
        from .npu.metadata import prepare_npu_gated_deltanet_metadata

        return prepare_npu_gated_deltanet_metadata(
            cu_seqlens=cu_seqlens_list,
            device=cu_seqlens.device,
            total_tokens=total_tokens,
            num_heads=num_heads,
        )
    return GatedDeltaNetMetadata(seq_idx=gen_seq_idx(total_tokens, cu_seqlens))
