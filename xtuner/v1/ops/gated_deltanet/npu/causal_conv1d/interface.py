"""Ascend causal convolution autograd interface."""

from functools import lru_cache
from typing import Dict, Optional

import torch

from .causal_conv1d_triton_ascend import causal_conv1d_bwd_impl, causal_conv1d_fwd_impl, get_num_cores


class CausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        H: int,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        activation: str = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        chunk_indices: Dict[str, Optional[torch.LongTensor]] = None,
        output_final_state: bool = False,
    ):
        weight = weight.transpose(-1, -2).contiguous()
        ctx.save_for_backward(x, weight, bias, residual, initial_state)
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        ctx.H = H

        y, final_state = causal_conv1d_fwd_impl(
            x=x,
            weight=weight,
            H=H,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            chunk_indices_origin=chunk_indices,
            output_final_state=output_final_state,
        )
        ctx.final_state = final_state

        return y, final_state

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: Optional[torch.Tensor] = None):
        x, weight, bias, residual, initial_state = ctx.saved_tensors
        # The public time-major transpose is outside this autograd function.
        # Its backward normally restores head-first strides; materialize only
        # for callers whose downstream graph returns an incompatible view.
        if not dy.is_contiguous():
            dy = dy.contiguous()
        activation = ctx.activation
        cu_seqlens = ctx.cu_seqlens
        chunk_indices = ctx.chunk_indices
        H = ctx.H

        dx, dw, db, dr, dh0 = causal_conv1d_bwd_impl(
            x=x,
            dy=dy,
            H=H,
            dht=dht,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            chunk_indices_origin=chunk_indices,
        )

        return dx, dw.transpose(0, 1).contiguous(), None, db, dr, dh0, None, None, None, None


def causal_conv1d_triton_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    H: int,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    activation: str = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Dict[str, Optional[torch.LongTensor]] = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the NPU-native causal convolution.

    Args:
        x: Input tensor of shape ``[B, T, D]``.
        weight: Weight tensor of shape ``[D, W]``.
        H: Number of heads in the output view.
        bias: Optional bias tensor of shape ``[D]``.
        residual: Optional residual tensor of shape ``[B, T, D]``.
        initial_state: Optional initial state for sequence processing.
        activation: Optional activation function name.
        cu_seqlens: Optional cumulative lengths for packed sequences.
        chunk_indices: Optional precomputed chunk indices keyed by block size.
        output_final_state: Whether to return the final convolution state.

    Returns:
        A contiguous output of shape ``[B, H, T, D / H]`` and the optional
        final state.
    """
    return CausalConv1dFunction.apply(
        x, weight, H, bias, residual, initial_state, activation, cu_seqlens, chunk_indices, output_final_state
    )


@lru_cache(maxsize=8)
def _prepare_causal_conv_metadata(
    cu_seqlens: tuple[int, ...],
    device: str,
    total_tokens: int,
    forward_block_size: int,
    backward_block_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not cu_seqlens or cu_seqlens[0] != 0 or cu_seqlens[-1] != total_tokens:
        raise ValueError("cu_seqlens must start at zero and end at B * T")

    chunk_indices = {}
    for block_size in {forward_block_size, backward_block_size}:
        pairs = []
        for sequence_id, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
            pairs.extend((sequence_id, chunk_id) for chunk_id in range((end - start + block_size - 1) // block_size))
        chunk_indices[str(block_size)] = torch.tensor(pairs, device=device, dtype=torch.int64).reshape(-1, 2)

    cu_seqlens_tensor = torch.tensor(cu_seqlens, device=device, dtype=torch.int64)
    return cu_seqlens_tensor, chunk_indices


@torch.compiler.disable
def causal_conv1d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    cu_seqlens: torch.Tensor,
    cu_seqlens_list: Optional[list[int]] = None,
) -> torch.Tensor:
    """Apply NPU causal-conv with a ``[B, T, H, K]`` public layout."""
    if x.ndim != 4:
        raise ValueError(f"causal-conv input must have shape [B, T, H, K], got {tuple(x.shape)}")
    batch_size, seq_len, num_heads, head_dim = x.shape
    if cu_seqlens_list is None:
        raise ValueError("NPU causal-conv requires cu_seq_lens_q_list")

    num_cores = int(get_num_cores())
    tiles = 1 << (((max(16, batch_size * seq_len) + num_cores - 1) // num_cores) - 1).bit_length()
    cu_seqlens, chunk_indices = _prepare_causal_conv_metadata(
        tuple(cu_seqlens_list),
        str(x.device),
        batch_size * seq_len,
        min(32, tiles),
        min(4, tiles),
    )

    native, _ = causal_conv1d_triton_native(
        x=x.reshape(batch_size, seq_len, num_heads * head_dim),
        weight=weight,
        H=num_heads,
        bias=bias,
        activation=activation,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_final_state=False,
    )
    return native.transpose(1, 2)
