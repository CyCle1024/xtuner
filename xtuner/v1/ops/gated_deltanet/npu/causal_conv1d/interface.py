"""Ascend causal convolution autograd interface."""

from typing import Dict, Optional

import torch

from .causal_conv1d_triton_ascend import causal_conv1d_bwd_impl, causal_conv1d_fwd_impl


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


def causal_conv1d_triton(
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
    """Apply causal 1D convolution with an integrated backward pass.

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
        The output of shape ``[B, H, T, D / H]`` and the optional final state.
    """
    return CausalConv1dFunction.apply(
        x, weight, H, bias, residual, initial_state, activation, cu_seqlens, chunk_indices, output_final_state
    )
