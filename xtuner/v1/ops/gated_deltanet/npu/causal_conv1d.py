from typing import Optional, Dict

import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import torch.nn as nn

# Load the repo's modified convolution.py by path so it overrides the pip-installed
# mojo_opset original. The original (from https://github.com/XPU-Forces/mojo_opset,
# pre-installed in the image) lacks the `H` kwarg this caller passes; the repo's copy
# adds it (torch-compile patch, do_not_specialize=['NUM_CHKS'] + D<BD guard). Only this
# one file is tracked in the repo; the rest of mojo_opset (utils, backends, ...) is
# provided by the installed package and resolved via the override's own
# `from mojo_opset...import` statements when it is exec'd below.
import importlib.util
from pathlib import Path

_root = Path(__file__).resolve().parent
while _root.name and not (
    _root / "mojo_opset" / "backends" / "ttx" / "kernels" / "npu" / "a2" / "convolution.py"
).exists():
    _root = _root.parent
assert _root.name, "repo root containing mojo_opset/.../a2/convolution.py not found"
_override_path = (
    _root / "mojo_opset" / "backends" / "ttx" / "kernels" / "npu" / "a2" / "convolution.py"
)
_spec = importlib.util.spec_from_file_location("xtuner._mojo_opset_conv_override", _override_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
causal_conv1d_fwd_impl = _mod.causal_conv1d_fwd_impl
causal_conv1d_bwd_impl = _mod.causal_conv1d_bwd_impl

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
        # Save necessary tensors for backward pass
        # x = x.transpose(-1, -2).contiguous()
        weight = weight.transpose(-1, -2).contiguous()
        ctx.save_for_backward(x, weight, bias, residual, initial_state)
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        ctx.H = H
        
        # Call the forward implementation
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
        # y = y.transpose(-1, -2).contiguous()
        # weight = weight.transpose(-1, -2).contiguous()
        # Save final_state if needed for backward
        ctx.final_state = final_state

        return y, final_state

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: Optional[torch.Tensor] = None):
        # Retrieve saved tensors from forward pass
        x, weight, bias, residual, initial_state = ctx.saved_tensors
        activation = ctx.activation
        cu_seqlens = ctx.cu_seqlens
        chunk_indices = ctx.chunk_indices
        H = ctx.H

        # Call the backward implementation with dht (could be None)
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

        # Return gradients in the order of forward inputs
        # Note: We don't return gradients for non-tensor inputs (activation, cu_seqlens, chunk_indices, output_final_state)
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
    """
    Causal 1D convolution with integrated forward and backward pass.

    Args:
        x: Input tensor of shape [B, T, D]
        weight: Weight tensor of shape [W, D]
        bias: Optional bias tensor of shape [D]
        residual: Optional residual tensor of shape [B, T, D]
        initial_state: Optional initial state tensor for sequence processing
        activation: Optional activation function name
        cu_seqlens: Optional cumulative sequence lengths for variable-length sequences
        output_final_state: Whether to output the final state

    Returns:
        y: Output tensor of shape [B, T, D]
        final_state: Optional final state tensor if output_final_state is True
    """
    return CausalConv1dFunction.apply(
        x, weight, H, bias, residual, initial_state, activation, cu_seqlens, chunk_indices, output_final_state
    )