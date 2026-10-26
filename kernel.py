"""Fused softmax in Triton + a naive PyTorch baseline.

Why fusion helps: the naive version launches several kernels (max, subtract,
exp, sum, divide), and each one reads/writes the full tensor to GPU memory (HBM).
Softmax is memory-bound, so those extra round trips dominate runtime.
The fused kernel loads each row once, does all the math in registers, and
writes the result once.
"""
import torch
import triton
import triton.language as tl


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Baseline: row-wise softmax written as separate PyTorch ops (many kernel launches)."""
    x_max = x.max(dim=1, keepdim=True).values
    z = x - x_max                      # subtract max for numerical stability
    num = torch.exp(z)
    den = num.sum(dim=1, keepdim=True)
    return num / den


@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)                       # one program instance per row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols                         # handles non-power-of-2 widths

    row_start = in_ptr + row * in_row_stride
    # Out-of-range lanes get -inf so they contribute exp(-inf) = 0.
    x = tl.load(row_start + offs, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)                         # accumulate in fp32 for accuracy

    x = x - tl.max(x, axis=0)                    # numerical stability
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    out_start = out_ptr + row * out_row_stride
    tl.store(out_start + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


MAX_COLS = 65536  # this simple kernel keeps a whole row in one block


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the last dim of a 2D CUDA tensor."""
    assert x.is_cuda, "input must be on GPU"
    assert x.ndim == 2, "expected a 2D tensor (rows, cols)"
    n_rows, n_cols = x.shape
    if x.numel() == 0:
        return torch.empty_like(x)
    assert n_cols <= MAX_COLS, f"n_cols > {MAX_COLS} not supported by this simple kernel"

    x = x.contiguous()
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    num_warps = 4 if block <= 2048 else 8 if block <= 8192 else 16

    _softmax_kernel[(n_rows,)](
        out, x,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )
    return out
