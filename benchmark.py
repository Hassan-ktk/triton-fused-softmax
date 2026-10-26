"""Benchmark naive PyTorch vs torch.softmax vs Triton. Run: python benchmark.py

triton.testing.do_bench handles warmup, CUDA synchronization, L2 cache
flushing between runs, and returns the median over many repetitions.
Bandwidth assumes 1 read + 1 write of the tensor (the minimum possible traffic).
"""
import torch
import triton
from kernel import naive_softmax, triton_softmax

N_ROWS = 4096
COLS = [256, 512, 1024, 2048, 4096, 8192, 16384]
DTYPE = torch.float16


def bench(fn, x):
    # quantiles=[0.5, ...] -> returns median first
    return triton.testing.do_bench(lambda: fn(x), warmup=25, rep=200, quantiles=[0.5, 0.2, 0.8])


def gbps(x, ms):
    bytes_moved = 2 * x.numel() * x.element_size()
    return bytes_moved / (ms * 1e-3) / 1e9


if __name__ == "__main__":
    assert torch.cuda.is_available(), "A CUDA GPU is required."
    print(f"GPU: {torch.cuda.get_device_name(0)}  dtype={DTYPE}  rows={N_ROWS}\n")
    header = f"{'cols':>6} | {'naive ms':>9} {'torch ms':>9} {'triton ms':>10} | {'triton GB/s':>11} | {'vs naive':>8} {'vs torch':>8}"
    print(header)
    print("-" * len(header))
    for c in COLS:
        x = torch.randn(N_ROWS, c, device="cuda", dtype=DTYPE)
        # sanity check before timing: never benchmark a wrong kernel
        torch.testing.assert_close(triton_softmax(x), torch.softmax(x, -1), atol=1e-3, rtol=1e-3)

        t_naive = bench(naive_softmax, x)[0]
        t_torch = bench(lambda t: torch.softmax(t, dim=-1), x)[0]
        t_tri = bench(triton_softmax, x)[0]
        print(f"{c:>6} | {t_naive:>9.4f} {t_torch:>9.4f} {t_tri:>10.4f} | "
              f"{gbps(x, t_tri):>11.1f} | {t_naive/t_tri:>7.2f}x {t_torch/t_tri:>7.2f}x")
