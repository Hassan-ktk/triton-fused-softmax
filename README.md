# Fused Softmax in Triton, with a Verifier and Benchmark Harness

A fused row-wise softmax kernel in Triton, compared against a naive multi-kernel PyTorch
baseline, plus an automated numerical verifier and a reliable benchmark. A "cheating" demo
shows the verifier rejecting shortcuts that would fake a speedup.

## Files
| File | Purpose |
|---|---|
| `kernel.py` | Naive baseline + Triton fused softmax |
| `verify.py` | Correctness checks: dtypes, odd shapes, large values, edge cases |
| `benchmark.py` | Timing with warmup, sync, median, and achieved bandwidth |
| `cheat_demo.py` | Four cheating implementations the verifier must catch |

## Setup
```
pip install torch triton
python verify.py
python benchmark.py
python cheat_demo.py
```
Needs an NVIDIA GPU (Colab T4 works).

## Why it's faster
Softmax is memory-bound. The naive version runs ~5 separate kernels (max, subtract, exp,
sum, divide), each reading and writing the full tensor from HBM. The fused kernel reads each
row once, computes in registers, and writes once, cutting memory traffic roughly 3-5x.

## Results (fill in from your run)
GPU: ____  dtype: float16  rows: 4096

| cols | naive ms | torch ms | triton ms | triton GB/s | vs naive |
|---|---|---|---|---|---|
| ... | | | | | |

## Verifier design
- Float32 reference, tolerances per dtype (fp32 1e-5, fp16 1e-3, bf16 1e-2)
- Non-power-of-2 widths, empty input, non-contiguous input
- Large-magnitude inputs (catches missing max-subtraction)
- Fresh random inputs every run (catches caching/hardcoding)
- Rejects NaN/Inf, wrong shape/dtype, rows not summing to 1

## Limitations
- One block per row: supports up to 65536 columns. Larger needs a multi-block (online softmax) loop.
- Benchmark numbers vary by GPU; compare ratios, not absolute values.

## Possible extensions
- Online softmax for very long rows
- Fused attention (FlashAttention-style)
- Profile with Nsight Compute and add a roofline plot
