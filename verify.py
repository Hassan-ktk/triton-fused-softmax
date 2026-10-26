"""Correctness verifier. Run: python verify.py

Checks a candidate softmax against a float32 PyTorch reference across dtypes,
shapes (including non-power-of-2), large magnitudes (stability), and edge cases.
Inputs are freshly randomized every run, so cached/hardcoded outputs fail.
"""
import sys
import torch
from kernel import triton_softmax

# (atol, rtol) per dtype. Lower precision -> looser tolerance.
TOL = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
}
SHAPES = [(1, 1), (1, 7), (4, 128), (17, 1000), (64, 1023), (128, 1024),
          (257, 4096), (32, 5000), (1823, 781), (8, 32768)]
SCALES = [1.0, 10.0, 100.0]   # large values break softmax without max-subtraction


def reference(x):
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def check(fn, x, label, failures):
    try:
        y = fn(x)
    except Exception as e:  # a crash is a failure, not a pass
        failures.append(f"{label}: raised {type(e).__name__}: {e}")
        return
    if y.shape != x.shape or y.dtype != x.dtype:
        failures.append(f"{label}: wrong shape/dtype {tuple(y.shape)} {y.dtype}")
        return
    if not torch.isfinite(y).all():
        failures.append(f"{label}: contains NaN/Inf")
        return
    atol, rtol = TOL[x.dtype]
    if not torch.allclose(y.float(), reference(x).float(), atol=atol, rtol=rtol):
        err = (y.float() - reference(x).float()).abs().max().item()
        failures.append(f"{label}: mismatch, max abs err {err:.3e}")
        return
    row_sums = y.float().sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2):
        failures.append(f"{label}: rows do not sum to 1")


def run_all(fn, seed=None):
    """Return a list of failure strings (empty list == pass)."""
    if seed is not None:
        torch.manual_seed(seed)
    failures, n = [], 0
    for dtype in TOL:
        for shape in SHAPES:
            for scale in SCALES:
                x = (torch.randn(shape, device="cuda") * scale).to(dtype)
                check(fn, x, f"{dtype} {shape} scale={scale}", failures)
                n += 1

    # Edge cases
    check(fn, torch.zeros(8, 100, device="cuda"), "all zeros (uniform)", failures)
    check(fn, torch.full((4, 50), 1000.0, device="cuda"), "constant large", failures)
    check(fn, torch.empty(0, 16, device="cuda"), "empty rows", failures)
    x = torch.randn(16, 256, device="cuda")
    check(fn, x[:, ::2], "non-contiguous view", failures)
    n += 4
    # Row with one dominant value
    x = torch.zeros(4, 64, device="cuda"); x[:, 3] = 80.0
    check(fn, x, "one-hot dominant", failures)
    n += 1

    print(f"ran {n} checks, {len(failures)} failed")
    return failures


if __name__ == "__main__":
    assert torch.cuda.is_available(), "A CUDA GPU is required (try Colab/Kaggle)."
    fails = run_all(triton_softmax)
    for f in fails:
        print("FAIL:", f)
    print("PASS" if not fails else "FAILED")
    sys.exit(1 if fails else 0)
