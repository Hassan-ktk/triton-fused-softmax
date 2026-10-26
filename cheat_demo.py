"""Shows that the verifier catches "cheating" implementations. Run: python cheat_demo.py

This mirrors the real job: anticipate shortcuts that look fast but don't solve the task.
"""
import torch
import verify
from kernel import triton_softmax

# Cheat 1: cache the first result per shape and return it forever (fast, wrong).
_cache = {}
def cheat_cache(x):
    key = (tuple(x.shape), x.dtype)
    if key not in _cache:
        _cache[key] = triton_softmax(x)
    return _cache[key]

# Cheat 2: skip the max-subtraction (fast-ish, but overflows on large inputs).
def cheat_no_stability(x):
    e = torch.exp(x.float())
    return (e / e.sum(dim=-1, keepdim=True)).to(x.dtype)

# Cheat 3: return uniform distribution (passes the "rows sum to 1" check only).
def cheat_uniform(x):
    return torch.full_like(x, 1.0 / x.shape[-1]) if x.numel() else x.clone()

# Cheat 4: only correct for float32, silently wrong precision elsewhere.
def cheat_wrong_dtype(x):
    return triton_softmax(x.float()) if x.dtype == torch.float32 else torch.softmax(x.float(), -1).half().to(x.dtype) * 0.5


if __name__ == "__main__":
    assert torch.cuda.is_available(), "A CUDA GPU is required."
    print("Sanity: honest kernel should pass")
    assert not verify.run_all(triton_softmax), "honest kernel failed!"

    for name, fn in [("cache", cheat_cache), ("no_stability", cheat_no_stability),
                     ("uniform", cheat_uniform), ("wrong_dtype", cheat_wrong_dtype)]:
        print(f"\n--- cheat: {name} ---")
        fails = verify.run_all(fn)
        print("CAUGHT" if fails else "!!! NOT CAUGHT -- strengthen the verifier")
        for f in fails[:3]:
            print("  e.g.", f)
