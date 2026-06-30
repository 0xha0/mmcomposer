"""Auxiliary-input epilogue (phase 2) -- CURRENTLY A BUG REPRODUCER, not a clean demo.

This is the intended quickstart for a multi-arg epilogue that combines the matmul
result with an extra same-shape [M,N] input, fused into the GEMM store path:

    gate = lambda x, c: x * c
    out  = mmc.matmul(a, b, epilogue=gate, aux=[c])     # out = (a @ b) * c

The kernel path is implemented and numerically correct (verified on 163/198
production configs), BUT this entry point is NOT yet runnable end-to-end: it goes
through the tuned-variant autotune sweep, and ~35/198 configs hit
CUDA_ERROR_LAUNCH_FAILED with the extra-input load. A launch failure poisons the
CUDA context for the whole process, so the sweep (and this script) crash -- the
failure surfaces as "unspecified launch failure" (often only at shutdown, via
CUDAEvent destructor warnings) or "tuning produced no valid config".

So running this script is EXPECTED TO CRASH today; it exists to reproduce that
bug. See linear/epilogue-aux-status.md for the full diagnosis (it's a secondary
kernel interaction, not an out-of-bounds; plain matmul tuning is unaffected,
198/198) and the planned fixes (process-isolated tuning / reuse-geometry).

    python examples/quickstart_epilogue_aux.py                   # FFN 32768x4608x768 (crashes)
    python examples/quickstart_epilogue_aux.py 8192              # square 8192

Once the bug is fixed this file becomes the real runnable aux quickstart.
"""
import sys

import torch
from triton.testing import do_bench

import mmcomposer as mmc

gate = lambda x, c: x * c            # noqa: E731   out = (a @ b) * c

args = sys.argv[1:]
if len(args) == 0:
    M, N, K = 32768, 4608, 768
elif len(args) == 1:
    M = N = K = int(args[0])
elif len(args) == 3:
    M, N, K = (int(x) for x in args)
else:
    sys.exit("usage: quickstart_epilogue_aux.py [N | M N K]")

a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
c = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")     # the extra input [M,N]

# Tuned-variant sweep runs here -> may hit a config that LAUNCH_FAILs and poisons
# the context (the bug this script reproduces).
out = mmc.matmul(a, b, epilogue=gate, aux=[c])

ref = (a.float() @ b.float()) * c.float()
ok = torch.allclose(out.float(), ref, rtol=2e-2, atol=1e-1)
print(f"shape M={M} N={N} K={K}   (a@b)*c allclose vs torch = {ok}")
assert ok

buf = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
flops = 2.0 * M * N * K
g = do_bench(lambda: mmc.matmul(a, b, epilogue=gate, aux=[c], out=buf, sync=False),
             warmup=1000, rep=1000, return_mode="median")
t = do_bench(lambda: torch.mm(a, b) * c, warmup=1000, rep=1000, return_mode="median")
print(f"mmc fused (matmul*c)      {g:8.3f} ms   {flops / (g * 1e-3) / 1e12:7.0f} TFLOPS")
print(f"torch (matmul then *c)    {t:8.3f} ms   {flops / (t * 1e-3) / 1e12:7.0f} TFLOPS"
      f"   (mmc/torch = {g / t:.3f})")
