"""Quickstart: inference-style fused GEMM + SwiGLU (dual-B), no preact store.

    python examples/quickstart_swiglu_inference.py                  # default FFN shape 30000x4608x768
    python examples/quickstart_swiglu_inference.py 8192             # square 8192
    python examples/quickstart_swiglu_inference.py 30000 4608 768   # M N K

A = [M, K]; packed projection weight B = [K, N], split by column views into
B_left, B_gate = [K, N/2]; bf16, M arbitrary (ragged token counts welcome),
N a multiple of 256, K a multiple of 64.

This inference path uses store_preact=False and returns only:

    d = [M, N/2]    left * silu(gate)   <- the SwiGLU activation
"""
import sys

import torch
import torch.nn.functional as F
from triton.testing import do_bench

import mmcomposer as mmc

# Shape from the command line: no args -> production FFN shape; one arg N -> square N;
# three args -> M N K.
args = sys.argv[1:]
if len(args) == 0:
    M, N, K = 30000, 4608, 768
elif len(args) == 1:
    M = N = K = int(args[0])
elif len(args) == 3:
    M, N, K = (int(x) for x in args)
else:
    sys.exit("usage: quickstart_swiglu_inference.py [N | M N K]")

H = N // 2
a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
b_left = b[:, :H]
b_gate = b[:, H:]

# One fused launch -> SwiGLU activation only.
d = mmc.matmul_swiglu_dual_b(a, b_left, b_gate, store_preact=False)

# Correctness vs torch (bf16 tolerances).
d_ref = (a @ b_left) * F.silu(a @ b_gate)
ok_d = torch.allclose(d, d_ref, rtol=2e-2, atol=1e-1)
ragged = f"  (M ragged: M % 128 = {M % 128})" if M % 128 else ""
print(f"shape M={M} N={N} K={K}   D allclose = {ok_d}{ragged}")
assert ok_d

# GPU kernel time (triton do_bench: warmup 1000 ms, rep 1000 ms, median):
# the fused kernel vs torch doing the same SwiGLU eagerly (two GEMMs + gate).
flops = 2.0 * M * N * K
g = do_bench(lambda: mmc.matmul_swiglu_dual_b(a, b_left, b_gate,
                                             store_preact=False, out=d,
                                             sync=False),
             warmup=1000, rep=1000, return_mode="median")
t = do_bench(lambda: (a @ b_left) * F.silu(a @ b_gate),
             warmup=1000, rep=1000, return_mode="median")
print(f"mmc fused    {g:8.3f} ms   {flops / (g * 1e-3) / 1e12:7.0f} TFLOPS")
print(f"torch eager  {t:8.3f} ms   {flops / (t * 1e-3) / 1e12:7.0f} TFLOPS"
      f"   (mmc/torch = {g / t:.3f})")
