"""Probe: how much does LDX actually matter?

The autotuner picks different LDX values across runs for the same shape
(see the post-migration README discussion).  That's the signature of an
irrelevant tunable.  This probe times all 4 LDX values at fixed (NS, GSM)
across a few representative shapes, so we can see the actual spread.
"""
import os, sys, ctypes, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda_utils import (
    cu, init_cuda, compile_kernel, launch, time_kernel_us,
    encode_tensor_map, TMA_BFLOAT16, TMA_SWIZZLE_128B,
)
from cuda.bindings import driver

BM, BN, BK = 128, 256, 64
CTA_GROUP = 2
BN_LOCAL = BN // CTA_GROUP
WARP_SIZE = 32
ELEM_BYTES = 2
NS_SWEEP  = [3, 4, 5, 6, 7]
GSM_SWEEP = [1, 4, 8, 16]
NW_SWEEP  = [8]
LDX_SWEEP = [8, 16, 32, 64]

A_SLOT_BYTES = BM       * BK * ELEM_BYTES
B_SLOT_BYTES = BN_LOCAL * BK * ELEM_BYTES
SLOT_BYTES   = A_SLOT_BYTES + B_SLOT_BYTES
BN_PAD       = BN + 8
EPI_STAGING  = BM * BN_PAD * ELEM_BYTES

# A representative range of shapes: small / mid / large.
PROBE_SHAPES = [2048, 4096, 8192, 12288]
# A reasonable (NS, GSM) for each shape — based on what the autotuner
# tends to pick in the converged-config range.
PROBE_NS_GSM = {2048: (5, 1), 4096: (6, 1), 8192: (6, 8), 12288: (6, 8)}


def shared_for(ns):
    return max(ns * SLOT_BYTES, EPI_STAGING) + 1024


def kname(ns, gsm, nw, ldx):
    return f"matmul_tune_ns{ns}_gsm{gsm}_nw{nw}_ldx{ldx}"


device, ctx = init_cuda()

# Compile only the 4 LDX variants at each (NS, GSM) we need.
needed = []
for sz, (ns, gsm) in PROBE_NS_GSM.items():
    for ldx in LDX_SWEEP:
        needed.append(kname(ns, gsm, 8, ldx))
needed = list(set(needed))
print(f"Compiling {len(needed)} variants ...", flush=True)
module, fns = compile_kernel(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.cu"),
    device, kernels=needed)

# Set SMEM cap on every kernel we'll use.
for sz, (ns, gsm) in PROBE_NS_GSM.items():
    for ldx in LDX_SWEEP:
        k = fns[kname(ns, gsm, 8, ldx)]
        cu(driver.cuFuncSetAttribute(
            k,
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared_for(ns)))


def setup(M, N, K):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    A_tmap = encode_tensor_map(
        dtype=TMA_BFLOAT16, rank=2, gptr=A.data_ptr(),
        global_dim=[K, M], global_strides=[K * ELEM_BYTES],
        box_dim=[BK, BM], element_strides=[1, 1], swizzle=TMA_SWIZZLE_128B)
    B_tmap = encode_tensor_map(
        dtype=TMA_BFLOAT16, rank=2, gptr=B.data_ptr(),
        global_dim=[N, K], global_strides=[N * ELEM_BYTES],
        box_dim=[64, BK], element_strides=[1, 1], swizzle=TMA_SWIZZLE_128B)
    arg_a = (ctypes.c_byte * 128).from_buffer_copy(A_tmap.tobytes())
    arg_b = (ctypes.c_byte * 128).from_buffer_copy(B_tmap.tobytes())
    arg_c = ctypes.c_void_p(C.data_ptr())
    args = [arg_a, arg_b, arg_c,
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]
    grid = ((M // (CTA_GROUP * BM)) * (N // BN) * CTA_GROUP, 1, 1)
    return A, B, C, args, grid


# Time each (shape, LDX) twice — once in order x8→x64, once reversed,
# so we can see how much of the spread is order-of-execution noise.
print()
print(f"  {'shape':<7}  {'(NS,GSM)':>8}   {'x8 TF':>7}  {'x16 TF':>7}  {'x32 TF':>7}  {'x64 TF':>7}   {'spread':>7}")
print(f"  {'─'*7}  {'─'*8}   {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}   {'─'*7}")
for sz in PROBE_SHAPES:
    ns, gsm = PROBE_NS_GSM[sz]
    A, B, C, args, grid = setup(sz, sz, sz)
    flops = 2.0 * sz * sz * sz

    # Two orderings — take min of (fwd, rev) at each LDX to reduce thermal bias.
    results_fwd = {}
    for ldx in LDX_SWEEP:
        k = fns[kname(ns, gsm, 8, ldx)]
        us = time_kernel_us(lambda kk=k: launch(
            kk, grid=grid, block=(8 * WARP_SIZE, 1, 1),
            shared=shared_for(ns), args=args, sync=False),
            warmup_ms=50, rep_ms=500)
        results_fwd[ldx] = us
    results_rev = {}
    for ldx in reversed(LDX_SWEEP):
        k = fns[kname(ns, gsm, 8, ldx)]
        us = time_kernel_us(lambda kk=k: launch(
            kk, grid=grid, block=(8 * WARP_SIZE, 1, 1),
            shared=shared_for(ns), args=args, sync=False),
            warmup_ms=50, rep_ms=500)
        results_rev[ldx] = us

    tflops = {}
    for ldx in LDX_SWEEP:
        # Use min of two orderings as the best estimate.
        us = min(results_fwd[ldx], results_rev[ldx])
        tflops[ldx] = flops / (us * 1e-6) / 1e12

    tfs = list(tflops.values())
    spread_pct = (max(tfs) - min(tfs)) / min(tfs) * 100

    row = f"  {sz}^3   ({ns},{gsm:>2d})   "
    row += "  ".join(f"{tflops[ldx]:>7.1f}" for ldx in LDX_SWEEP)
    row += f"   {spread_pct:>5.1f}%"
    print(row)

cu(driver.cuDevicePrimaryCtxRelease(device))
