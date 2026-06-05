"""Runnable companion for Chapter 03b — double-buffer baseline.

Compiles all four GROUP_SIZE_M variants of the baseline kernel and
benchmarks each at three square shapes.  Picks the best GSM at the
biggest shape and reports the speedup over GSM=1 (no swizzle).
"""

import os
import sys
import ctypes

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda_utils import (
    cu, init_cuda, compile_kernel, launch, time_kernel_us,
    encode_tensor_map, TMA_BFLOAT16, TMA_SWIZZLE_128B,
)

from cuda.bindings import driver


BM, BN, BK   = 128, 256, 64
NS           = 2
ELEM_BYTES   = 2
THREADS      = 128
SLOT_BYTES   = BM * BK * ELEM_BYTES + BN * BK * ELEM_BYTES
BN_PAD       = BN + 8
EPI_BYTES    = BM * BN_PAD * ELEM_BYTES
# K-loop ring (NS slots) AND epilogue staging share the same dynamic
# SMEM region, but are time-disjoint — alloc max of the two.
SHARED_BYTES = max(NS * SLOT_BYTES, EPI_BYTES) + 1024
HERE         = os.path.dirname(os.path.abspath(__file__))


device, ctx = init_cuda()

KERNELS = ["matmul_dbuf_gsm1", "matmul_dbuf_gsm4",
           "matmul_dbuf_gsm8", "matmul_dbuf_gsm16"]
mod, fns = compile_kernel(os.path.join(HERE, "kernel.cu"),
                          device, kernels=KERNELS)
k = {name: fns[name] for name in KERNELS}

for kern in k.values():
    cu(driver.cuFuncSetAttribute(
        kern,
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        SHARED_BYTES))


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

    grid = (M // BM * N // BN, 1, 1)
    return A, B, C, grid, args


def time_kernel(kernel, grid, args):
    return time_kernel_us(lambda: launch(
        kernel, grid=grid, block=(THREADS, 1, 1),
        shared=SHARED_BYTES, args=args, sync=False))


for (M, N, K) in [(2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)]:
    A, B, C, grid, args = setup(M, N, K)
    flops = 2.0 * M * N * K

    # Correctness on GSM=1.
    C.zero_()
    launch(k["matmul_dbuf_gsm1"], grid=grid, block=(THREADS, 1, 1),
           shared=SHARED_BYTES, args=args)
    C_ref = (A.float() @ B.float()).to(torch.bfloat16)
    rel = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()
    ok = "✓" if rel < 5e-2 else "✗"

    print(f"{ok}  M=N=K={M:>4}   grid={M//BM}×{N//BN}={(M//BM)*(N//BN)} CTAs   rel err={rel:.2%}")
    for name in KERNELS:
        us = time_kernel(k[name], grid, args)
        tf = flops / (us * 1e-6) / 1e12
        gsm = int(name.rsplit("gsm", 1)[1])
        print(f"     GSM={gsm:>2d}:  {us:7.1f} us/call   {tf:6.1f} TFLOPS")
    print()


cu(driver.cuModuleUnload(mod))
cu(driver.cuDevicePrimaryCtxRelease(device))
