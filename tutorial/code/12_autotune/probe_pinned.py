"""Pin ch12 to the same (NS, GSM) config that b42 picked, and time both
side-by-side.  If they tie, the original gap was purely autotuner
ranking noise — not a kernel-level perf difference.  If ch12 is still
slower at the same config, there's a real kernel/harness delta to chase.

b42 effectively pins LDX = 8 (uses tcgen05_ld_32x32b_x8), NW = 8.
So we set ch12 to LDX=8, NW=8 and match b42's NS + GSM.
"""
import os
import sys
import ctypes

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda_utils import (
    cu, init_cuda, compile_kernel, launch,
    encode_tensor_map, TMA_BFLOAT16, TMA_SWIZZLE_128B,
)
from cuda.bindings import driver

sys.path.insert(0, "/data/home/tong/projects/mymatmul")
from mymatmul.gpu.blackwell.matmul_b42_gsm import matmul_b42_gsm
from mymatmul.gpu.blackwell import matmul_b42_gsm as _b42_mod


BM, BN, BK = 128, 256, 64
CTA_GROUP = 2
BN_LOCAL = BN // CTA_GROUP
WARP_SIZE = 32
ELEM_BYTES = 2
NW_PIN  = 8
LDX_PIN = 8         # match b42

SHAPES = list(range(2048, 6144 + 1, 1024))

L2_FLUSH_BYTES = 256 * 1024 * 1024


def main():
    device, ctx = init_cuda()

    def kname(ns, gsm, nw, ldx):
        return f"matmul_tune_ns{ns}_gsm{gsm}_nw{nw}_ldx{ldx}"

    # Only compile the variants we actually need: NS ∈ {3..7}, GSM ∈ {1,4,8,16},
    # at the pinned (NW, LDX).  That's just 20 kernels, not 160.
    NS_NEEDED  = [3, 4, 5, 6, 7]
    GSM_NEEDED = [1, 4, 8, 16]
    needed_names = [kname(ns, gsm, NW_PIN, LDX_PIN)
                    for ns in NS_NEEDED for gsm in GSM_NEEDED]
    print(f"Compiling {len(needed_names)} pinned variants ...", flush=True)
    module, fns = compile_kernel(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.cu"),
        device,
        kernels=needed_names,
    )
    kernels = {(ns, gsm): fns[kname(ns, gsm, NW_PIN, LDX_PIN)]
               for ns in NS_NEEDED for gsm in GSM_NEEDED}

    # Bump SMEM caps per NS.
    A_bytes = BM * BK * ELEM_BYTES
    B_bytes = BN_LOCAL * BK * ELEM_BYTES
    for (ns, gsm), fn in kernels.items():
        cu(driver.cuFuncSetAttribute(
            fn,
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            ns * (A_bytes + B_bytes)))

    def get_fn(ns, gsm):
        return kernels.get((ns, gsm))

    _l2_scratch = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    def invalidate_l2():
        _l2_scratch.zero_()

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

    def shared_for(ns):
        A_bytes = BM * BK * ELEM_BYTES
        B_bytes = BN_LOCAL * BK * ELEM_BYTES
        return ns * (A_bytes + B_bytes)

    def time_ch12(fn, ns, args, grid, n_batches=11, iters=50):
        times_us = []
        for _ in range(n_batches):
            invalidate_l2()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                launch(fn, grid=grid, block=(NW_PIN * WARP_SIZE, 1, 1),
                       shared=shared_for(ns), args=args, sync=False)
            end.record()
            torch.cuda.synchronize()
            times_us.append(start.elapsed_time(end) / iters * 1e3)
        times_us.sort()
        return times_us[len(times_us) // 2]

    def time_b42(A, B, n_batches=11, iters=50):
        for _ in range(2):
            _ = matmul_b42_gsm(A, B)
        torch.cuda.synchronize()
        times_us = []
        for _ in range(n_batches):
            invalidate_l2()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                _ = matmul_b42_gsm(A, B)
            end.record()
            torch.cuda.synchronize()
            times_us.append(start.elapsed_time(end) / iters * 1e3)
        times_us.sort()
        return times_us[len(times_us) // 2]

    print(f"\nPinned-config probe: ch12 forced to b42's (NS, GSM); "
          f"LDX=8, NW=8 on both.\n")
    print(f"  {'shape':<7}  {'b42 cfg':>15}  "
          f"{'ch12 us':>9}  {'b42 us':>9}  {'ch12 TF':>9}  {'b42 TF':>8}  "
          f"{'Δ TF':>7}  {'Δ %':>6}")
    print(f"  {'─'*7}  {'─'*15}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}  "
          f"{'─'*7}  {'─'*6}")

    for sz in SHAPES:
        M = N = K = sz
        A, B, C, args, grid = setup(M, N, K)

        # Trigger b42's autotuner to populate _best for this shape (silently).
        _ = matmul_b42_gsm(A, B)
        torch.cuda.synchronize()
        b42_cfg = _b42_mod._best.get((M, N, K))
        if b42_cfg is None:
            print(f"  {sz}^3   (b42 autotune missing)")
            continue
        bbn, bbk, bns, bgsm = b42_cfg

        # Skip if b42 picked something ch12 can't replicate (e.g. BK=128).
        if (bbn, bbk) != (BN, BK):
            print(f"  {sz}^3   ({bbn},{bbk},{bns},{bgsm})  "
                  f"— skipped: ch12 sweep is BN={BN}, BK={BK} only")
            continue

        fn = get_fn(bns, bgsm)
        if fn is None:
            print(f"  {sz}^3   ({bbn},{bbk},{bns},{bgsm})  "
                  f"— ch12 has no kernel for NS={bns} GSM={bgsm} (LDX={LDX_PIN}, NW={NW_PIN})")
            continue

        # Correctness sanity.
        C.zero_()
        launch(fn, grid=grid, block=(NW_PIN * WARP_SIZE, 1, 1),
               shared=shared_for(bns), args=args)
        torch.cuda.synchronize()
        C_ref = (A.float() @ B.float()).to(torch.bfloat16)
        rel = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()

        us_ch12 = time_ch12(fn, bns, args, grid)
        us_b42  = time_b42(A, B)
        flops = 2.0 * M * N * K
        tf_ch12 = flops / (us_ch12 * 1e-6) / 1e12
        tf_b42  = flops / (us_b42  * 1e-6) / 1e12

        d_tf = tf_b42 - tf_ch12
        d_pc = d_tf / tf_ch12 * 100
        flag = "✓" if rel < 1e-1 else "✗"

        print(f"  {sz}^3   ({bbn:3d},{bbk:2d},{bns:1d},{bgsm:2d})  "
              f"{us_ch12:>9.2f}  {us_b42:>9.2f}  "
              f"{tf_ch12:>9.1f}  {tf_b42:>8.1f}  "
              f"{d_tf:>+7.1f}  {d_pc:>+6.1f}  {flag}")

    cu(driver.cuDevicePrimaryCtxRelease(device))


if __name__ == "__main__":
    main()
