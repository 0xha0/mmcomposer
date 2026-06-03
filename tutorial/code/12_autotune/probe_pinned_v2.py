"""Re-verify the pinned-config probe with order-swapping and multiple trials.

The original probe_pinned.py timed ch12 first, then b42 for each shape.
If timing the second kernel happens to find the GPU hotter, that biases
the comparison toward whichever was timed first.  Here we:

  - Run BOTH orderings per shape: (ch12 then b42) and (b42 then ch12).
  - Repeat each ordering N trials and report min across trials (closest
    to true device time).
  - Apply a global pre-warm before any timing.

If the kernel is genuinely equal at the same config, the two orderings
should agree (within run-to-run noise).  If ch12's apparent win was
just an order artifact, the swapped order will favor b42.
"""
import os
import sys
import ctypes
import time

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
LDX_PIN = 8

SHAPES = list(range(2048, 6144 + 1, 1024))
TRIALS = 3                                     # repeat each ordering 3x
WARMUP_SEC = 5.0
ITERS_PER_REP = 50
REPS_TIMING = 7                                # min over 7 batches of 50 launches

L2_FLUSH_BYTES = 256 * 1024 * 1024


def main():
    device, ctx = init_cuda()

    def kname(ns, gsm):
        return f"matmul_tune_ns{ns}_gsm{gsm}_nw{NW_PIN}_ldx{LDX_PIN}"

    NS_NEEDED  = [3, 4, 5, 6, 7]
    GSM_NEEDED = [1, 4, 8, 16]
    needed_names = [kname(ns, gsm) for ns in NS_NEEDED for gsm in GSM_NEEDED]
    print(f"Compiling {len(needed_names)} pinned variants ...", flush=True)
    module, fns = compile_kernel(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.cu"),
        device, kernels=needed_names,
    )
    kernels = {(ns, gsm): fns[kname(ns, gsm)]
               for ns in NS_NEEDED for gsm in GSM_NEEDED}

    A_bytes = BM * BK * ELEM_BYTES
    B_bytes = BN_LOCAL * BK * ELEM_BYTES
    for (ns, gsm), fn in kernels.items():
        cu(driver.cuFuncSetAttribute(
            fn,
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            ns * (A_bytes + B_bytes)))

    _l2 = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    def flush_l2(): _l2.zero_()

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

    def time_ch12(fn, ns, args, grid):
        sh = ns * (A_bytes + B_bytes)
        vals = []
        for _ in range(REPS_TIMING):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(ITERS_PER_REP):
                launch(fn, grid=grid, block=(NW_PIN * WARP_SIZE, 1, 1),
                       shared=sh, args=args, sync=False)
            e.record()
            torch.cuda.synchronize()
            vals.append(s.elapsed_time(e) / ITERS_PER_REP * 1e3)
        return min(vals)

    def time_b42(A, B):
        # Trigger b42's autotuner first if not cached.
        for _ in range(2):
            _ = matmul_b42_gsm(A, B)
        torch.cuda.synchronize()
        vals = []
        for _ in range(REPS_TIMING):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(ITERS_PER_REP):
                _ = matmul_b42_gsm(A, B)
            e.record()
            torch.cuda.synchronize()
            vals.append(s.elapsed_time(e) / ITERS_PER_REP * 1e3)
        return min(vals)

    # ── Pre-warm: 5s of mixed launches across all shapes ────────────────
    per_shape = {}
    for sz in SHAPES:
        A, B, C, args, grid = setup(sz, sz, sz)
        per_shape[sz] = (A, B, C, args, grid)

    print(f"\nPre-warm: {WARMUP_SEC:.0f}s of mixed launches ...", flush=True)
    warm_t0 = time.time()
    cfgs = list(kernels.keys())
    i = 0
    while time.time() - warm_t0 < WARMUP_SEC:
        sz = SHAPES[i % len(SHAPES)]
        cfg = cfgs[i % len(cfgs)]
        ns = cfg[0]
        _, _, _, args, grid = per_shape[sz]
        launch(kernels[cfg], grid=grid, block=(NW_PIN * WARP_SIZE, 1, 1),
               shared=ns * (A_bytes + B_bytes), args=args, sync=False)
        i += 1
        if i % 1000 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    print(f"  ran {i} warmup launches\n", flush=True)

    # ── Trigger b42's autotuner upfront for all shapes (uses its own ──
    #    methodology — we don't time those).
    for sz in SHAPES:
        A, B, _, _, _ = per_shape[sz]
        _ = matmul_b42_gsm(A, B)
        torch.cuda.synchronize()

    # ── For each shape: do TRIALS trials of each ordering ──
    print(f"  {'shape':<7}  {'b42 cfg':>17}  {'ordering':>14}  {'ch12 us':>9}  "
          f"{'b42 us':>9}  {'ch12 TF':>8}  {'b42 TF':>8}  {'Δ TF':>7}")
    print(f"  {'─'*7}  {'─'*17}  {'─'*14}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*7}")

    for sz in SHAPES:
        M = N = K = sz
        A, B, C, args, grid = per_shape[sz]

        b42_cfg = _b42_mod._best.get((M, N, K))
        if b42_cfg is None:
            print(f"  {sz}^3: b42 autotune missing")
            continue
        bbn, bbk, bns, bgsm = b42_cfg
        if (bbn, bbk) != (BN, BK):
            print(f"  {sz}^3   ({bbn},{bbk},{bns},{bgsm})  — b42 picked BK!=64; skipped")
            continue

        fn = kernels.get((bns, bgsm))
        if fn is None:
            print(f"  {sz}^3   ({bbn},{bbk},{bns},{bgsm})  — no ch12 kernel for this NS/GSM at LDX=8")
            continue

        # Correctness check (once per shape).
        C.zero_()
        launch(fn, grid=grid, block=(NW_PIN * WARP_SIZE, 1, 1),
               shared=bns * (A_bytes + B_bytes), args=args)
        torch.cuda.synchronize()
        C_ref = (A.float() @ B.float()).to(torch.bfloat16)
        rel = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()
        assert rel < 1e-1, f"correctness failed: rel={rel:.3e}"

        flops = 2.0 * M * N * K

        # Trial pairs: 3 trials of (ch12, b42) and 3 trials of (b42, ch12).
        trials_ch12_first = []
        trials_b42_first  = []
        for _t in range(TRIALS):
            # Order 1: ch12 then b42
            us_c = time_ch12(fn, bns, args, grid)
            us_b = time_b42(A, B)
            trials_ch12_first.append((us_c, us_b))
            # Order 2: b42 then ch12
            us_b = time_b42(A, B)
            us_c = time_ch12(fn, bns, args, grid)
            trials_b42_first.append((us_c, us_b))

        for label, trials in [("ch12 first", trials_ch12_first),
                              ("b42  first", trials_b42_first)]:
            ch12_min = min(t[0] for t in trials)
            b42_min  = min(t[1] for t in trials)
            tf_c = flops / (ch12_min * 1e-6) / 1e12
            tf_b = flops / (b42_min  * 1e-6) / 1e12
            d_tf = tf_b - tf_c
            print(f"  {sz}^3   ({bbn:3d},{bbk:2d},{bns:1d},{bgsm:2d})    "
                  f"{label}    {ch12_min:>9.2f}  {b42_min:>9.2f}  "
                  f"{tf_c:>8.1f}  {tf_b:>8.1f}  {d_tf:>+7.1f}")

    cu(driver.cuDevicePrimaryCtxRelease(device))


if __name__ == "__main__":
    main()
