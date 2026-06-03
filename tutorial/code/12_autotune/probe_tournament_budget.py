"""Test the hypothesis that ch12's autotuner mis-ranks deep-NS configs
at small/mid shapes because the 7×50 tournament is undersampled.

For shapes {3072, 4096, 5120} — the shapes where the pinned probe
showed ch12 winning at b42's chosen NS but the autotuner had picked
shallower NS — run two tournaments per shape:

  default : 7 batches × 50 iters    (what main.py uses)
  beefy   :11 batches × 100 iters   (~3× the samples)

Print the FULL ranked table of all 20 (NS, GSM) configs at the pinned
(LDX=8, NW=8) so we can see how close the top configs are and whether
the ranking changes with more samples.
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

BM, BN, BK = 128, 256, 64
CTA_GROUP = 2
BN_LOCAL = BN // CTA_GROUP
WARP_SIZE = 32
ELEM_BYTES = 2
NW_PIN  = 8
LDX_PIN = 8

NS_SWEEP  = [3, 4, 5, 6, 7]
GSM_SWEEP = [1, 4, 8, 16]
SHAPES = [3072, 4096, 5120]      # the three "ch12 picks too shallow" shapes

L2_FLUSH_BYTES = 256 * 1024 * 1024


def main():
    device, ctx = init_cuda()

    def kname(ns, gsm):
        return f"matmul_tune_ns{ns}_gsm{gsm}_nw{NW_PIN}_ldx{LDX_PIN}"

    names = [kname(ns, gsm) for ns in NS_SWEEP for gsm in GSM_SWEEP]
    print(f"Compiling {len(names)} pinned variants ...", flush=True)
    module, fns = compile_kernel(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel.cu"),
        device, kernels=names,
    )
    kernels = {(ns, gsm): fns[kname(ns, gsm)]
               for ns in NS_SWEEP for gsm in GSM_SWEEP}

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

    def time_median(fn, ns, args, grid, n_batches, iters):
        sh = ns * (A_bytes + B_bytes)
        times_us = []
        for _ in range(n_batches):
            flush_l2()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                launch(fn, grid=grid, block=(NW_PIN * WARP_SIZE, 1, 1),
                       shared=sh, args=args, sync=False)
            end.record()
            torch.cuda.synchronize()
            times_us.append(start.elapsed_time(end) / iters * 1e3)
        times_us.sort()
        return times_us[len(times_us) // 2]

    for sz in SHAPES:
        M = N = K = sz
        A, B, C, args, grid = setup(M, N, K)
        grid_m_clusters = M // (CTA_GROUP * BM)
        flops = 2.0 * M * N * K

        print(f"\n────────── M=N=K={sz} ──────────")
        print(f"  {'(NS,GSM)':>10}  {'default (7×50)':>16}  {'beefy (11×100)':>16}  {'rank shift':>10}")
        print(f"  {'─'*10}  {'─'*16}  {'─'*16}  {'─'*10}")

        # Time every (NS, GSM) at both budgets.
        configs_legal = [(ns, gsm) for ns in NS_SWEEP for gsm in GSM_SWEEP
                         if gsm <= grid_m_clusters]
        default_us = {}
        beefy_us   = {}
        for (ns, gsm) in configs_legal:
            fn = kernels[(ns, gsm)]
            default_us[(ns, gsm)] = time_median(fn, ns, args, grid, 7, 50)
        for (ns, gsm) in configs_legal:
            fn = kernels[(ns, gsm)]
            beefy_us[(ns, gsm)]   = time_median(fn, ns, args, grid, 11, 100)

        default_ranked = sorted(default_us.items(), key=lambda kv: kv[1])
        beefy_ranked   = sorted(beefy_us.items(),   key=lambda kv: kv[1])
        default_rank   = {cfg: i for i, (cfg, _) in enumerate(default_ranked)}
        beefy_rank     = {cfg: i for i, (cfg, _) in enumerate(beefy_ranked)}

        for (cfg, _us) in default_ranked:
            d_tf = flops / (default_us[cfg] * 1e-6) / 1e12
            b_tf = flops / (beefy_us[cfg]   * 1e-6) / 1e12
            shift = beefy_rank[cfg] - default_rank[cfg]
            shift_str = ""
            if shift > 0:
                shift_str = f"↓{shift}"
            elif shift < 0:
                shift_str = f"↑{-shift}"
            else:
                shift_str = "="
            mark_d = " ← default" if default_rank[cfg] == 0 else ""
            mark_b = " ← beefy"   if beefy_rank[cfg]   == 0 else ""
            print(f"  ({cfg[0]},{cfg[1]:>2d})    "
                  f"{default_us[cfg]:>6.2f}us {d_tf:>5.0f}TF  "
                  f"{beefy_us[cfg]:>6.2f}us {b_tf:>5.0f}TF  {shift_str:>10}"
                  f"{mark_d}{mark_b}")

        # Highlight the default vs beefy winner.
        d_win, d_us = default_ranked[0]
        b_win, b_us = beefy_ranked[0]
        d_tf = flops / (d_us * 1e-6) / 1e12
        b_tf = flops / (b_us * 1e-6) / 1e12
        print(f"\n  default tournament picks: NS={d_win[0]} GSM={d_win[1]} → {d_tf:.1f} TF")
        print(f"  beefy   tournament picks: NS={b_win[0]} GSM={b_win[1]} → {b_tf:.1f} TF")
        if d_win != b_win:
            d_tf_at_b = flops / (default_us[b_win] * 1e-6) / 1e12
            b_tf_at_d = flops / (beefy_us[d_win] * 1e-6) / 1e12
            print(f"  → autotuner CHANGED its mind with more samples.")
            print(f"    (beefy winner timed at default settings = {d_tf_at_b:.1f} TF)")
        else:
            print(f"  → autotuner picks SAME config at both budgets.")

    cu(driver.cuDevicePrimaryCtxRelease(device))


if __name__ == "__main__":
    main()
