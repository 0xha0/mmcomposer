"""Probe: switch ch12 timing to `min` (over fewer, shorter reps) + a long
mixed-config pre-warm to settle GPU clocks BEFORE any measurement.

Methodology change from main.py:
  median over 7×50 (default) or 11×100 (beefy)
  → min over 5×20, with a 5-second mixed pre-warm at the start.

`min` is the closest measurement to the device's true peak — thermal
throttling and scheduling jitter can only slow a kernel down, never
speed it up.  Combined with thermal-stable pre-warm, per-config
rankings should converge.

Smoking-gun test: run the entire sweep TWICE per shape — once in
forward (NS=3..7, GSM=1..16) order, once in reverse — and check that
both orderings produce the SAME winner.  If yes, the autotuner is
stable.  If no, we still have an order-of-execution problem.

Borrowed from /data/home/tong/projects/swiglu-recompute/bench_stacked_blocks.py
which got stable timings with iters=8, reps=10, min.
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

BM, BN, BK = 128, 256, 64
CTA_GROUP = 2
BN_LOCAL  = BN // CTA_GROUP
WARP_SIZE = 32
ELEM_BYTES = 2
NW_PIN  = 8
LDX_PIN = 8

NS_SWEEP  = [3, 4, 5, 6, 7]
GSM_SWEEP = [1, 4, 8, 16]
SHAPES    = [3072, 4096, 5120]

# Timing knobs (swiglu-recompute style).
WARMUP_SEC      = 5.0     # mixed-config pre-warm to settle GPU clocks
ITERS_PER_REP   = 20      # launches inside one timed batch
REPS            = 5       # number of timed batches per config
                          #   total = 100 launches/config — about 30%
                          #   of the default 7×50=350.

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

    def time_min(fn, ns, args, grid):
        sh = ns * (A_bytes + B_bytes)
        # No per-batch L2 flush, no per-batch warmup — the global pre-warm
        # handled both.  This matches swiglu-recompute's pattern.
        vals = []
        for _ in range(REPS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(ITERS_PER_REP):
                launch(fn, grid=grid, block=(NW_PIN * WARP_SIZE, 1, 1),
                       shared=sh, args=args, sync=False)
            e.record()
            torch.cuda.synchronize()
            vals.append(s.elapsed_time(e) / ITERS_PER_REP * 1e3)   # to µs
        return min(vals)

    # ── Per-shape setup, ahead of warmup ─────────────────────────────────
    per_shape = {}
    for sz in SHAPES:
        A, B, C, args, grid = setup(sz, sz, sz)
        per_shape[sz] = (A, B, C, args, grid)

    # ── Long mixed-config pre-warm to settle thermal/clock state ────────
    print(f"\nPre-warm: {WARMUP_SEC:.0f} s of mixed-shape, mixed-config launches "
          f"to settle clocks ...", flush=True)
    warm_t0 = time.time()
    i = 0
    while time.time() - warm_t0 < WARMUP_SEC:
        sz = SHAPES[i % len(SHAPES)]
        ns_g = (NS_SWEEP[(i // len(SHAPES)) % len(NS_SWEEP)],
                GSM_SWEEP[(i // (len(SHAPES) * len(NS_SWEEP))) % len(GSM_SWEEP)])
        fn = kernels[ns_g]
        ns = ns_g[0]
        _, _, _, args, grid = per_shape[sz]
        launch(fn, grid=grid, block=(NW_PIN * WARP_SIZE, 1, 1),
               shared=ns * (A_bytes + B_bytes), args=args, sync=False)
        i += 1
        if i % 1000 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    print(f"  ran {i} warmup launches\n", flush=True)

    # ── For each shape, time every (NS, GSM) twice: forward + reverse ───
    for sz in SHAPES:
        M = N = K = sz
        A, B, C, args, grid = per_shape[sz]
        grid_m_clusters = M // (CTA_GROUP * BM)
        flops = 2.0 * M * N * K

        configs_legal = [(ns, gsm) for ns in NS_SWEEP for gsm in GSM_SWEEP
                         if gsm <= grid_m_clusters]
        configs_rev   = list(reversed(configs_legal))

        fwd_us, rev_us = {}, {}
        for cfg in configs_legal:
            fn = kernels[cfg]
            fwd_us[cfg] = time_min(fn, cfg[0], args, grid)
        for cfg in configs_rev:
            fn = kernels[cfg]
            rev_us[cfg] = time_min(fn, cfg[0], args, grid)

        # Avg of fwd+rev = best estimate of true per-config min.
        merged = {cfg: min(fwd_us[cfg], rev_us[cfg]) for cfg in configs_legal}
        ranked = sorted(merged.items(), key=lambda kv: kv[1])
        fwd_rank = {cfg: i for i, (cfg, _)
                    in enumerate(sorted(fwd_us.items(), key=lambda kv: kv[1]))}
        rev_rank = {cfg: i for i, (cfg, _)
                    in enumerate(sorted(rev_us.items(), key=lambda kv: kv[1]))}

        print(f"────────── M=N=K={sz} ──────────")
        print(f"  {'(NS,GSM)':>10}  {'fwd us':>8}  {'rev us':>8}  "
              f"{'min(f,r) us':>11}  {'min TF':>7}  {'f rank':>7}  {'r rank':>7}")
        print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*11}  {'─'*7}  {'─'*7}  {'─'*7}")
        for cfg, _us in ranked:
            tf = flops / (merged[cfg] * 1e-6) / 1e12
            mark = ""
            if fwd_rank[cfg] == 0 and rev_rank[cfg] == 0:
                mark = " ← winner (both)"
            elif fwd_rank[cfg] == 0:
                mark = " ← fwd winner"
            elif rev_rank[cfg] == 0:
                mark = " ← rev winner"
            print(f"  ({cfg[0]},{cfg[1]:>2d})    "
                  f"{fwd_us[cfg]:>8.2f}  {rev_us[cfg]:>8.2f}  "
                  f"{merged[cfg]:>11.2f}  {tf:>7.1f}  "
                  f"{fwd_rank[cfg]:>7d}  {rev_rank[cfg]:>7d}{mark}")
        fwd_winner = min(fwd_us.items(), key=lambda kv: kv[1])[0]
        rev_winner = min(rev_us.items(), key=lambda kv: kv[1])[0]
        if fwd_winner == rev_winner:
            print(f"\n  ✓ Both orderings pick (NS={fwd_winner[0]}, GSM={fwd_winner[1]}).\n")
        else:
            print(f"\n  ✗ Order-dependent: fwd picks (NS={fwd_winner[0]}, GSM={fwd_winner[1]}), "
                  f"rev picks (NS={rev_winner[0]}, GSM={rev_winner[1]}).\n")

    cu(driver.cuDevicePrimaryCtxRelease(device))


if __name__ == "__main__":
    main()
