# Autotuning — picking the right config per shape

> 📁 **Code on GitHub:** [`tutorial/code/12_autotune/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/12_autotune) — `kernel.cu` + `main.py`.

By the end of chapter 11 our kernel has accumulated four template
parameters worth tuning:

| knob | values explored | introduced in |
|---|---|---|
| `NS` (multi-stage depth)         | 3, 4, 5, 6, 7    | ch04 / ch08 |
| `GROUP_SIZE_M` (CTA swizzle)     | 1, 4, 8, 16      | ch09 |
| `NUM_WARPS` (epilogue warps)     | 4, 8             | ch10 |
| `LD_X` (`tcgen05.ld` packing)    | 8, 16, 32, 64    | ch10 |

Every previous chapter pinned them at fixed values "tuned at 8192³".
This chapter does the obvious next thing: **the best config varies by
problem shape, so let's pick the right one per call.**

## The pattern

Three pieces, total ~30 lines of Python:

1. **Compile** the cross product of `(NS, GSM, NW, LDX)` variants at
   startup — for our chapter that's `5 × 4 × 1 × 4 = 80` kernel
   functions in one `kernel.cu`.
2. **Time** each variant for the requested shape with
   `triton.testing.do_bench` and pick the fastest.
3. **Cache** the winner keyed by `(M, N, K)` so subsequent calls at
   the same shape skip the sweep.

```python
import triton.testing

class Autotuner:
    def __init__(self, kernels):
        self.kernels = kernels       # {(NS, GSM, NW, LDX): CUfunction}
        self.cache   = {}            # {(M, N, K): cfg}

    def pick(self, M, N, K, args, grid):
        key = (M, N, K)
        if key in self.cache:
            return self.kernels[self.cache[key]], self.cache[key]

        best_us, best_cfg = float("inf"), None
        for cfg, kern in self.kernels.items():
            ns, gsm, nw, ldx = cfg
            ms_med, _, _ = triton.testing.do_bench(
                lambda kern=kern: launch(
                    kern, grid=grid, block=(nw * WARP_SIZE, 1, 1),
                    shared=shared_for(ns), args=args, sync=False),
                warmup=20, rep=200, quantiles=(0.5, 0.0, 1.0))
            us = ms_med * 1000.0
            if us < best_us:
                best_us, best_cfg = us, cfg

        self.cache[key] = best_cfg
        return self.kernels[best_cfg], best_cfg
```

That's the whole tuner.

## Why `do_bench` and not a hand-rolled timer

GPU kernel timing has well-known foot-guns: per-launch sync vs batched
sync, L2-cache warmth bias across consecutive configs, mean-vs-median
ranking, picking iter counts that put fast and slow kernels on the
same measurement budget.

`triton.testing.do_bench` already handles all of these:

- Flushes a scratch buffer through L2 between samples.
- Adaptively picks iter count to fit the rep window.
- Uses CUDA events with correct sync semantics.
- Returns quantiles for free.

Rolling our own timer that ranks configs reliably is a project of its
own — well beyond what this chapter wants to teach.  We use the
library function and focus on the autotuning loop itself.

> 📝 **Production autotuners** (Triton, CUTLASS-Python, TVM) layer
> ML-guided search, persistent disk caches, multi-shape clustering,
> and cross-shape priors on top of this basic loop.  Those are several
> chapters of an autotuning textbook this isn't.

## One pruning detail worth flagging

The `GROUP_SIZE_M` knob (CTA swizzle, ch09) has a kernel-internal
clamp:

```cpp
gsm = min(GROUP_SIZE_M, grid_m_clusters);
```

At small shapes this collapses several `GSM` values into identical
SASS.  At `M = 2048`, `grid_m_clusters = 8`, so `GSM = 16` produces
literally the same code as `GSM = 8`.  If we don't prune those
equivalent variants before timing, the autotuner spends real time
ranking configs that can't differ except by measurement noise:

```python
grid_m_clusters = M // (CTA_GROUP * BM)
for cfg, kern in self.kernels.items():
    ns, gsm, nw, ldx = cfg
    if gsm > grid_m_clusters:    # would be clamped → skip
        continue
    ...
```

**General lesson: before timing a variant, check whether the kernel
can actually distinguish it from a variant you're already timing.**

## Per-shape results

Sweep `M = N = K ∈ {2048, 3072, …, 12288}` (11 shapes).  Measured on
B200; PyTorch matmul as the cuBLAS baseline.

| shape  | best (NS, GSM, NW, LDX) | **ours TFLOPS** | cuBLAS | ratio |
|---|---|---|---|---|
| 2048³  | (5, 1, 8, 16)   |  **802** |  879 |  91 % |
| 3072³  | (6, 1, 8,  8)   | **1226** | 1377 |  89 % |
| 4096³  | (6, 1, 8,  8)   | **1209** | 1354 |  89 % |
| 5120³  | (6, 8, 8, 32)   | **1254** | 1385 |  90 % |
| 6144³  | (6, 16, 8, 8)   | **1297** | 1407 |  92 % |
| 7168³  | (6, 8, 8, 16)   | **1313** | 1381 |  95 % |
| 8192³  | (5, 8, 8, 32)   | **1313** | 1389 |  95 % |
| 9216³  | (6, 8, 8, 16)   | **1317** | 1349 |  98 % |
| 10240³ | (7, 8, 8,  8)   | **1319** | 1334 |  99 % |
| 11264³ | (6, 8, 8,  8)   | **1347** | 1325 | 102 % |
| 12288³ | (5, 8, 8,  8)   | **1306** | 1428 |  91 % |

Three things to read off:

- **The picked config varies by shape.** No single `(NS, GSM, NW, LDX)`
  is optimal everywhere — that's the whole point of autotuning.
- **`NS ∈ {5, 6, 7}` always wins.** Shallow stages (`NS = 3, 4`)
  never show up across this sweep, so they'd be a candidate to drop
  from a production search space.  We keep them in this chapter's
  sweep so the autotuner has a wider space to demonstrate over.
- **Larger shapes hit higher ratios.** At small shapes (2K-6K) per-CTA
  setup overhead is a bigger fraction of each kernel's wall-clock;
  from 7K up the kernel reaches its compute-bound plateau (~1300
  TFLOPS, 96-98 % of cuBLAS).

## Cost

- **First-call autotune** at 10K-12K takes ~20 seconds because each
  config samples non-trivially-long calls; smaller shapes finish in a
  few seconds.  Subsequent calls at the same shape are essentially
  free (cache hit).
- **The 80-variant nvcc compile** takes a couple of minutes on a cold
  run.  `nvcc` caches cubins by mtime, so subsequent runs of an
  unchanged kernel start instantly.

## What you've built

By the end of chapter 12 our kernel does what a production matmul
kernel does, in roughly the same shape:

- TMA descriptors for swizzled SMEM loads.
- Multi-stage ring buffer.
- Warp-specialized TMA / MMA / epilogue.
- K-major B, no host transpose.
- 2-CTA cluster MMA with `cta_group::2`.
- Triton-style chunked grid walk for L2 reuse.
- Two-phase coalesced epilogue, parameterized by warp count and load
  width.
- Per-shape autotuning over four knobs.

That's **89-102 % of cuBLAS across an 11-shape sweep on B200** —
with a ~1300 TFLOPS plateau from 7K onward, a 95-99 % sweet spot at
8K-11K, and one shape (11264³) where the autotuned config edges past
cuBLAS for this run.  The remaining gap at small shapes is the
territory production libraries claim with SASS-level micro-tuning
that's outside this tutorial's scope.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.  First run compiles
~80 kernel variants (a few minutes); subsequent runs reuse the cubin
cache and start instantly.
