# Autotuning — picking the right config per shape

> 📁 **Code on GitHub:** [`tutorial/code/12_autotune/`](https://github.com/tongzhou8086/mmcomposer/tree/master/tutorial/code/12_autotune) — `kernel.cu` + `main.py`.

By the end of chapter 11 our kernel has accumulated four template
parameters worth tuning:

| knob | values explored | introduced in | actually tuned? |
|---|---|---|---|
| `NS` (multi-stage depth)         | 3, 4, 5, 6, 7    | ch04 / ch08 | ✓ |
| `GROUP_SIZE_M` (CTA swizzle)     | 1, 4, 8, 16      | ch09 | ✓ |
| `NUM_WARPS` (epilogue warps)     | 4, 8             | ch10 | held at 8 |
| `LD_X` (`tcgen05.ld` packing)    | 8, 16, 32, 64    | ch10 | held at 8 |

`NUM_WARPS` was pruned in earlier diagnostic runs (NW = 8 wins or ties
NW = 4 at every shape — see ch10).  `LD_X` turned out to be within
~1 % noise across all four values at every shape (see
[`probe_ldx.py`](probe_ldx.py)) — pruning it cuts the autotune search
4× with no quality loss.  What's left to tune is the
`(NS, GROUP_SIZE_M)` cross product.

Every previous chapter pinned the tunable knobs at fixed values "tuned
at 8192³".  This chapter does the obvious next thing: **the best
config varies by problem shape, so let's pick the right one per call.**

## The pattern

Three pieces, total ~30 lines of Python:

1. **Compile** the cross product of `(NS, GSM)` variants at startup —
   for our chapter that's `5 × 4 = 20` kernel functions in one
   `kernel.cu` (the `NUM_WARPS` and `LD_X` axes are held at their
   default values).
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
    ns, gsm = cfg
    if gsm > grid_m_clusters:    # would be clamped → skip
        continue
    ...
```

**General lesson: before timing a variant, check whether the kernel
can actually distinguish it from a variant you're already timing.**

## When a knob isn't really a knob

We started with four candidate tunables (`NS`, `GSM`, `NW`, `LD_X`).
Two of them turned out not to be worth autotuning:

- **`NUM_WARPS = 8`** wins or ties `4` at every shape we measured
  (see ch10).  We hold it at `8`.
- **`LD_X`** is within ~1 % across `{8, 16, 32, 64}` at every shape —
  see [`probe_ldx.py`](probe_ldx.py).  Run it yourself: the spread is
  0.1–1.7 % shape-by-shape, well below the autotuner's tournament noise
  floor.  We hold it at `8`.

The signature of an irrelevant tunable is that the autotuner picks
different values for it across runs at the same shape.  When we first
benchmarked with all four knobs in the sweep, `LD_X` picks looked
like:

| shape | run 1 | run 2 | run 3 |
|---|---|---|---|
| 8192³ | 32 | 64 | 64 |
| 10240³ | 8 | 64 | 16 |
| 12288³ | 8 | 64 | 64 |

If `LD_X` had a meaningful effect, the same shape would reliably pick
the same `LD_X`.  Instead the autotuner grabs whatever happens to be
measurement-fastest in that tournament, which means it's ranking
noise — wasted autotune budget.  Pruning `LD_X` cuts the search 4×
without losing perf.

That leaves `NS × GSM = 5 × 4 = 20` configs to tune.

## Per-shape results

Sweep `M = N = K ∈ {2048, 3072, …, 12288}` (11 shapes).  Measured on
B200; PyTorch matmul as the cuBLAS baseline.

| shape  | best (NS, GSM) | **ours TFLOPS** | cuBLAS | ratio |
|---|---|---|---|---|
| 2048³  | (5, 1)  |  **798** |  877 |  91 % |
| 3072³  | (6, 1)  | **1259** | 1452 |  87 % |
| 4096³  | (7, 1)  | **1278** | 1412 |  90 % |
| 5120³  | (7, 4)  | **1317** | 1463 |  90 % |
| 6144³  | (5, 8)  | **1377** | 1485 |  93 % |
| 7168³  | (7, 8)  | **1370** | 1453 |  94 % |
| 8192³  | (5, 8)  | **1386** | 1424 |  97 % |
| 9216³  | (6, 8)  | **1386** | 1426 |  97 % |
| 10240³ | (6, 8)  | **1393** | 1441 |  97 % |
| 11264³ | (6, 8)  | **1400** | 1431 |  98 % |
| 12288³ | (5, 8)  | **1401** | 1482 |  95 % |

Three things to read off:

- **The picked config varies by shape.** No single `(NS, GSM)` is
  optimal everywhere — that's the whole point of autotuning.  Small
  shapes prefer `GSM = 1` (the chunked walk has no L2 reuse to give
  it); large shapes consolidate on `GSM = 8`.
- **`NS ∈ {5, 6, 7}` always wins.** Shallow stages (`NS = 3, 4`)
  never show up across this sweep, so they'd be a candidate to drop
  from a production search space.  We keep them in this chapter's
  sweep so the autotuner has a wider space to demonstrate over.
- **Larger shapes hit higher ratios.** At small shapes (2K-6K) per-CTA
  setup overhead is a bigger fraction of each kernel's wall-clock;
  from 7K up the kernel reaches its compute-bound plateau
  (~1380-1400 TFLOPS, 94-98 % of cuBLAS).

## Cost

- **First-call autotune** at 10K-12K takes ~5 seconds (20 configs of
  non-trivially-long calls); smaller shapes finish in a second or two.
  Subsequent calls at the same shape are essentially free (cache hit).
- **The 20-variant nvcc compile** takes ~half a minute on a cold run.
  `nvcc` caches cubins by mtime, so subsequent runs of an unchanged
  kernel start instantly.

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
- Per-shape autotuning over the two knobs that actually matter
  (`NS`, `GSM`).

That's **87-98 % of cuBLAS across an 11-shape sweep on B200** —
with a ~1400 TFLOPS plateau from 6K onward, and a 94-98 % sweet spot
at 7K-11K.  The remaining gap at small shapes is the territory
production libraries claim with SASS-level micro-tuning that's outside
this tutorial's scope.

## Run

```bash
pip install -r ../requirements.txt
python main.py
```

A Blackwell GPU (sm_100a / B200) is required.  First run compiles
~20 kernel variants (~30 seconds); subsequent runs reuse the cubin
cache and start instantly.
