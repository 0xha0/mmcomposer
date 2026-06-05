# Double-buffer baseline

The "all toggles off" point on the ladder.  This kernel uses every
foundational primitive the tutorial has taught up through ch07 —
TMA, tcgen05 MMA, K-major B, the coalesced 2-phase epilogue — but
**deliberately omits warp specialization**.  A single warp issues
both TMA loads and tcgen05 MMA instructions in a serial K-loop, with
a 2-slot SMEM ring buffer providing the "double buffer" overlap.

## Why this chapter exists

The MVP web UI (`webui/app.py`) has two optimization toggles:

1. **Multi-staging + warp specialization**
2. **2-CTA cluster MMA**

When both are **off**, the user is asking for the simplest possible
real kernel.  The natural ladder point is `ch07` minus warp
specialization — that's this chapter.  When the user toggles on
multi-staging+warp-spec, the UI moves them up to ch07 (NS-deep ring,
dedicated TMA + MMA warps).  When they also toggle on 2-CTA, they
move up to ch09 (cluster MMA + chunked walk).

## What it has

| Feature | Source chapter |
|---|---|
| TMA 2D loads, A and B descriptors | ch00, ch01 |
| `tcgen05.mma` async tensor cores via TMEM | ch02 |
| Outer K-loop accumulating into one TMEM | ch03 |
| 2-slot SMEM ring buffer (NS=2 hardcoded) | ch04 |
| K-major B descriptor (`idesc` bit 16, LBO) | ch06 |
| Coalesced 2-phase SMEM-staged epilogue | ch07 |
| Templated `GROUP_SIZE_M` CTA swizzle | ch09 |

## What it doesn't have

- **Warp specialization.**  One warp does all the TMA issue + MMA
  issue + commit work, serially.  This is the load-bearing thing
  the MVP's "Multi-staging + warp specialization" toggle turns on.
- **Deeper multi-stage (NS > 2).**  NS=2 is hardcoded — the
  "double buffer" name.
- **Cluster MMA.**  Single-CTA only.  No `__cluster_dims__`, no
  `cta_group::2`.

## The sync K-loop pattern

The interesting structural difference from ch07: ch07 has warps 0 and
1 running independent K-loops talking through per-slot mbarriers.
Here, warp 0 walks a *single* fused loop:

```cpp
// Prologue: prefetch NS K-tiles into the ring.
for (int s = 0; s < NS; s++) {
    tma_load_into_slot(s);
    mbarrier_arrive_expect_tx(tile_ready[s], SLOT_BYTES);
}

for (int k = 0; k < num_k_iters; k++) {
    const int slot = k % NS;

    // Wait for THIS slot's TMA to land.
    mbarrier_wait(tile_ready[slot]);
    // Fire MMAs into TMEM.
    for (int kk = 0; kk < K_MMAS; kk++) tcgen05_mma(...);
    tcgen05_commit(mma_done[slot]);

    // Prefetch the (k + NS)-th tile into the freed slot.
    if (k + NS < num_k_iters) {
        mbarrier_wait(mma_done[slot]);      // wait for slot to drain
        tma_load_into_slot(slot, k + NS);
        mbarrier_arrive_expect_tx(tile_ready[slot], SLOT_BYTES);
    }
}
```

Even without warp specialization, *some* overlap still happens:
`tcgen05.mma` and `cp.async.bulk.tensor` are both async from the
warp's perspective.  The single warp issues them and continues; the
mbarriers synchronize their effects.  But the warp's instruction
stream is serialized — TMA-issue and MMA-issue can't happen at the
same cycle — which is exactly the cost warp specialization pays to
recover.

## CTA swizzle

`GROUP_SIZE_M` is a template parameter; four launchers expose
GSM ∈ {1, 4, 8, 16}.  GSM=1 collapses to the natural N-fast walk; >1
swaps it for a chunked M-fast walk inside groups of GSM M-rows, for
better L2 reuse on B.  See ch09 for the full rationale.

## Per-shape numbers (B200, BF16)

Square M=N=K, measured via `time_kernel_us` (~µs/call).  GSM picks
the winner per row.

| shape | best GSM | TFLOPS | vs ch07 (~1110 @ 8K) |
|---|---|---|---|
| 2048³ | any | ~540 | ~49 % |
| 4096³ | 4 | ~770 | ~70 % |
| 8192³ | 16 | ~830 | ~75 % |

The gap to ch07 is exactly the warp-spec win — that's the
optimization the MVP's first toggle then unlocks.

## Run

```bash
python main.py
```

Compiles all 4 GSM launchers (~30 s on a cold cubin cache),
checks correctness against PyTorch `A @ B`, then sweeps GSM at three
shapes.
