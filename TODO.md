# TODO

## Epilogue: move `tma_wait_group` after the TMEM→register loads

**File:** `webui/kernels/_overlap_epilogue.cu.frag` (the `EPILOGUE_TMA_PIPELINED` path)

**Change:** In the per-chunk loop, the store-buffer wait is currently hoisted to
the top of the loop, *before* the Phase-1 TMEM→register loads:

```c
for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
    if (ew == 0)
        tma_wait_group<TMA_STORE_STAGES - 1>();   // <-- currently here (loop top)

    float t[LOADS_PER_WARP][8];
    for (...) tcgen05_ld_32x32b_x8(...);          // Phase 1: load into registers
    tcgen05_wait_ld();
    ... bar.sync ...                               // propagates to all epilogue warps
    ... swizzled SMEM write into store_stage buf   // Phase 2: writes the buffer
```

Move it to just **after** `tcgen05_wait_ld()` (and still **before** the existing
`bar.sync` that precedes the buffer write):

```c
    tcgen05_wait_ld();
    if (ew == 0)
        tma_wait_group<TMA_STORE_STAGES - 1>();   // <-- move here
    ... bar.sync ...
```

**Why:** The store buffer is only *written* in Phase 2, so the wait only needs to
guard Phase 2. Hoisting it above the loads means the chunk's `tcgen05.ld` loads
cannot start until the previous reuse of that buffer has drained — losing overlap
when the epilogue is store-bound. Moving it down lets the loads overlap with the
in-flight TMA store drain.

**Correctness:** Keep it before the existing `bar.sync` (`frag:49`/`62`). The wait
is issued by a single elected lane (`ew == 0`); that barrier is what propagates
"buffer free" to all epilogue warps before any of them write the buffer (a WAR
hazard on the shared store buffer). Holds the `t[]` registers live across the
wait — fine, the epilogue is SMEM-occupancy-bound, not register-bound.

**Reference implementation (already done by hand):**
`study-swiglu-extra-store/fused_matmul_swiglu_out_fast_dual_b_ns6_s2.cu` already
uses the desired ordering and is the model to port from. Provenance: this kernel
was produced by generating a base kernel from the generator and then hand-adapting
it into the fused SwiGLU variant; the epilogue was restructured during that manual
adaptation. So this is a **backport** of that hand-improvement into the generator,
not a fix for a regression — the generator never had the better placement.

- `wait_for_store_buffer()` lambda (`:486-490`) bundles the elected-lane
  `tma_wait_group<TMA_STORE_STAGES - 1>()` **and** the propagation
  `bar.sync 1, EPI_THREADS` together.
- It is called **after** `tcgen05_wait_ld()` (`:518` → `:526`), before the
  swizzled buffer write.
- The TMEM-free signal is **decoupled**: handled on the last chunk right after
  the loads (`:520-524`), separate from the store-buffer wait.

When porting to the fragment, mirror this: bundle wait+barrier into one helper
placed after the loads, and split the TMEM-free arrival out of it. The current
fragment's single `bar.sync` does double duty (buffer-free propagation **and**
TMEM-free ordering), so preserve both when restructuring. (Macro names differ:
study kernel uses `EPI_TMEM_EMPTY_ARRIVE`, fragment uses `EPI_TMEM_FREE_ARRIVE`.)

**Also update (this is why it's deferred):** regenerate the affected golden
codegen files and any tests that snapshot the spliced output, e.g.
`webui/codegen/tests/golden/*_overlap_tma_pipelined.cu`,
`webui/codegen/tests/golden/tier3_overlap_bn512_single_tmem.cu`, and whatever
`webui/codegen/tests/test_generate*.py` compares against.
