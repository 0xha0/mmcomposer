# mbarrier — async synchronization primitive

> **Status:** stub — content TBD.

`mbarrier` is the SMEM-resident synchronization object that ties async
producers (TMA, tcgen05 commit) to their consumers (MMA warp, epilogue
warps).  It replaces ad-hoc spin-polling and works across CTAs in a
thread block cluster.

## Why mbarrier exists

* **Async ops need a completion signal.**  TMA and tcgen05.commit both
  fire-and-forget; the issuing thread doesn't stall.  We need a way for
  *another* thread (or warp) to know when those ops have retired.
* **Counter-based instead of sequence-based.**  Mbarrier tracks two
  counters (arrival and tx-count); completion = both zero.  This
  lets producers and consumers coordinate without a global ordering.
* **Hardware-resident.**  Lives in SMEM as a 64-bit value (per
  mbarrier).  Polling is via a single PTX instruction, not a memory
  fence + load loop.

## The lifecycle

1. **Init.**  `mbarrier.init.shared::cta.b64 [addr], count` sets
   `{arrival_count = count, tx_count = 0, phase = 0}`.
2. **Producer side.**  Async op fires; either explicitly via
   `mbarrier.arrive.expect_tx [addr], bytes` (which sets the expected
   tx-count and pre-decrements the arrival counter), or implicitly via
   the `mbarrier::complete_tx::bytes` modifier on TMA bulks (which
   auto-decrements tx-count by the bytes transferred).
3. **Wait.**  `mbarrier.try_wait.parity.shared::cta.b64 P, [addr],
   phase` blocks the consumer until the current phase has completed.
   Phase parity flips automatically each completion.
4. **Reuse.**  After completion, the counters auto-reset to
   `{count, 0}` and the consumer waits on the *next* phase.

## The two counters

The mbarrier keeps track of two things in parallel:

### `arrival_count` — counts thread-level arrivals

Set at init time to `N`, the number of arrivals the mbarrier expects
before the current phase can complete.  Each `mbarrier.arrive` (or
`mbarrier.arrive.expect_tx`) executed by a thread decrements it by 1.

When the kernel uses an mbarrier as a thread-synchronization barrier
(like a fancy `__syncthreads`), N is set to the number of threads
that will participate, and each one calls `arrive` exactly once.

When the kernel uses an mbarrier purely for **TMA completion** —
which is what TMA does — the "arrivals" are not really threads
synchronizing with each other.  N is typically set to **1**, and a
single thread (often the same one that issued the TMA) calls
`mbarrier.arrive.expect_tx(bytes)` once.  That single call:

- decrements `arrival_count` from 1 to 0 (the lone "arrival"), and
- adds `bytes` to `tx_count` (declaring how many bytes are coming).

So in TMA-only code paths, "N arrivals" is just a one-time signal
from the producer thread, not the multi-thread rendezvous you'd
get with `__syncthreads`.  It's the *tx_count* that does the real
work — see below.

For 2-CTA cluster TMA (covered in Part 2), the init count goes up to
`CTA_GROUP = 2`: each CTA's TMA warp arrives once, on the same
shared mbarrier.

### `tx_count` — counts bytes-in-flight

Set to 0 at init.  Two things modify it:

- `mbarrier.arrive.expect_tx(bytes)` **adds** `bytes` to it (declaring
  expected work).
- TMA bulks with the `mbarrier::complete_tx::bytes` modifier
  **subtract** the bytes they actually transferred as they land.

When `tx_count` reaches zero, the expected data has fully arrived.

### Phase completion

The current phase completes when **both** counters reach zero
simultaneously — i.e., the right number of arrivals have happened
*and* the expected bytes have all landed.  At that moment the
hardware atomically auto-resets both counters (and flips the parity
bit — see next section), so the same mbarrier can be reused for the
next phase.

## Phase parity (the tricky part)

In addition to the two counters described above, an mbarrier carries
a third piece of state: a single **parity bit**.  Conceptually, the
mbarrier packs all three fields into its 8 bytes:

```
struct mbarrier {
    uint  arrival_count;      // counts down as threads arrive
    uint  tx_count;           // counts down as async-tx bytes land
    bit   parity;             // hardware-managed flip-flop
};
```

When both counters reach zero, the hardware **atomically**:

1. Flips `parity` (the bit inside the mbarrier).
2. Resets `arrival_count` back to the value passed at init.
3. Resets `tx_count` to 0, ready for the next `expect_tx`.

All three transitions happen in one cycle.  This is the **auto-reset**
that makes mbarriers cheap to reuse across many iterations of a
producer/consumer loop — no software intervention is needed to recycle
the mbarrier between phases.

### The phase-parity wait

`try_wait.parity P, [mbar], phaseParity` succeeds when the mbarrier's
*current* parity bit is **opposite** to the `phaseParity` operand.
Equivalently: it succeeds once "the phase whose parity bit was
`phaseParity` has just completed."

To use the same mbarrier across many phases, each waiter keeps a
software **mirror** of the parity bit it expects to see next.  After
each successful wait, it flips its mirror so the next wait checks
against the new parity:

```
hardware parity (inside the mbarrier):  0 → 1 → 0 → 1 → 0 → 1 → ...
software mirror (per-thread variable):  0 → 1 → 0 → 1 → 0 → 1 → ...
```

The two stay in lockstep because the mirror is flipped right after
each wait returns — exactly when the hardware has just flipped its
own parity.

If PTX had a `try_wait.next_phase` form that figured this out
automatically, no software mirror would be needed.  As it stands,
maintaining the mirror is the user's responsibility, and getting it
wrong is one of the easier ways to deadlock an mbarrier-based kernel.

### The pre-arrive trick

After init, the mbarrier's parity is 0 and no phase has completed yet.
If the first user-side `arrive` is *also* a logical reset (e.g. when
the first iteration of a ring buffer has nothing real to wait for),
one common idiom is to **pre-arrive** on the mbarrier immediately
after init.  That brings the counters to zero, triggers an auto-reset
to parity = 1, and lets the first `try_wait.parity` with mirror = 0
return immediately — avoiding a special-case branch at iteration 0.
Concrete patterns for both the waiter loop and the pre-arrive trick
are shown in Part 2.

## The instruction family

```
mbarrier.init.shared::cta.b64 [addr], count;
mbarrier.arrive.shared::cta.b64 _, [addr];
mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [addr], bytes;
mbarrier.try_wait.parity.shared::cta.b64 P, [addr], phaseParity;
```

* `.shared::cta` vs `.shared::cluster` — destination address space.
* `.release.cta` — release-semantics at CTA scope.  For cross-CTA mbar
  use `.release.cluster` if the consumer is on a peer CTA.
* `.acquire.cta` — implicit on `try_wait.parity` when paired with a
  `.release` arrive.

## Scope and cluster mbarriers

Within a single CTA, mbarriers are just SMEM addresses.  In a 2-CTA
cluster, **peer-CTA mbarriers are addressed via a special bit pattern**:
clearing bit 24 of the SMEM address routes the address to CTA 0's
mbarrier from CTA 1.  This is how cross-CTA tx-count bookkeeping works:
both CTAs' TMA bulks target the same CTA 0 mbarrier even though they're
issued from different SMs.  The concrete address-mask idiom is shown in
Part 2's cluster chapter.

## Common pitfalls

* **Arrival count mismatch.**  Init with count=N but only N-1 arrivals
  ever happen → permanent hang.
* **Forgetting `expect_tx`.**  If the TMA bulks have a tx-count modifier
  but no `arrive.expect_tx` was issued, tx-count never reaches zero.
* **Wrong scope qualifier.**  Cross-CTA mbarrier touched with
  `.release.cta` instead of `.release.cluster` → memory ordering
  inconsistent, races possible.
* **Stale phase.**  If your software phase mirror drifts (e.g. you flip
  it twice for one completion), every subsequent wait blocks forever.
* **Init not yet visible.**  `mbarrier.init` is followed by
  `fence.mbarrier_init.release.cluster` before any async ops touch
  the mbar — otherwise the producer might race ahead of the init.
