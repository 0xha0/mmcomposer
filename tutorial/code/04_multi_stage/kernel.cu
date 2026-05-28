// Runnable companion for Chapter 04 — multi-stage buffering.
//
// Single-CTA matmul with NUM_STAGES = 2 SMEM slots.  TMA warp and MMA
// warp run independent K-loops, communicating only through a per-slot
// mbarrier pair (tile_ready[s] / mma_done[s]).  Result: TMA and MMA
// overlap, ~2x throughput over chapter 03.
//
// Problem:  C[BM, BN] = A[BM, K] @ B[K, BN],   BM = 128, BN = 256.
// K is runtime, must be a multiple of BK = 64.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

constexpr int BM      = 128;
constexpr int BN      = 256;
constexpr int BK      = 64;
constexpr int MMA_K   = 16;
constexpr int BF16_BYTES = 2;        // byte size of the operand element
constexpr int K_MMAS  = BK / MMA_K;          // 4

constexpr int NS      = 2;                   // NUM_STAGES — the new knob

constexpr int A_SLOT_BYTES = BM * BK * BF16_BYTES;    // 16 KB
constexpr int B_SLOT_BYTES = BN * BK * BF16_BYTES;    // 32 KB
constexpr int SLOT_BYTES   = A_SLOT_BYTES + B_SLOT_BYTES;          // 48 KB / slot
constexpr int TILE_BYTES_TOTAL = NS * SLOT_BYTES;                  // NS × 48 KB

constexpr int THREADS   = 128;
constexpr int WARP_SIZE = 32;


// ── helpers (same wrappers as ch03) ─────────────────────────────────
__device__ __forceinline__ bool elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t .reg .pred px;\n\t"
        "elect.sync _|px, %1;\n\t"
        "@px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred) : "r"(0xFFFFFFFF));
    return pred;
}

__device__ __forceinline__ void tma_2d_load(
    uint32_t smem_dst, const void* tmap, int x, int y, uint32_t mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tmap), "r"(x), "r"(y), "r"(mbar) : "memory");
}

__device__ __forceinline__ uint64_t make_desc(uint32_t smem_addr) {
    constexpr uint64_t SBO = 8 * 128;
    uint64_t a = ((uint64_t)smem_addr >> 4) & 0x3FFFULL;
    uint64_t b = ((SBO)              >> 4) & 0x3FFFULL;
    return a | (b << 32) | (1ULL << 46) | (2ULL << 61);   // SWIZZLE_128B
}

__device__ __forceinline__ uint32_t make_idesc_bf16(int m, int n) {
    uint32_t d = 0;
    d |= (1u << 4); d |= (1u << 7); d |= (1u << 10);
    d |= (((uint32_t)(n >> 3) & 0x3F) << 17);
    d |= (((uint32_t)(m >> 4) & 0x1F) << 24);
    return d;
}

__device__ __forceinline__ void tcgen05_alloc(uint32_t smem_dst, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_dst), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_dealloc(uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(n_cols) : "memory");
}
__device__ __forceinline__ void tcgen05_mma(
    uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
    uint32_t idesc, bool enable_d
) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "setp.ne.b32 P, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, P;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
           "r"((uint32_t)enable_d) : "memory");
}
__device__ __forceinline__ void tcgen05_commit(uint32_t smem_bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(smem_bar) : "memory");
}
__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}
__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_ld_32x32b_x8(uint32_t taddr, float* out) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=f"(out[0]), "=f"(out[1]), "=f"(out[2]), "=f"(out[3]),
          "=f"(out[4]), "=f"(out[5]), "=f"(out[6]), "=f"(out[7])
        : "r"(taddr));
}

__device__ __forceinline__ void mbarrier_init(uint32_t mb, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mb), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_no_tx(uint32_t mb) {
    // Plain arrive, no tx-count change.  Used to pre-arm an mbarrier so
    // its first wait returns immediately.
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mb) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t mb, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(mb), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbarrier_wait_phase(uint32_t mb, uint32_t phase) {
    asm volatile(
        "{\n\t .reg .pred P;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t bra WAIT_%=;\n\t DONE_%=:\n\t }"
        :: "r"(mb), "r"(phase) : "memory");
}


// ── Kernel ──────────────────────────────────────────────────────────
extern "C" __global__ void matmul_multi_stage(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    __nv_bfloat16* __restrict__ C_ptr,
    int K
) {
    extern __shared__ __align__(1024) char smem[];
    const uint32_t SMEM_BASE = (uint32_t)__cvta_generic_to_shared(smem);

    // Per-slot SMEM addresses: A[s] at SMEM_BASE + s*SLOT_BYTES, B[s] right after A[s].
    auto A_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * SLOT_BYTES;
    };
    auto B_base = [SMEM_BASE](int s) -> uint32_t {
        return SMEM_BASE + s * SLOT_BYTES + A_SLOT_BYTES;
    };

    __shared__ uint64_t tile_ready[NS];
    __shared__ uint64_t mma_done[NS];
    __shared__ uint64_t all_mmas_done;
    __shared__ uint32_t tmem_addr_holder[1];

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid % WARP_SIZE;

    // ── One-time setup ──────────────────────────────────────────────
    if (warp_id == 0 && elect_sync()) {
        #pragma unroll
        for (int s = 0; s < NS; s++) {
            mbarrier_init((uint32_t)__cvta_generic_to_shared(&tile_ready[s]), 1);
            mbarrier_init((uint32_t)__cvta_generic_to_shared(&mma_done[s]),   1);
        }
        mbarrier_init((uint32_t)__cvta_generic_to_shared(&all_mmas_done), 1);

        // Pre-arrive mma_done[NS-1] so the TMA warp's *first* steady-state
        // wait returns immediately (no MMA has fired yet for that slot,
        // but the slot is in fact free to load into).
        mbarrier_arrive_no_tx((uint32_t)__cvta_generic_to_shared(&mma_done[NS - 1]));

        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        tcgen05_alloc((uint32_t)__cvta_generic_to_shared(tmem_addr_holder), BN);
    }
    __syncthreads();
    const uint32_t taddr = tmem_addr_holder[0];
    const uint32_t idesc = make_idesc_bf16(BM, BN);

    const int num_k_iters = K / BK;

    // ── TMA warp: prologue + steady-state ───────────────────────────
    //
    // Loads `num_k_iters` tiles into the NS-slot ring buffer.  The
    // first NS-1 are loaded unconditionally (prologue); the rest wait
    // on the corresponding mma_done before reusing a slot.
    if (warp_id == 0 && elect_sync()) {
        uint32_t mma_done_phase[NS] = {};

        // Prologue
        #pragma unroll
        for (int s = 0; s < NS - 1; s++) {
            const uint32_t mb = (uint32_t)__cvta_generic_to_shared(&tile_ready[s]);
            tma_2d_load(A_base(s), &A_tmap, /*x=*/ s * BK, /*y=*/ 0, mb);
            tma_2d_load(B_base(s), &B_tmap, /*x=*/ s * BK, /*y=*/ 0, mb);
            mbarrier_arrive_expect_tx(mb, SLOT_BYTES);
        }

        // Steady-state: TMA the (k + NS - 1)-th tile into slot (k + NS - 1) % NS,
        // after the slot's previous MMA has drained.
        for (int k = 0; k < num_k_iters - (NS - 1); k++) {
            const int slot = (k + NS - 1) % NS;
            const uint32_t done_mb  = (uint32_t)__cvta_generic_to_shared(&mma_done[slot]);
            const uint32_t ready_mb = (uint32_t)__cvta_generic_to_shared(&tile_ready[slot]);

            mbarrier_wait_phase(done_mb, mma_done_phase[slot]);
            tma_2d_load(A_base(slot), &A_tmap, /*x=*/ (k + NS - 1) * BK, /*y=*/ 0, ready_mb);
            tma_2d_load(B_base(slot), &B_tmap, /*x=*/ (k + NS - 1) * BK, /*y=*/ 0, ready_mb);
            mbarrier_arrive_expect_tx(ready_mb, SLOT_BYTES);
            mma_done_phase[slot] ^= 1;
        }
    }

    // ── MMA warp: flat K-loop ───────────────────────────────────────
    else if (warp_id == 1 && elect_sync()) {
        uint32_t tile_ready_phase[NS] = {};

        for (int k = 0; k < num_k_iters; k++) {
            const int slot = k % NS;
            const uint32_t ready_mb = (uint32_t)__cvta_generic_to_shared(&tile_ready[slot]);
            const uint32_t done_mb  = (uint32_t)__cvta_generic_to_shared(&mma_done[slot]);

            mbarrier_wait_phase(ready_mb, tile_ready_phase[slot]);
            tcgen05_fence_after_thread_sync();

            #pragma unroll
            for (int kk = 0; kk < K_MMAS; kk++) {
                const uint64_t a_desc = make_desc(A_base(slot) + kk * MMA_K * BF16_BYTES);
                const uint64_t b_desc = make_desc(B_base(slot) + kk * MMA_K * BF16_BYTES);
                const bool first_ever = (k == 0) && (kk == 0);
                tcgen05_mma(taddr, a_desc, b_desc, idesc, /*enable_d=*/ !first_ever);
            }
            tcgen05_commit(done_mb);
            tile_ready_phase[slot] ^= 1;
        }

        // One more commit, this time to all_mmas_done, so the epilogue
        // can wait for the final K-iter's MMAs to drain on a known mbar.
        tcgen05_commit((uint32_t)__cvta_generic_to_shared(&all_mmas_done));
    }

    // ── All warps wait for the full K-loop to finish ─────────────────
    mbarrier_wait_phase((uint32_t)__cvta_generic_to_shared(&all_mmas_done), 0);

    // ── Epilogue (identical to ch03) ────────────────────────────────
    //
    // **Each thread owns one entire output row.**
    //
    //     each warp           →  32 rows × all N cols
    //     4 warps × 32 lanes  →  BM=128 rows × BN=256 = whole output tile
    tcgen05_fence_after_thread_sync();

    const int my_row = warp_id * 32 + lane;
    const uint32_t taddr_row_base = taddr + ((uint32_t)(warp_id * 32) << 16);

    #pragma unroll
    for (int n = 0; n < BN; n += 8) {
        float tmp[8];
        tcgen05_ld_32x32b_x8(taddr_row_base + (uint32_t)n, tmp);
        tcgen05_wait_ld();

        __nv_bfloat162 packed[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            packed[i] = __floats2bfloat162_rn(tmp[2 * i], tmp[2 * i + 1]);
        }
        *reinterpret_cast<int4*>(&C_ptr[my_row * BN + n]) =
            *reinterpret_cast<int4*>(packed);
    }

    __syncthreads();
    if (warp_id == 0 && elect_sync()) {
        tcgen05_dealloc(taddr, BN);
    }
}
