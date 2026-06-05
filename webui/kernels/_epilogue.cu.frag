    // ── Shared epilogue (TMEM → SMEM → GMEM) ────────────────────────
    // mvp_core stitches this fragment into every tier's kernel.cu at the
    // epilogue marker, so the epilogue lives in exactly one place.  Each
    // tier supplies a small contract just before the marker:
    //   cta_rank, off_m_cluster, off_n   — tile-origin primitives
    //       (single-CTA tiers set cta_rank=0, off_m_cluster=off_m)
    //   EPI_DEALLOC(taddr, n)            — that tier's tcgen05 dealloc
    //       (single-CTA: tcgen05_dealloc; cluster: tcgen05_dealloc_g2)
    constexpr int EPI_LD = BN + 8;   // C_sh leading dim (+8 = bank-conflict pad)
    auto C_sh = reinterpret_cast<__nv_bfloat16(*)[EPI_LD]>(smem);

    tcgen05_fence_after_thread_sync();

    // ── Phase 1: TMEM → SMEM, generalized variable-warp 2D grid ──────
    // Partition NUM_WARPS as ROW_STRIPS (BM/32) row groups × COL_GROUPS
    // column slices so every warp works even at NW=8/16.  The cluster
    // adds a cta_rank*BM logical-row offset into this CTA's TMEM.
    constexpr int ROW_STRIPS    = BM / 32;
    constexpr int COL_GROUPS    = NUM_WARPS / ROW_STRIPS;
    constexpr int COLS_PER_WARP = BN / COL_GROUPS;

    const int row_warp = warp_id % ROW_STRIPS;
    const int col_warp = warp_id / ROW_STRIPS;
    const int my_row   = row_warp * 32 + lane;
    const uint32_t taddr_row =
        taddr + ((uint32_t)(cta_rank * BM + row_warp * 32) << 16);
    const int col_base = col_warp * COLS_PER_WARP;

    #pragma unroll
    for (int n = col_base; n < col_base + COLS_PER_WARP; n += 8) {
        float tmp[8];
        tcgen05_ld_32x32b_x8(taddr_row + (uint32_t)n, tmp);
        tcgen05_wait_ld();

        __nv_bfloat162 packed[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            packed[i] = __floats2bfloat162_rn(tmp[2 * i], tmp[2 * i + 1]);
        }
        // SMEM store — int4 = 16 B = 8 BF16, one chunk of this row.
        *reinterpret_cast<int4*>(&C_sh[my_row][n]) =
            *reinterpret_cast<int4*>(packed);
    }

    __syncthreads();
    if (warp_id == 0 && elect_sync()) {
        EPI_DEALLOC(taddr, BN);
    }

    // ── Phase 2: SMEM → GMEM, flat thread-major coalesced int4 stores ──
    // Consecutive lanes write consecutive 16-byte int4 chunks to GMEM
    // → one coalesced transaction per warp per store.
    constexpr int CHUNK_BF16        = 8;
    constexpr int CHUNKS_PER_ROW    = BN / CHUNK_BF16;
    constexpr int STORES_PER_THREAD = (BM * BN) / (THREADS * CHUNK_BF16);
    static_assert(STORES_PER_THREAD * THREADS * CHUNK_BF16 == BM * BN,
                  "BM*BN must be a multiple of THREADS*8 for the flat tile-walk");
    const int out_m_base = off_m_cluster + cta_rank * BM;

    #pragma unroll
    for (int s = 0; s < STORES_PER_THREAD; s++) {
        const int flat = tid + s * THREADS;
        const int row  = flat / CHUNKS_PER_ROW;
        const int col  = (flat % CHUNKS_PER_ROW) * CHUNK_BF16;
        const int gr   = out_m_base + row;
        const int gc   = off_n + col;
        *reinterpret_cast<int4*>(&C_ptr[gr * N + gc]) =
            *reinterpret_cast<const int4*>(&C_sh[row][col]);
    }
