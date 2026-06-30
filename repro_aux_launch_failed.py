"""Minimal reproducer for the epilogue-sweep launch failure.

See linear/epilogue-aux-status.md.  Background: certain production configs fail to
launch (CUDA_ERROR_INVALID_VALUE, or CUDA_ERROR_LAUNCH_FAILED which poisons the
context).  The failure is NON-DETERMINISTIC -- the same config has been observed to
pass, INVALID_VALUE, or LAUNCH_FAILED across runs -- which is why the epilogue
tuned-variant sweep crashes intermittently.

This script builds a few suspect configs directly (no autotune sweep) and launches
both the PLAIN GEMM and the (a@b)*c epilogue synced, reporting each.  NOTE: it was
initially believed plain matmul was immune (one clean 198/198 sweep), but isolated
runs have shown plain failing on these configs too -- so this is most likely a
non-deterministic launch-flakiness in the configs themselves, not purely the
extra-input path.  Run a few times; outcomes vary.

    srunpy repro_aux_launch_failed.py
"""
import torch
from mmcomposer import autotune, mvp_core as mc, epilogue as epi
import mmcomposer.mmc as mmc

M, N, K = 32768, 4608, 768
tier = next(t for t in mc.TIER_MAP.values() if t and t["dir"] == "tier3_cluster_swizzle")
a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
c = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
cuda = epi.to_cuda(lambda x, cc: x * cc)

# production configs observed to crash with the extra-input epilogue
BAD = [dict(ns=5, nw=4, gsm=8, tss=1), dict(ns=6, nw=4, gsm=1, tss=1),
       dict(ns=6, nw=8, gsm=1, tss=1), dict(ns=4, nw=4, gsm=1, tss=1),
       dict(ns=6, nw=4, gsm=2, tss=1), dict(ns=6, nw=8, gsm=2, tss=1)]


def cfg_of(d):
    k = dict(bm=128, bn=256, bk=64, ns=d["ns"], gsm=d["gsm"], nw=d["nw"], two_cta=1,
             persistent=1, overlap=1, tma_pipelined=1, tma_store_stages=d["tss"],
             single_tmem=0, split_epilogue=0, l1_no_alloc=0, ld_width=8)
    return autotune._record_config(tier, k), f"ns{d['ns']} nw{d['nw']} gsm{d['gsm']} tss{d['tss']}"


for d in BAD:
    cfg, tag = cfg_of(d)
    try:
        mmc._build(cfg)(a, b, None, sync=True); torch.cuda.synchronize()     # plain: fine
        plain = "plain OK"
    except Exception as e:
        plain = f"plain {type(e).__name__}"
    try:
        mmc._build_epilogue(cfg, cuda, 1)(a, b, None, aux=[c], sync=True); torch.cuda.synchronize()
        print(f"{tag}: {plain} | (a@b)*c OK", flush=True)
    except Exception as e:
        print(f"{tag}: {plain} | (a@b)*c *** {type(e).__name__}: {str(e)[:60]}", flush=True)
        print("\nREPRODUCED: a config failed to launch (see plain vs (a@b)*c above). "
              "A LAUNCH_FAILED poisons the context, which is what kills the sweep.")
        break
else:
    print("\n(no crash this run -- the failure is non-deterministic; re-run or add configs)")
