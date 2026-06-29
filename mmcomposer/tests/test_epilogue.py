#!/usr/bin/env python3
"""Tests for the epilogue DSL (pure -- no GPU): tracing + lowering to CUDA."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))  # repo root

import pytest

from mmcomposer import epilogue as epi
from mmcomposer.epilogue import sigmoid, relu, exp, tanh, sqrt, maximum, minimum


def test_identity():
    assert epi.to_cuda(lambda x: x) == "x"


def test_arithmetic_and_constants():
    assert epi.to_cuda(lambda x: x * 2.0) == "(x * 2.0f)"
    assert epi.to_cuda(lambda x: 2 * x) == "(2.0f * x)"
    assert epi.to_cuda(lambda x: x + 1) == "(x + 1.0f)"
    assert epi.to_cuda(lambda x: 1 - x) == "(1.0f - x)"
    assert epi.to_cuda(lambda x: -x) == "(-x)"
    assert epi.to_cuda(lambda x: 1.0 / x) == "(1.0f / x)"


def test_primitives_map_to_intrinsics():
    assert epi.to_cuda(lambda x: exp(x)) == "__expf(x)"
    assert epi.to_cuda(lambda x: tanh(x)) == "tanhf(x)"
    assert epi.to_cuda(lambda x: sqrt(x)) == "sqrtf(x)"
    assert epi.to_cuda(lambda x: abs(x)) == "fabsf(x)"
    assert epi.to_cuda(lambda x: maximum(x, 0.0)) == "fmaxf(x, 0.0f)"
    assert epi.to_cuda(lambda x: minimum(x, 6.0)) == "fminf(x, 6.0f)"


def test_composites_expand_to_primitives():
    # sigmoid(x) = 1/(1+exp(-x))
    assert epi.to_cuda(sigmoid) == "(1.0f / (1.0f + __expf((-x))))"
    # relu(x) = maximum(x, 0)
    assert epi.to_cuda(relu) == "fmaxf(x, 0.0f)"


def test_silu_and_def_form():
    silu_lambda = lambda x: x * sigmoid(x)            # noqa: E731
    expected = "(x * (1.0f / (1.0f + __expf((-x)))))"
    assert epi.to_cuda(silu_lambda) == expected

    def silu(x):                                       # def works too
        return x * sigmoid(x)
    assert epi.to_cuda(silu) == expected


def test_pow_small_int_expands():
    assert epi.to_cuda(lambda x: x ** 2) == "(x * x)"
    assert epi.to_cuda(lambda x: x ** 3) == "(x * x * x)"
    assert epi.to_cuda(lambda x: x ** 0) == "1.0f"


def test_relu6_compose():
    assert epi.to_cuda(lambda x: minimum(maximum(x, 0.0), 6.0)) == "fminf(fmaxf(x, 0.0f), 6.0f)"


def test_digest_stable_and_distinct():
    assert epi.digest(lambda x: x * sigmoid(x)) == epi.digest(lambda x: x * sigmoid(x))
    assert epi.digest(relu) != epi.digest(sigmoid)


def test_rejects_control_flow_and_bad_returns():
    with pytest.raises(TypeError):
        epi.to_cuda(lambda x: x if x else 0)          # bool() on Expr -> blocked
    with pytest.raises(TypeError):
        epi.to_cuda(lambda x: (x, x))                 # must return one value
    with pytest.raises(TypeError):
        epi.to_cuda(lambda x: x ** x)                 # non-constant exponent


def test_fused_epilogue_matches_torch_gpu():
    """GPU: a tight pretune, then matmul(epilogue=...) vs torch; identity is exact."""
    import os
    import tempfile
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    import torch.nn.functional as F
    from mmcomposer import autotune, mvp_core as mc
    import mmcomposer.mmc as mmc

    old = os.environ.get("MMCOMPOSER_CACHE_DIR")
    os.environ["MMCOMPOSER_CACHE_DIR"] = tempfile.mkdtemp(prefix="mmc_epi_test_")
    try:
        M = N = K = 512
        ws = list(dict.fromkeys(t["dir"] for k, t in mc.TIER_MAP.items() if t and k[0]))
        tight = {"bn": [256], "ns": [4], "gsm": [8], "nw": [8], "two_cta": [1],
                 "persistent": [1], "overlap": [1], "split_epilogue": [0],
                 "l1_no_alloc": [0], "tma_pipelined": [1], "tma_store_stages": [2],
                 "single_tmem": [0]}
        s = autotune.tune(M, N, K, tier_dirs=ws, filters=tight,
                          cublas_samples=1, cublas_warmup_samples=0)
        assert s["ok"], "pre-tune failed"

        torch.manual_seed(0)
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        base = a.float() @ b.float()

        # identity epilogue must equal a plain matmul, bit for bit
        ci = mmc.matmul(a, b, epilogue=lambda x: x)
        cp = mmc.matmul(a, b)
        assert (ci.float() - cp.float()).abs().max().item() == 0.0

        def rel(got, ref):
            return ((got.float() - ref).norm() / ref.norm()).item()

        c = mmc.matmul(a, b, epilogue=lambda x: x * sigmoid(x))
        assert rel(c, F.silu(base)) < 5e-2

        c2 = mmc.matmul(a, b, epilogue=relu)
        assert rel(c2, F.relu(base)) < 5e-2
    finally:
        if old is None:
            os.environ.pop("MMCOMPOSER_CACHE_DIR", None)
        else:
            os.environ["MMCOMPOSER_CACHE_DIR"] = old


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
