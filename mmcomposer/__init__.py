"""MMComposer -- a CUDA matmul kernel generator + autotuner for Blackwell (B200).

    import mmcomposer as mmc
    c = mmc.matmul(a, b)               # auto-tunes + caches on first sight of a shape
    gemm = mmc.get_tuned_kernel(a, b)  # reusable callable for that shape
    mmc.tune(M, N, K)                  # explicit offline pre-tune

Stage-A package (see mmcomposer/DESIGN.md): the verified core currently lives
under ``webui/`` and is re-exported here so the public API is importable as
``mmcomposer``.  Stage B physically relocates the core into this package for a
standalone ``pip install`` (relative imports + kernels as package data).
"""
import pathlib as _pathlib
import sys as _sys

_WEBUI = _pathlib.Path(__file__).resolve().parent.parent / "webui"
for _p in (_WEBUI, _WEBUI / "kernels"):
    _ps = str(_p)
    if _ps not in _sys.path:
        _sys.path.insert(0, _ps)

from mmc import matmul, get_tuned_kernel, tune          # noqa: E402,F401
import combos, compiler, runtime, benchmark             # noqa: E402,F401
import cache, leaderboard, autotune                     # noqa: E402,F401

__all__ = ["matmul", "get_tuned_kernel", "tune",
           "combos", "compiler", "runtime", "benchmark",
           "cache", "leaderboard", "autotune"]
