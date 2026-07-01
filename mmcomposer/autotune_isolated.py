"""Process-isolated autotune.

This is the fault-contained sibling of :mod:`mmcomposer.autotune`: codegen and
nvcc still run in the parent, but each candidate's CUDA verify+benchmark step
runs in a fresh Python child.  A launch failure can poison the child CUDA
context, but the parent continues and records the configs that survived.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time

from . import autotune
from . import benchmark as bench
from . import cache as kcache
from . import combos
from . import compiler
from . import runtime


def _torch_ref_from_cuda_expr(expr: str, x, aux):
    """Evaluate the internal EDL CUDA expression against torch tensors.

    ``expr`` comes from ``epilogue.to_cuda`` for code we generated ourselves.
    This restricted eval exists so child workers do not need to pickle the
    original Python callable.
    """
    import torch

    def _tensor(v):
        if torch.is_tensor(v):
            return v
        return torch.as_tensor(v, dtype=x.dtype, device=x.device)

    def _max(a, b):
        return torch.maximum(_tensor(a), _tensor(b))

    def _min(a, b):
        return torch.minimum(_tensor(a), _tensor(b))

    pyexpr = re.sub(r"(?<=\d)f\b", "", expr)
    env = {
        "__builtins__": {},
        "x": x,
        "__fdividef": lambda a, b: a / b,
        "__expf": torch.exp,
        "tanhf": torch.tanh,
        "sqrtf": torch.sqrt,
        "__logf": torch.log,
        "fabsf": torch.abs,
        "fmaxf": _max,
        "fminf": _min,
        "powf": lambda a, b: a ** b,
    }
    for i, t in enumerate(aux):
        env[f"c{i}"] = t
    return eval(pyexpr, env, {})  # noqa: S307 - internal generated expression


def _child_main(payload_path: str) -> int:
    import torch

    payload = json.loads(pathlib.Path(payload_path).read_text())
    M, N, K = payload["shape"]
    config = payload["config"]
    n_extra = payload["n_extra"]
    epilogue = payload.get("epilogue")
    tol = payload["tol"]
    bench_kw = payload.get("bench_kw") or {}

    torch.manual_seed(payload.get("seed", 1234))
    torch.cuda.manual_seed_all(payload.get("seed", 1234))

    result = {
        "status": None,
        "rel_err": None,
        "tflops": None,
        "latency_us": None,
        "error_type": None,
        "error": None,
    }
    try:
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        aux = [torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
               for _ in range(n_extra)]
        ref = a.float() @ b.float()
        if epilogue is not None:
            ref = _torch_ref_from_cuda_expr(epilogue, ref, [t.float() for t in aux])

        c = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        gemm = runtime.kernel(config, payload["cubin"])
        gemm(a, b, c, aux=aux, sync=True)
        rel = bench.rel_error(c, ref)
        result["rel_err"] = rel
        if rel >= tol:
            result["status"] = "wrong"
        else:
            r = bench.benchmark(lambda: gemm(a, b, c, aux=aux, sync=False),
                                flops=bench.gemm_flops(M, N, K), **bench_kw)
            result.update(status="ok", tflops=r.tflops, latency_us=r.latency_us)
    except Exception as e:  # noqa: BLE001
        result.update(status="fail", error_type=type(e).__name__, error=str(e))

    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _parse_child_result(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return {"status": "fail", "error_type": "NoChildJson", "error": stdout[-2000:]}


def _run_child(payload: dict, timeout_s: int) -> dict:
    fd, path = tempfile.mkstemp(prefix="mmcomposer-isolated-", suffix=".json")
    os.close(fd)
    p = pathlib.Path(path)
    p.write_text(json.dumps(payload))
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "mmcomposer.autotune_isolated", "--child", path],
            cwd=str(pathlib.Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        result = _parse_child_result(proc.stdout)
        result["returncode"] = proc.returncode
        if proc.stderr:
            result["stderr_tail"] = proc.stderr[-2000:]
        if proc.returncode != 0 and result.get("status") == "ok":
            result.update(status="fail", error_type="ChildReturnCode",
                          error=proc.stderr[-2000:])
        return result
    except subprocess.TimeoutExpired as e:
        return {"status": "fail", "error_type": "TimeoutExpired", "error": str(e),
                "returncode": None}
    finally:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def _smem_filtered_combos(M, N, K, tier_dirs, filters):
    combo_list = [(tier, k) for (tier, k) in combos.valid_combos(tier_dirs, filters)
                  if autotune._fits(tier, k, M, N, K)]
    _, num_sms = runtime._ensure_cuda()
    rt, driver = runtime._backends()
    max_smem = rt.cu(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        runtime._device))

    def ok(tier, k):
        cfg = runtime.config_from_combo(tier, k)
        _, _, shared = runtime.launch_dims(cfg, M, N, K, num_sms=num_sms)
        return shared <= max_smem

    return [(tier, k) for tier, k in combo_list if ok(tier, k)]


def tune(M, N, K, *, tier_dirs, filters, dtype="bf16", arch=kcache.DEFAULT_ARCH,
         tol=autotune.CORRECT_TOL, warmup_ms=None, rep_ms=None,
         cublas_samples=3, cublas_warmup_samples=1, fresh=True,
         cache_obj=None, on_event=None, epilogue=None, epi_tag=None,
         ref_fn=None, n_extra=0, child_timeout_s=600) -> dict:
    """Run an isolated timing sweep.

    Signature intentionally mirrors :func:`mmcomposer.autotune.tune`.  ``ref_fn``
    is accepted for API compatibility; child workers evaluate ``epilogue`` from
    the generated CUDA expression instead.
    """
    del ref_fn
    import torch

    kc = cache_obj if cache_obj is not None else kcache.Cache()
    key = kcache.shape_key(M, N, K, dtype, arch, epi=epi_tag)
    if fresh:
        kc.clear(key)

    def emit(phase, **kw):
        if on_event:
            on_event(key=key, phase=phase, **kw)

    combo_list = _smem_filtered_combos(M, N, K, tier_dirs, filters)
    n_valid = len(combo_list)
    emit("enumerate", n_valid=n_valid)

    build_root = kcache.cache_root() / "build" / arch
    builds = [(tier, k, autotune._render(tier, k, build_root,
                                         epilogue=epilogue, n_extra=n_extra))
              for tier, k in combo_list]
    emit("compile", done=0, total=n_valid)
    comp = compiler.compile_many([src for _, _, src in builds])
    ok = [(tier, k, src) for tier, k, src in builds if comp[src].ok]
    n_compiled = len(ok)
    emit("compiled", n_compiled=n_compiled, n_valid=n_valid)

    bench_kw = {}
    if warmup_ms is not None:
        bench_kw["warmup_ms"] = warmup_ms
    if rep_ms is not None:
        bench_kw["rep_ms"] = rep_ms

    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    flops = bench.gemm_flops(M, N, K)
    cub = bench.benchmark_median(lambda: torch.mm(a, b), flops=flops,
                                 samples=cublas_samples,
                                 warmup_samples=cublas_warmup_samples,
                                 **bench_kw).tflops
    del a, b
    torch.cuda.empty_cache()
    emit("cublas", cublas_tflops=cub, total=n_compiled)

    n_correct = 0
    for i, (tier, k, src) in enumerate(ok, 1):
        payload = {
            "shape": [M, N, K],
            "config": autotune._record_config(tier, k),
            "cubin": comp[src].cubin,
            "epilogue": epilogue,
            "n_extra": n_extra,
            "tol": tol,
            "bench_kw": bench_kw,
            "seed": 1234,
        }
        t0 = time.monotonic()
        result = _run_child(payload, child_timeout_s)
        if result.get("status") == "ok":
            rec = {"config": autotune._record_config(tier, k),
                   "tflops": result["tflops"],
                   "vs_cublas": (result["tflops"] / cub) if cub else None,
                   "rel_err": result["rel_err"]}
            kc.put(key, rec)
            n_correct += 1
        emit("benchmark", done=i, total=n_compiled, cublas_tflops=cub,
             child_status=result.get("status"), elapsed_s=time.monotonic() - t0)

    best = kc.best(key)
    return {"ok": best is not None, "key": key, "cublas_tflops": cub, "best": best,
            "n_valid": n_valid, "n_compiled": n_compiled, "n_correct": n_correct,
            "error": None if best else "no correct combos measured for this shape"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return _child_main(args.child)
    ap.error("autotune_isolated is currently a library module; use mmc.tune(..., isolated=True)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
