#!/usr/bin/env python3
"""autotune.py — terminal autotune.

Sweeps every valid knob combination for a shape on a B200 and prints the top
configs by measured TFLOPS.  Same sweep+rank path as the webui Autotune button
(`live_bench.run_autotune`), just without the live leaderboard — it runs to
completion, then prints the ranking.

Writes a throwaway matrix under tests/_scratch/; the committed
`kernels/compat_matrix.json` is never touched.

Usage (from repo root, on a node with srun + B200 access):
    python webui/autotune.py 8192                # square 8192^3
    python webui/autotune.py 32768x4608x768      # rectangular MxNxK
    python webui/autotune.py 8192 --scope full --top 20

Scopes (mirror the UI radio):
    production (default) — warp-spec-on combos with BN>=128: the smaller,
                           practical search (warp-spec ~always helps, BN<128
                           doesn't for reasonably large N).
    full                 — every combo, incl. warp-spec-off and BN=64.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # webui/

import mvp_core as mc       # noqa: E402
import live_bench as lb     # noqa: E402


def parse_shape(tok: str) -> tuple[int, int, int]:
    """'8192' -> (8192,8192,8192); '32768x4608x768' -> (32768,4608,768)."""
    tok = tok.lower().strip()
    if "x" in tok:
        M, N, K = (int(v) for v in tok.split("x"))
        return M, N, K
    s = int(tok)
    return s, s, s


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Terminal autotune: sweep valid knob combos on a B200, print the top configs.")
    ap.add_argument("shape", help="square 'S' or rectangular 'MxNxK' (e.g. 8192 or 32768x4608x768)")
    ap.add_argument("--scope", choices=["production", "full"], default="production",
                    help="production (default): warp-spec-on, BN>=128; full: everything")
    ap.add_argument("--top", type=int, default=10, help="how many top configs to print (default 10)")
    ap.add_argument("--timeout", type=int, default=3600, help="sweep timeout in seconds (default 3600)")
    args = ap.parse_args()

    M, N, K = parse_shape(args.shape)

    # Mirror the UI's scope -> (tier dirs, BN options).  The two warp-spec arms
    # share one dir (TWO_CTA distinguishes them); the sweep expands each dir to
    # all its arms, so each dir is passed once.
    ws_dirs  = list(dict.fromkeys(t["dir"] for k, t in mc.TIER_MAP.items() if t and k[0]))
    all_dirs = list(dict.fromkeys(t["dir"] for t in mc.TIER_MAP.values() if t))
    production = args.scope == "production"
    tier_dirs = ws_dirs if production else all_dirs
    bn_opts   = [128, 256] if production else None

    print(f"# autotune {M}x{N}x{K}  scope={args.scope}  "
          f"(running the full sweep on a B200 via srun — this can take a while)", flush=True)

    res = lb.run_autotune(tier_dirs, M, N, K, bn_opts=bn_opts, timeout=args.timeout)

    if not res.get("ok"):
        print(f"\nFAILED: {res.get('error')}")
        if res.get("stderr"):
            print("--- driver stderr (tail) ---")
            print(res["stderr"])
        return 1

    cub = res.get("cublas_tflops")
    rows = res["results"][:args.top]
    print(f"\ncuBLAS reference: {cub:.0f} TFLOPS" if cub else "\ncuBLAS reference: n/a")
    print(f"Top {len(rows)} of {res['n_combos']} valid combos at {M}x{N}x{K}, by TFLOPS:\n")

    hdr = (f"{'#':>2}  {'TFLOPS':>7}  {'%cuBLAS':>7}  {'WS':>3} {'2CTA':>4}  "
           f"{'BN':>3} {'NS':>2} {'GSM':>3} {'NW':>2}  {'PERS':>4} "
           f"{'LDW':>3} {'OV':>2} {'SPLIT':>5} {'L1NA':>4}")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows, 1):
        ws  = "on" if mc.toggles_for_dir(r["tier"])[0] else "off"
        cta = "on" if r.get("two_cta") else "off"
        vsc = f"{r['vs_cublas'] * 100:.0f}%" if r.get("vs_cublas") else "-"
        print(f"{i:>2}  {r['tflops']:>7.0f}  {vsc:>7}  {ws:>3} {cta:>4}  "
              f"{r['bn']:>3} {r['ns']:>2} {r['gsm']:>3} {r['nw']:>2}  "
              f"{r['persistent']:>4} {r.get('ld_width', 8):>3} "
              f"{r.get('overlap', 0):>2} {r.get('split_epilogue', 0):>5} {r.get('l1_no_alloc', 0):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
