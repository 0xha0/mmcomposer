# GEMM on B200 Presentation

This directory contains a Beamer deck for:

```text
How to Design a High Performance GEMM Kernel (on B200)?
```

Build:

```bash
make -C presentation
```

Or, from inside this directory:

```bash
make
```

The default build uses Tectonic with shell escape enabled so slides can use
`minted` for syntax-highlighted code blocks. The Python-side dependency for
`minted` is Pygments:

```bash
conda install -y --override-channels -c conda-forge tectonic pygments
```

For a full TeX Live environment, `latexmk` is also supported:

```bash
make latexmk
```

## Format Choice

Beamer is a good first choice for this talk because the material needs math,
code snippets, pipeline diagrams, and tables. It also produces a portable PDF
that is easy to share.

HTML/reveal.js would be better if we later want interactive animations or live
UI demos inside the deck. For now, Beamer plus SVG/PNG figures is the simpler
and more reproducible path.

## Planned Story

1. GPU and B200 execution background.
2. Major GEMM optimizations that get close to cuBLAS:
   tile size and arithmetic intensity, warp specialization with TMA loads,
   `tcgen05.mma`, 2-CTA MMA, pipelined TMA stores, persistent grid and
   epilogue overlap.
3. The resulting general GEMM framework.
4. MMComposer: UI, CLI, correctness sweeps, and recorded autotune runs.
