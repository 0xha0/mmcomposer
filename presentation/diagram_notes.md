# Container Diagram Notes

## Representation

- Use different shapes for operations and containers.
- Containers are drawn as circles.
- Operations are drawn as squares or rectangles.
- A directed edge from a container to an operation means the operation drains
  that input container.
- A directed edge from an operation to a container means the operation fills
  that output container.

## Operation Start Rule

An operation can start only when both conditions are satisfied:

1. The input data is ready.
2. The output container is free.

## Signal Types

There are two synchronization signals:

1. `value-ready`: a container is full, so its value can be consumed.
2. `resource-free`: a container is empty, so it can be reused.

For TMEM, `value-ready` has a stronger condition: all contributing MMAs must be
done. A single MMA finishing is not enough to make the final TMEM tile ready for
the epilogue.

## Main Synchronization Pattern

Start with an ASCII overview before the per-warp breakdown:

```text
  +------+    +-----------+    +---------------------+
  | HBM  | -> | TMA warps | -> | SMEM compute buffer |
  +------+    +-----------+    +---------------------+
                  |  SMEM full               ^
                  v                           | SMEM free
  +------+    +-----------+    +------+
  | SMEM | -> | MMA warps | -> | TMEM |
  +------+    +-----------+    +------+
                  |  TMEM full*              ^
                  |                           | TMEM free
                  v                           |
  +------+    +----------------+    +-----------+
  | TMEM | -> | epilogue warps | -> | registers |
  +------+    +----------------+    +-----------+
```

Use `full` for a value-ready signal and `free` for a resource-free signal.
The `TMEM full` signal is sent only after the final MMA finishes.

## TMA Warp Diagram

Dataflow:

```text
HBM container -> TMA operation -> SMEM compute ring-buffer container
```

Slides/overlays:

1. HBM container is full. SMEM compute buffer container is empty.
2. HBM container is half full. SMEM compute buffer container is half full.
   This shows data flowing from HBM into the compute buffer: the input is being
   drained and the output is being computed.
3. HBM container is empty. SMEM compute buffer container is full.
4. Zoom in on the output SMEM compute buffer. It sends one outward signal to
   the MMA warp: `value-ready`.

For this TMA slide sequence, indicate that the SMEM output will be consumed by
the MMA warp, but do not draw the full GEMM graph.

## MMA Warp Diagram

Dataflow:

```text
SMEM compute ring-buffer container -> MMA operation -> TMEM container
```

Slides/overlays:

1. SMEM compute buffer is full. TMEM output container is empty.
2. SMEM compute buffer is half full. TMEM output container is half full.
3. SMEM compute buffer is empty. TMEM output container is full.
4. Zoom in on the final state:
   - MMA warps send `all-MMAs-done`, which makes the TMEM tile ready for the
     epilogue warps.
   - The empty SMEM compute buffer sends `resource-free` back to the TMA warp.

## Epilogue Warp Diagram

Dataflow:

```text
TMEM container -> epilogue operation -> register container
```

Slides/overlays:

1. TMEM is full, but the epilogue starts only when the `all-MMAs-done` signal
   has arrived and the register output container is free.
2. TMEM is half drained. Registers are half full.
3. TMEM is empty. Registers are full.
4. Zoom in on the final state:
   - The empty TMEM container sends `resource-free` back to the MMA warps.

Do not show SMEM staging in this epilogue diagram. It is intentionally omitted
to keep the orchestration focus on draining TMEM into registers.
