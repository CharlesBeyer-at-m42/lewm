---
status: accepted
date: 2026-09-23
---

# Fork ggml on a LeWM branch to add the backward ops training needs

ggml v0.25.0 has no backward pass for LayerNorm (`GGML_OP_NORM`), for GELU (either form), or for
concat, and no straight-through activation. LpWM training needs all of them: about 60 LayerNorms,
GELU in every feed-forward and projector, and the reprelu link whose forward is ReLU but whose
gradient is GELU's. We carry these ops on a `lewm` branch of the ggml submodule, with CPU and CUDA
kernels. The branch stays local: it is not hosted and the ops are not upstreamed (decided 2026-10-06).

## Considered options

- **Compose from primitives with backward** (mean, sub, sqr, sqrt, div for LayerNorm; exp-based
  tanh for GELU). Rejected: ggml has no erf primitive, so GELU would change from exact to the tanh
  approximation and break the inference parity that milestone 1 established. It also leaves the
  straight-through link unsolved and costs several nodes per op.
- **CPU-only custom ops** via `ggml_map_custom`. Rejected: contradicts the CUDA decision; the
  4070 is the only thing that trains 15 GB of frames in reasonable time.

## Consequences

The submodule points at a local-only branch. The only copies of those commits live in this
checkout's `lewm/ggml`, so a backup of the workspace is the backup of the fork, and a fresh clone of
`lewm` needs that directory copied in rather than `git submodule update`. Concat is avoided rather than extended: the cls token is placed with
`ggml_set`/`ggml_acc`, which have backward passes.
