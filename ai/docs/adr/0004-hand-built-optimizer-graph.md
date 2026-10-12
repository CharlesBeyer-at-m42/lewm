---
status: accepted
date: 2026-09-23
---

# LeWM builds its own training graph instead of using ggml-opt

ggml ships a high-level optimizer context (`ggml-opt.h`). LeWM does not use it. The forward graph,
`ggml_build_backward_expand`, and one `ggml_opt_step_adamw` per parameter are assembled by LeWM,
which is roughly what ggml-opt does internally.

Two reasons, both structural. First, muP needs a learning rate per parameter tensor (base LR times
base width over fan-in), and ggml-opt exposes a single global learning rate; the low-level step op
takes a parameter tensor per call, so a hand-built graph can supply per-tensor values. Second,
ggml-opt's dataset and graph abstractions assume one flat input tensor of shape
[datapoint, batch] and one output compared against labels; LpWM training has image sequences,
chunked actions, host-generated randomness for RDMReg and dropout, and a loss that is not a
function of an output-versus-label pair.

Consequence: the training loop, checkpoint sidecar (AdamW moments and step), and gradient
accumulation are LeWM code and must be tested by LeWM, hence the gradient-parity gate.
