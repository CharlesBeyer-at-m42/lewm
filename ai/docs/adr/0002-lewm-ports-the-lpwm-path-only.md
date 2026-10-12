---
status: accepted
date: 2026-09-23
---

# LeWM ports the LpWM path only; baseline, ladder, proprio, and environments stay in Python

The reference implementation contains two world models and several ablation variants. LeWM implements only the LpWM path: from-scratch vision encoder, action encoder, autoregressive adaptive-layer-norm predictor, and link. Both CLS and patch encoder modes are supported from the start because they differ only in a mask and a positional embedding, and retrofitting the flattening convention later is error-prone.

Explicitly out of scope, so nobody "fixes" their absence:

- **The DINO-WM baseline** (frozen DINOv2 encoder, concatenation predictor, pixel decoder). It exists for comparison in the paper. DINOv2 in ggml is a separate project. Run baseline numbers in PyTorch.
- **The predictor ladder** of linear-dynamics variants. Ablation material. The predictor sits behind an interface so a ladder variant can be added later if needed.
- **The proprio encoder.** Instantiated and checkpointed by the reference implementation but never called on the LpWM path. The exporter skips it.
- **The environments.** Wall is small tensor math, but PushT depends on pymunk physics and pygame rendering. Planning and evaluation against environments stay in Python.

Consequence: the first milestone is forward-pass inference verified by a parity test against a reference dump; training on ggml is the target after that.
