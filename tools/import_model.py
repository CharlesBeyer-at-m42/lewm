#!/usr/bin/env python
"""Importer: LeWM model file (GGUF) -> reference checkpoint the reference planner can load.

The inverse of export_model.py. A reference checkpoint pickles whole modules (encoder, predictor,
action_encoder, link, proprio_encoder, epoch), so a template checkpoint supplies the module
objects and the pieces LeWM does not carry (proprio encoder); the LpWM-path weights are then
replaced by the model file's tensors. The result is written as a new reference run directory,
``<reference>/runs/outputs/<run-name>/``, with the template run's ``hydra.yaml`` beside it, which
is exactly what ``plan.py`` expects:

    ../lpworldmodel/.venv/bin/python tools/import_model.py \\
        --model runs/wall_run/model_latest.gguf \\
        --template ../lpworldmodel/runs/outputs/lewm_ref_wall/checkpoints/model_latest.pth \\
        --run-name lewm_wall_import --epoch 5
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

import lewm_ref as L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="LeWM model file (GGUF)")
    ap.add_argument("--template", type=Path, required=True,
                    help="reference model_*.pth whose modules and hydra.yaml are reused")
    ap.add_argument("--run-name", required=True, help="name of the new run under <reference>/runs/outputs/")
    ap.add_argument("--epoch", type=int, default=None, help="epoch recorded in the checkpoint (default: template's)")
    ap.add_argument("--outputs", type=Path, default=L.REFERENCE_ROOT / "runs" / "outputs",
                    help="reference outputs directory (default: ../lpworldmodel/runs/outputs)")
    args = ap.parse_args()

    payload = torch.load(args.template, map_location="cpu", weights_only=False)
    modules = {k: payload[k] for k in L.COMPONENTS}
    for k, m in modules.items():
        if hasattr(m, "module"):
            raise RuntimeError(f"{k} is wrapped ({type(m).__name__}); unwrap before import")

    r, tensors = L.read_gguf_tensors(args.model)
    hp = L.Hparams.from_gguf(r)
    expected = L.hparams_from_modules(modules["encoder"], modules["action_encoder"], modules["predictor"],
                                      payload.get("link"))
    if hp != expected:
        raise RuntimeError(f"model file hyperparameters differ from the template's:\n  file: {hp}\n  template: {expected}")

    consumed = 0
    for prefix, module in modules.items():
        sd = {}
        for name, arr in tensors.items():
            if name.startswith(prefix + "."):
                sd[name[len(prefix) + 1:]] = torch.from_numpy(arr)
        for name, t in module.state_dict().items():
            if name in sd and tuple(sd[name].shape) != tuple(t.shape):
                # the GGUF reader squeezes leading singleton dims (e.g. cls_token (1,1,384) -> (384,))
                if sd[name].numel() != t.numel():
                    raise RuntimeError(f"{prefix}.{name}: shape {tuple(sd[name].shape)} != {tuple(t.shape)}")
                sd[name] = sd[name].reshape(t.shape)
        module.load_state_dict(sd, strict=True)
        consumed += len(sd)
    if consumed != len(tensors):
        raise RuntimeError(f"{len(tensors) - consumed} tensors in the model file were not consumed")

    # Cross-check: the freshly built modules from load_model_file must agree with the imported ones.
    _, enc2, act2, pred2, _ = L.load_model_file(args.model)
    with torch.no_grad():
        for m in modules.values():
            m.eval()
        torch.manual_seed(0)
        img = torch.rand(1, 3, hp.image_size, hp.image_size)
        a = modules["encoder"](img)
        b = enc2(img)
        a = a["visual"] if isinstance(a, dict) else a
        b = b["visual"] if isinstance(b, dict) else b
        diff = (a - b).abs().max().item()
        if diff > 1e-5:
            raise RuntimeError(f"imported encoder disagrees with a fresh build: max abs diff {diff}")

    out_dir = args.outputs / args.run_name
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    template_run = args.template.resolve().parent.parent
    shutil.copyfile(template_run / "hydra.yaml", out_dir / "hydra.yaml")

    payload.update(modules)
    if args.epoch is not None:
        payload["epoch"] = args.epoch
    for k in list(payload):
        if k.startswith("optimizer") or k.endswith("_optimizer") or k.endswith("_scheduler"):
            del payload[k]
    torch.save(payload, out_dir / "checkpoints" / "model_latest.pth")
    torch.save(payload, out_dir / "checkpoints" / f"model_{payload['epoch']}.pth")
    print(f"imported {consumed} tensors from {args.model} (encoder max abs diff vs fresh build {diff:.2e})")
    print(f"wrote {out_dir}/checkpoints/model_latest.pth, epoch {payload['epoch']}")


if __name__ == "__main__":
    main()
