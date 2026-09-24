#!/usr/bin/env python
"""Exporter: reference checkpoint, or seeded random init -> LeWM model file (GGUF).

Examples (from lewm/):
  ../lpworldmodel/.venv/bin/python tools/export_model.py \\
      --checkpoint ../lpworldmodel/runs/outputs/lewm_ref_wall/checkpoints/model_latest.pth \\
      --out testdata/wall_ref.gguf

  ../lpworldmodel/.venv/bin/python tools/export_model.py --random --seed 0 \\
      --config ../lpworldmodel/runs/outputs/lewm_ref_wall/hydra.yaml --action-dim 10 \\
      --out testdata/random_seed0.gguf

Tensor names mirror the reference parameter names, prefixed by component. The proprio encoder
is skipped (see docs/adr/0002). fp32 only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import lewm_ref as L


def modules_from_config(config: Path, action_dim: int, seed: int, mup_init: bool = False):
    """Random init exactly as train.py would build the modules, minus muP re-init.

    muP init only changes the *distribution* of random weights; parity does not care which
    random weights it tests, so plain module init is used for determinism and simplicity.
    """
    import hydra
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config)
    torch.manual_seed(seed)
    encoder = hydra.utils.instantiate(cfg.encoder)
    action_encoder = hydra.utils.instantiate(
        cfg.action_encoder, in_chans=action_dim, emb_dim=cfg.action_emb_dim
    )
    predictor = hydra.utils.instantiate(
        cfg.predictor,
        num_frames=cfg.num_hist,
        num_patches=encoder.num_patches,
        input_dim=encoder.emb_dim,
        hidden_dim=encoder.emb_dim,
        output_dim=encoder.emb_dim,
    )
    link = hydra.utils.instantiate(cfg.link) if cfg.get("link") else None
    if mup_init:
        # exactly what train.py does for a fresh muP run: encoder and predictor only
        from models.mup import mup_init_
        mup_init_(encoder, tag="encoder", verbose=False)
        mup_init_(predictor, tag="predictor", verbose=False)
        return encoder, action_encoder, predictor, link
    # AdaLN-zero and cls_token are zero-initialised by the reference; for a parity test that is
    # a weak signal (gates multiply everything by 0), so perturb them like a few training steps.
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed + 1)
        for name, p in predictor.named_parameters():
            if "adaLN_modulation" in name:
                p.add_(torch.randn(p.shape, generator=g) * 0.05)
        encoder.cls_token.add_(torch.randn(encoder.cls_token.shape, generator=g) * 0.02)
    return encoder, action_encoder, predictor, link


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="reference model_*.pth (pickled modules)")
    src.add_argument("--random", action="store_true", help="seeded random init from --config")
    ap.add_argument("--config", type=Path, help="resolved hydra.yaml of a run (for --random)")
    ap.add_argument("--action-dim", type=int, default=10, help="chunked action width (for --random)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mup-base-width", type=int, default=0, help="muP base width for lr multipliers (default: encoder dim)")
    ap.add_argument("--mup-init", action="store_true", help="with --random: apply the reference's muP init (faithful fresh run), no perturbation")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.checkpoint:
        encoder, action_encoder, predictor, link, epoch = L.load_checkpoint_modules(args.checkpoint)
        source = f"checkpoint:{args.checkpoint.resolve()}:epoch={epoch}"
    else:
        if not args.config:
            ap.error("--random requires --config")
        encoder, action_encoder, predictor, link = modules_from_config(args.config, args.action_dim, args.seed, args.mup_init)
        source = f"random:seed={args.seed}:mup_init={args.mup_init}:config={args.config.resolve()}"

    for m in (encoder, action_encoder, predictor):
        m.eval().cpu()
    hp = L.hparams_from_modules(encoder, action_encoder, predictor, link)

    tensors = list(L.state_tensors(encoder, action_encoder, predictor))
    n_params = sum(int(a.size) for _, a in tensors)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"lewm.source": source}
    meta.update(L.mup_metadata(encoder, action_encoder, predictor, base_width=args.mup_base_width or hp.dim))
    L.write_gguf(args.out, hp, tensors, meta)

    print(f"wrote {args.out}  tensors={len(tensors)}  params={n_params:,}")
    for k, v in vars(hp).items():
        print(f"  {k:26s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
