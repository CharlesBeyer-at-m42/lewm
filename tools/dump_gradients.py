#!/usr/bin/env python
"""Gradient dump: one training batch through the reference implementation -> GGUF.

Rebuilds the reference modules from a LeWM model file, forms one batch exactly as the reference
data pipeline would (same split, same slice order), runs the LpWM training forward with dropout
disabled, records the randomness RDMReg consumed, backpropagates, and takes one muP-AdamW step.

Everything LeWM needs to reproduce the step is in the dump:
  in.visual              (B*T, 3, H, W)    normalised frames, T = num_hist + 1, frame-major per sample
  in.actions             (B, T, A)         chunked, normalised actions
  rdm.proj               (K, C)            random projections (rows unit-norm), K = num_projections
  rdm.target_sorted      (TP, K, B)        link(GN_p) projected and sorted along the sample axis B
  fwd.z                  (B, T, P, C)      codes
  fwd.z_pred             (B, num_hist, P, C)
  loss.z / loss.reg / loss.total           scalars
  grad.<component>.<param>                 gradient of loss.total
  post.<component>.<param>                 parameter after one AdamW step
Metadata `lewm.gradcheck.*` records batch size, step index, learning rate, weight decay, betas, eps.

Example (from lewm/):
  ../lpworldmodel/.venv/bin/python tools/dump_gradients.py --model testdata/random_seed0.gguf \\
      --data-dir ../data/bmw48/wall_single --out testdata/random_seed0.grad.gguf --batch-size 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

import lewm_ref as L


def disable_dropout(*modules):
    for m in modules:
        for sub in m.modules():
            if isinstance(sub, torch.nn.Dropout):
                sub.p = 0.0
            if hasattr(sub, "dropout") and isinstance(getattr(sub, "dropout"), float):
                sub.dropout = 0.0  # Attention.dropout feeds scaled_dot_product_attention


class RandomnessRecorder:
    """Records the tensors torch.randn and Laplace.sample produce while active."""

    def __init__(self):
        self.randn, self.laplace = [], []

    def __enter__(self):
        self._randn = torch.randn
        self._sample = torch.distributions.Laplace.sample

        def randn(*a, **k):
            t = self._randn(*a, **k)
            self.randn.append(t.detach().clone())
            return t

        def sample(dist, shape=torch.Size()):
            t = self._sample(dist, shape)
            self.laplace.append(t.detach().clone())
            return t

        torch.randn = randn
        torch.distributions.Laplace.sample = sample
        return self

    def __exit__(self, *exc):
        torch.randn = self._randn
        torch.distributions.Laplace.sample = self._sample


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True, help="wall_single directory")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--step", type=int, default=0, help="which batch of the reference's train order")
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--data-seed", type=int, default=0, help="training.seed of the reference (slice order)")
    ap.add_argument("--seed", type=int, default=1234, help="seed for RDMReg randomness")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base-width", type=int, default=384)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--reg-weight", type=float, default=0.5)
    ap.add_argument("--num-projections", type=int, default=8192)
    ap.add_argument("--target-p", type=float, default=1.0)
    ap.add_argument("--agg", default="b")
    args = ap.parse_args(argv)

    from datasets.wall_dset import load_wall_slice_train_val
    from datasets.img_transforms import default_transform
    from models.infojepa_modules import RDMReg, swd
    from models.mup import mup_param_groups
    from utils import seed as ref_seed

    hp, encoder, action_encoder, predictor, link = L.load_model_file(args.model)
    num_hist = hp.predictor_num_frames
    T = num_hist + 1
    for m in (encoder, action_encoder, predictor):
        m.train()
    disable_dropout(encoder, action_encoder, predictor)

    # --- batch, exactly as the reference pipeline orders it ---
    ref_seed(args.data_seed)
    datasets, traj = load_wall_slice_train_val(
        transform=default_transform(224), n_rollout=None, data_path=str(args.data_dir),
        normalize_action=True, split_ratio=0.9, split_mode="random",
        num_hist=num_hist, num_pred=1, frameskip=args.frameskip,
    )
    train_slices = datasets["train"]
    B = args.batch_size
    rows = [train_slices[i] for i in range(args.step * B, (args.step + 1) * B)]
    visual = torch.stack([r[0]["visual"] for r in rows]).float()  # (B, T, 3, H, W)
    actions = torch.stack([r[1] for r in rows]).float()            # (B, T, A)
    assert visual.shape[1] == T and actions.shape[-1] == hp.action_dim, (visual.shape, actions.shape)

    # --- forward: VWorldModel._forward_adaln without the optional terms ---
    torch.manual_seed(args.seed)
    b = visual.shape[0]
    u = encoder(rearrange(visual, "b t c h w -> (b t) c h w"))
    u = rearrange(u, "(b t) p d -> b t p d", b=b)
    z = link(u)
    a = action_encoder(actions)
    u_pred = predictor(z[:, :num_hist], a[:, :num_hist])
    z_pred = link(u_pred)
    z_loss = torch.nn.functional.mse_loss(z_pred, z[:, 1:])

    reg = RDMReg(target_p=args.target_p, num_projections=args.num_projections, agg=args.agg, mu=0.0)
    with RandomnessRecorder() as rec:
        reg_loss = reg.reg_loss(z, link=link)
    assert len(rec.laplace) == 1 and len(rec.randn) == 1, (len(rec.laplace), len(rec.randn))
    base, proj_raw = rec.laplace[0], rec.randn[0]
    proj = proj_raw / proj_raw.norm(dim=1, keepdim=True)
    # cross-check the recorded randomness reproduces the loss exactly
    from models.infojepa_modules import AGG_PATTERNS
    z_agg = rearrange(z, AGG_PATTERNS[args.agg])  # (B, TP, C) for agg=b: sample axis is B
    target = link(base).detach()
    pz = torch.sort(z_agg @ proj.T, dim=0).values
    pt = torch.sort(target @ proj.T, dim=0).values
    reg_check = ((pz - pt) ** 2).mean()
    assert torch.allclose(reg_check, reg_loss, rtol=1e-6, atol=1e-7), (reg_check.item(), reg_loss.item())

    loss = z_loss + args.reg_weight * reg_loss

    # --- backward + one muP-AdamW step as train.py ---
    params = {}
    for prefix, module in zip(L.COMPONENTS, (encoder, action_encoder, predictor)):
        for name, p in module.named_parameters():
            params[f"{prefix}.{name}"] = p
    for p in params.values():
        p.grad = None
    loss.backward()
    grads = {k: p.grad.detach().clone() for k, p in params.items()}

    opts = [
        torch.optim.AdamW(mup_param_groups(encoder, args.lr, args.base_width, weight_decay=args.wd, tag="encoder", verbose=False)),
        torch.optim.AdamW(mup_param_groups(predictor, args.lr, args.base_width, weight_decay=args.wd, tag="predictor", verbose=False)),
        torch.optim.AdamW(mup_param_groups(action_encoder, args.lr, args.base_width, weight_decay=args.wd, tag="action_encoder", verbose=False)),
    ]
    for o in opts:
        o.step()
    post = {k: p.detach().clone() for k, p in params.items()}

    # --- write ---
    tensors = [
        # ggml tensors have at most 4 dims: store frames as (B*T, 3, H, W), frame-major per sample
        ("in.visual", visual.reshape(b * T, *visual.shape[2:]).numpy()),
        ("in.actions", actions.numpy()),
        ("rdm.proj", proj.numpy()),
        ("rdm.target_sorted", pt.permute(1, 2, 0).contiguous().numpy()),  # (TP, K, B)
        ("fwd.z", z.detach().numpy()),
        ("fwd.z_pred", z_pred.detach().numpy()),
        ("loss.z", np.array([z_loss.item()], dtype=np.float32)),
        ("loss.reg", np.array([reg_loss.item()], dtype=np.float32)),
        ("loss.total", np.array([loss.item()], dtype=np.float32)),
    ]
    tensors += [(f"grad.{k}", v.numpy()) for k, v in grads.items()]
    tensors += [(f"post.{k}", v.numpy()) for k, v in post.items()]
    meta = {
        "lewm.gradcheck.model": str(args.model.resolve()),
        "lewm.gradcheck.batch_size": B,
        "lewm.gradcheck.step": args.step,
        "lewm.gradcheck.num_hist": num_hist,
        "lewm.gradcheck.lr": float(args.lr),
        "lewm.gradcheck.base_width": args.base_width,
        "lewm.gradcheck.wd": float(args.wd),
        "lewm.gradcheck.beta1": 0.9,
        "lewm.gradcheck.beta2": 0.999,
        "lewm.gradcheck.eps": 1e-8,
        "lewm.gradcheck.reg_weight": float(args.reg_weight),
        "lewm.gradcheck.num_projections": args.num_projections,
        "lewm.gradcheck.dropout": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    L.write_gguf(args.out, None, tensors, meta)
    print(f"wrote {args.out}  tensors={len(tensors)}  loss z={z_loss.item():.6f} reg={reg_loss.item():.6f} total={loss.item():.6f}")
    gmax = max(float(v.abs().max()) for v in grads.values())
    print(f"  max |grad| over all params: {gmax:.3e}; codes nonzero frac: {(z != 0).float().mean().item():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
