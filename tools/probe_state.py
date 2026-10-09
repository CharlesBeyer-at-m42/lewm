#!/usr/bin/env python
"""State probe: is the environment state linearly readable from the code?

Fits a ridge regression from the encoder's code to the PushT state (agent x/y and T-block x/y) on
frames from the exported dataset's train episodes and reports the validation-episode RMSE in state
units (pixels of the 512x512 arena) against the predict-the-mean baseline. A random-init model is
the floor worth comparing against: a random ViT already decodes position to some degree, so the
question is how much training adds. Takes a reference checkpoint or a LeWM model file; the
ground-truth states come from the reference dataset directory (states.pth per split).

    ../lpworldmodel/.venv/bin/python tools/probe_state.py --data ../data/lewm/pusht2000 \\
        --states ../data/bmw48/pusht_noise --model ../data/lewm/pusht_init_seed0.gguf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import lewm_ref as L
from reference_run import ExportedDataset

TARGETS = {"agent xy": slice(0, 2), "T xy": slice(2, 4)}


def sample_frames(ds: ExportedDataset, episodes: np.ndarray, n: int, rng: np.random.Generator):
    """n random (episode, frame) pairs, frames weighted by episode length."""
    lengths = ds.offset[episodes + 1] - ds.offset[episodes]
    ep = rng.choice(episodes, size=n, p=lengths / lengths.sum())
    fr = (rng.random(n) * (ds.offset[ep + 1] - ds.offset[ep])).astype(np.int64)
    return ep, fr


def encode(enc, link, ds: ExportedDataset, ep: np.ndarray, fr: np.ndarray, device: str, B: int = 64):
    raw, linked = [], []
    with torch.no_grad():
        for i in range(0, len(ep), B):
            idx = ds.offset[ep[i : i + B]] + fr[i : i + B]
            x = ds.frames[idx].astype(np.float32) / np.float32(255.0)
            x = (x - np.float32(0.5)) / np.float32(0.5)
            h = enc(torch.from_numpy(x).to(device)).flatten(1)  # (B, num_patches*D)
            raw.append(h.cpu()), linked.append(link(h).cpu())
    return torch.cat(raw).numpy().astype(np.float64), torch.cat(linked).numpy().astype(np.float64)


def ridge_fit(X, Y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xn = (X - mu) / sd
    A = Xn.T @ Xn + alpha * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xn.T @ (Y - Y.mean(0)))
    return lambda Z: ((Z - mu) / sd) @ W + Y.mean(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="reference model_*.pth")
    src.add_argument("--model", type=Path, help="LeWM model file (GGUF)")
    ap.add_argument("--data", type=Path, required=True, help="exported dataset directory")
    ap.add_argument("--states", type=Path, required=True, help="reference pusht data dir with train/ and val/ states.pth")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=10.0, help="ridge penalty on standardised features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.checkpoint is not None:
        enc, _, _, link, epoch = L.load_checkpoint_modules(args.checkpoint)
        label = f"{args.checkpoint.parent.parent.name}/{args.checkpoint.name} (epoch {epoch})"
    else:
        _, enc, _, _, link = L.load_model_file(args.model)
        label = str(args.model)
    link = link if link is not None else torch.nn.Identity()
    enc.to(args.device).eval(), link.to(args.device).eval()

    ds = ExportedDataset(args.data)
    n_val_eps = int(np.unique(ds.plan_val[:, 0]).size)
    n_tr_eps = ds.n_ep - n_val_eps
    st_tr = torch.load(args.states / "train" / "states.pth", weights_only=True)[:n_tr_eps].numpy()
    st_va = torch.load(args.states / "val" / "states.pth", weights_only=True)[:n_val_eps].numpy()
    assert st_tr.shape[0] == n_tr_eps and st_va.shape[0] == n_val_eps, (st_tr.shape, st_va.shape, n_tr_eps, n_val_eps)

    rng = np.random.default_rng(args.seed)
    ep_tr, fr_tr = sample_frames(ds, np.arange(n_tr_eps), args.n_train, rng)
    ep_va, fr_va = sample_frames(ds, np.arange(n_tr_eps, ds.n_ep), args.n_val, rng)
    Y_tr = st_tr[ep_tr, fr_tr, :4].astype(np.float64)
    Y_va = st_va[ep_va - n_tr_eps, fr_va, :4].astype(np.float64)

    feats_tr = encode(enc, link, ds, ep_tr, fr_tr, args.device)
    feats_va = encode(enc, link, ds, ep_va, fr_va, args.device)

    print(f"{label}: ridge from code to state, {args.n_train} train / {args.n_val} val frames, val RMSE in pixels")
    for tname, sl in TARGETS.items():
        base = np.sqrt(((Y_va[:, sl] - Y_tr[:, sl].mean(0)) ** 2).sum(1).mean())
        line = f"  {tname:9s} mean-baseline {base:6.1f}"
        for fname, Xtr, Xva in (("raw", feats_tr[0], feats_va[0]), ("linked", feats_tr[1], feats_va[1])):
            pred = ridge_fit(Xtr, Y_tr[:, sl], args.alpha)(Xva)
            rmse = np.sqrt(((pred - Y_va[:, sl]) ** 2).sum(1).mean())
            line += f"  {fname} {rmse:6.1f} (R2 {1 - rmse**2 / base**2:5.2f})"
        print(line)


if __name__ == "__main__":
    main()
