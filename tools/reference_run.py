#!/usr/bin/env python
"""Trajectory parity, PyTorch side: a short deterministic training run of the reference modules
that consumes exactly what lewm-train consumes.

Same model file for the initial weights, the same uint8 frame file and batch plan (export_dataset.py),
the same RDMReg randomness (lewm_rng streams 1 and 2, seeked per step like src/bin/lewm_train.cpp),
dropout disabled on both sides, muP-AdamW with the same settings. Writes one JSON line per step with
the losses, in the same fields lewm-train logs, so the two curves can be compared directly.

Example (from lewm/):
  ../lpworldmodel/.venv/bin/python tools/reference_run.py --model runs/traj/init.gguf \\
      --data ../data/lewm/wall --steps 40 --out runs/traj/reference.jsonl
  GGML_CUDA_NO_TF32=1 build-cuda/src/lewm-train --model runs/traj/init.gguf --data ../data/lewm/wall \\
      --out runs/traj/lewm --device gpu --dropout 0 --max-steps 40 --no-val --log-every 1
  ../lpworldmodel/.venv/bin/python tools/compare_runs.py runs/traj/reference.jsonl runs/traj/lewm/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

import lewm_ref as L
from lewm_rng import Rng, STREAM_PROJ, STREAM_TARGET

LAPLACE_SCALE = 0.70710678118654752440  # unit-variance Laplace: 1/sqrt(2)


class ExportedDataset:
    """Reads what export_dataset.py wrote, the way src/lewm/dataset.cpp does."""

    def __init__(self, d: Path):
        import gguf

        r = gguf.GGUFReader(str(d / "dataset.gguf"))
        f = lambda k: int(r.fields[k].parts[r.fields[k].data[0]][0])
        self.n_ep = f("lewm.data.n_episodes")
        self.fpe = f("lewm.data.frames_per_episode")
        self.steps = f("lewm.data.steps")
        self.H, self.W = f("lewm.data.height"), f("lewm.data.width")
        self.A = f("lewm.data.action_dim")
        self.fs = f("lewm.data.frameskip")
        self.T = f("lewm.data.num_frames")
        tens = {t.name: np.array(t.data) for t in r.tensors}
        self.actions = tens["actions"].astype(np.float32)  # (n_ep, steps, A)
        self.plan_train = tens["plan.train"].astype(np.int64)  # (n, 2)
        # frame offsets: uniform (Wall) or ragged from episode_frames (PushT), as dataset.cpp does
        if self.fpe > 0:
            counts = np.full(self.n_ep, self.fpe, dtype=np.int64)
        else:
            counts = tens["episode_frames"].astype(np.int64)
        self.offset = np.concatenate([[0], np.cumsum(counts)])
        self.frames = np.memmap(d / "frames.u8", dtype=np.uint8, mode="r", shape=(int(self.offset[-1]), 3, self.H, self.W))

    def batch(self, first: int, B: int):
        rows = self.plan_train[first : first + B]
        vis = np.empty((B, self.T, 3, self.H, self.W), dtype=np.float32)
        act = np.empty((B, self.T, self.A * self.fs), dtype=np.float32)
        for b, (ep, start) in enumerate(rows):
            idx = start + self.fs * np.arange(self.T)
            x = self.frames[self.offset[ep] + idx].astype(np.float32) / np.float32(255.0)
            vis[b] = (x - np.float32(0.5)) / np.float32(0.5)  # exactly the reference transform on uint8 input
            a = self.actions[ep, start : start + self.fs * self.T]  # (fs*T, A)
            act[b] = a.reshape(self.T, self.fs * self.A)  # "(n f) d -> n (f d)"
        return torch.from_numpy(vis), torch.from_numpy(act)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0, help="lewm-train --seed")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base-width", type=int, default=384)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--reg-weight", type=float, default=0.5)
    ap.add_argument("--num-projections", type=int, default=8192)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    from models.mup import mup_param_groups
    from dump_gradients import disable_dropout  # same helper

    hp, encoder, action_encoder, predictor, link = L.load_model_file(args.model)
    dev = torch.device(args.device)
    for m in (encoder, action_encoder, predictor, link):
        m.to(dev).train()
    disable_dropout(encoder, action_encoder, predictor)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    data = ExportedDataset(args.data)
    B, T, C, K = args.batch_size, data.T, hp.code_dim, args.num_projections
    P = hp.predictor_num_patches
    TP = T * P
    num_hist = hp.predictor_num_frames
    assert T == num_hist + 1

    opts = [
        torch.optim.AdamW(mup_param_groups(encoder, args.lr, args.base_width, weight_decay=args.wd, tag="encoder", verbose=False)),
        torch.optim.AdamW(mup_param_groups(predictor, args.lr, args.base_width, weight_decay=args.wd, tag="predictor", verbose=False)),
        torch.optim.AdamW(mup_param_groups(action_encoder, args.lr, args.base_width, weight_decay=args.wd, tag="action_encoder", verbose=False)),
    ]
    rng_proj, rng_tgt = Rng(args.seed, STREAM_PROJ), Rng(args.seed, STREAM_TARGET)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(args.out, "w") as f:
        for step in range(args.steps):
            visual, actions = data.batch(step * B, B)
            visual, actions = visual.to(dev), actions.to(dev)

            rng_proj.seek(step * K * C)
            proj = torch.from_numpy(rng_proj.normal(K * C).reshape(K, C)).to(dev)
            proj = proj / proj.norm(dim=1, keepdim=True)
            rng_tgt.seek(step * B * TP * C)
            base = torch.from_numpy(rng_tgt.laplace(B * TP * C, LAPLACE_SCALE).reshape(B, TP, C)).to(dev)

            u = encoder(rearrange(visual, "b t c h w -> (b t) c h w"))
            u = rearrange(u, "(b t) p d -> b t p d", b=B)
            z = link(u)
            a = action_encoder(actions)
            u_pred = predictor(z[:, :num_hist], a[:, :num_hist])
            z_pred = link(u_pred)
            z_loss = torch.nn.functional.mse_loss(z_pred, z[:, 1:])

            z_agg = rearrange(z, "b t p d -> b (t p) d")  # agg "b": samples along B
            target = link(base).detach()
            pz = torch.sort(z_agg @ proj.T, dim=0).values
            pt = torch.sort(target @ proj.T, dim=0).values
            reg = ((pz - pt) ** 2).mean()
            loss = z_loss + args.reg_weight * reg

            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            for o in opts:
                o.step()

            rec = {"kind": "step", "step": step + 1, "epoch": 0, "batch": step, "loss": loss.item(),
                   "z_loss": z_loss.item(), "reg_loss": reg.item(), "time": time.time() - t0}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"step {step + 1}/{args.steps}  loss {loss.item():.6f} (z {z_loss.item():.6f} reg {reg.item():.6f})  {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
