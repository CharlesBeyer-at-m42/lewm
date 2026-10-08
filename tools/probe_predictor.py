#!/usr/bin/env python
"""Predictor probe: does the predictor beat copying the last code?

One open-loop step on validation slices of an exported dataset: encode the T frames of a slice,
feed codes 0..T-2 and their actions to the predictor, and compare the predicted last code with the
encoded last code (MSE) against the copy-last baseline (code T-2 vs code T-1). The ratio is the
number to read: near 1.0 the predictor has learned nothing beyond persistence. The Wall reference
at epoch 5 scores ~0.05; the 2000-episode PushT runs score ~0.68, and their validation loss alone
did not reveal that. Takes a reference checkpoint or a LeWM model file:

    ../lpworldmodel/.venv/bin/python tools/probe_predictor.py --data ../data/lewm/pusht2000 \\
        --checkpoint ../lpworldmodel/runs/outputs/lewm_ref_pusht/checkpoints/model_1.pth
    ../lpworldmodel/.venv/bin/python tools/probe_predictor.py --data ../data/lewm/pusht2000 \\
        --model runs/pusht_run/model_latest.gguf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import lewm_ref as L
from reference_run import ExportedDataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="reference model_*.pth (pickled modules)")
    src.add_argument("--model", type=Path, help="LeWM model file (GGUF)")
    ap.add_argument("--data", type=Path, required=True, help="exported dataset directory")
    ap.add_argument("--split", choices=("val", "train"), default="val")
    ap.add_argument("--batches", type=int, default=20, help="number of batches from the start of the split")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.checkpoint is not None:
        enc, act_enc, pred, link, epoch = L.load_checkpoint_modules(args.checkpoint)
        label = f"{args.checkpoint} (epoch {epoch})"
    else:
        _, enc, act_enc, pred, link = L.load_model_file(args.model)
        label = str(args.model)
    if link is None:
        link = torch.nn.Identity()
    for m in (enc, act_enc, pred, link):
        m.to(args.device).eval()

    ds = ExportedDataset(args.data)
    plan = ds.plan_val if args.split == "val" else ds.plan_train
    B = args.batch_size
    if args.batches * B > len(plan):
        raise SystemExit(f"{args.split} split has {len(plan)} slices, fewer than {args.batches} x {B}")

    se_pred = se_copy = active = 0.0
    with torch.no_grad():
        for i in range(args.batches):
            vis, act = ds.batch(i * B, B, plan)
            vis, act = vis.to(args.device), act.to(args.device)
            b, T = vis.shape[:2]
            z = link(enc(vis.flatten(0, 1))).view(b, T, 1, -1)  # linked codes (b, T, 1, D)
            a = act_enc(act)  # action codes (b, T, D)
            zh = link(pred(z[:, :-1], a[:, :-1]))  # predicts codes 1..T-1 from 0..T-2
            se_pred += ((zh[:, -1] - z[:, -1]) ** 2).mean().item() * b
            se_copy += ((z[:, -2] - z[:, -1]) ** 2).mean().item() * b
            active += (z != 0).float().mean().item() * b  # fraction of nonzero code entries
    n = args.batches * B
    se_pred, se_copy, active = se_pred / n, se_copy / n, active / n
    # copy-last MSE of 0 means the codes do not change between frames: the representation collapsed
    ratio = f"{se_pred / se_copy:.3f}" if se_copy > 0 else "n/a (collapsed)"
    print(f"{label}: next-code MSE predictor {se_pred:.5f}  copy-last {se_copy:.5f}  ratio {ratio}  active code entries {active:.3f}")


if __name__ == "__main__":
    main()
