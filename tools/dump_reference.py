#!/usr/bin/env python
"""Reference dump: one fixed input through the reference implementation -> GGUF.

The modules are rebuilt FROM THE MODEL FILE (not re-initialised), so the dump is guaranteed to
correspond to the weights LeWM will load. CPU, fp32, eval mode.

Example (from lewm/):
  ../lpworldmodel/.venv/bin/python tools/dump_reference.py \\
      --model testdata/random_seed0.gguf --out testdata/random_seed0.dump.gguf \\
      --data-dir ../../data/bmw48/wall_single --episode 0 --start 0 --frameskip 5 --seed 0

Tensor names in the dump (numpy shapes; ggml sees them reversed):
  in.visual                 (T, 3, H, W)        T = num_hist + 1 frames, after the transform
  in.actions                (T_act, A)          seeded random chunked actions, normalised space
  encoder.tokens_in         (T, 1+G*G, D)       after cls concat + pos embedding
  encoder.block.{i}.out     (T, 1+G*G, D)
  encoder.norm.out          (T, 1+G*G, D)
  encoder.u                 (T, P, C)           projector output (pre-link)
  encoder.z                 (T, P, C)           link(u): the codes
  action_encoder.out        (T_act, C)          action codes
  predictor.tokens_in       (num_hist*P, C)     after temporal (+spatial) pos, frame-major
  predictor.block.{i}.out   (num_hist*P, C)
  predictor.norm.out        (num_hist*P, C)
  predictor.u_pred          (num_hist, P, C)    pre-link prediction, teacher-forced on z[:num_hist]
  predictor.z_pred          (num_hist, P, C)
  rollout.z                 (T_act+1, P, C)     open-loop rollout from in.visual[0] over in.actions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import lewm_ref as L


def load_wall_frames(data_dir: Path, episode: int, frame_idx: list[int]) -> torch.Tensor:
    from datasets.img_transforms import default_transform

    obs = torch.load(data_dir / "obses" / f"episode_{episode:03d}.pth")  # (T, 3, H, W) 0..255
    img = obs[frame_idx] / 255
    return default_transform(224)(img).float()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True, help="wall_single directory")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--rollout-steps", type=int, default=6, help="T_act: chunked actions for the rollout")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    hp, encoder, action_encoder, predictor, link = L.load_model_file(args.model)
    num_hist = hp.predictor_num_frames
    T = num_hist + 1
    T_act = max(args.rollout_steps, num_hist)

    frame_idx = [args.start + i * args.frameskip for i in range(T)]
    visual = load_wall_frames(args.data_dir, args.episode, frame_idx)  # (T, 3, 224, 224)
    actions = torch.randn(T_act, hp.action_dim, generator=torch.Generator().manual_seed(args.seed))

    taps: dict[str, np.ndarray] = {}

    def tap(name):
        def hook(_m, _inp, out):
            taps[name] = out.detach().float().numpy().copy()
        return hook

    def pre_tap(name):
        def hook(_m, inp):
            taps[name] = inp[0].detach().float().numpy().copy()
        return hook

    hooks = []
    hooks.append(encoder.transformer.layers[0].register_forward_pre_hook(pre_tap("encoder.tokens_in")))
    for i, blk in enumerate(encoder.transformer.layers):
        hooks.append(blk.register_forward_hook(tap(f"encoder.block.{i}.out")))
    hooks.append(encoder.transformer.norm.register_forward_hook(tap("encoder.norm.out")))
    hooks.append(predictor.transformer.layers[0].register_forward_pre_hook(pre_tap("predictor.tokens_in")))
    for i, blk in enumerate(predictor.transformer.layers):
        hooks.append(blk.register_forward_hook(tap(f"predictor.block.{i}.out")))
    hooks.append(predictor.transformer.norm.register_forward_hook(tap("predictor.norm.out")))

    with torch.no_grad():
        # --- encoder over all T frames (batch = frames) ---
        u = encoder(visual)  # (T, P, C)
        z = link(u)
        enc_taps = {k: v for k, v in taps.items() if k.startswith("encoder.")}
        enc_taps["encoder.u"] = u.numpy().copy()
        enc_taps["encoder.z"] = z.numpy().copy()

        # --- action codes ---
        a = action_encoder(actions.unsqueeze(0)).squeeze(0)  # (T_act, C)

        # --- teacher-forced prediction over the history window (batch = 1) ---
        taps.clear()
        u_pred = predictor(z[:num_hist].unsqueeze(0), a[:num_hist].unsqueeze(0))  # (1, num_hist, P, C)
        z_pred = link(u_pred)
        pred_taps = {k: v.squeeze(0) for k, v in taps.items() if k.startswith("predictor.")}
        pred_taps["predictor.u_pred"] = u_pred.squeeze(0).numpy().copy()
        pred_taps["predictor.z_pred"] = z_pred.squeeze(0).numpy().copy()

        # --- open-loop rollout from frame 0, mirroring VWorldModel._rollout_adaln ---
        for h in hooks:
            h.remove()
        emb = z[:1].unsqueeze(0)  # (1, 1, P, C) linked
        act_emb_all = a.unsqueeze(0)  # (1, T_act, C)

        def predict_next(emb):
            Lc = emb.shape[1]
            hwin = min(num_hist, Lc)
            pred = predictor(emb[:, Lc - hwin :], act_emb_all[:, Lc - hwin : Lc])
            return link(pred[:, -1:])

        while emb.shape[1] < T_act:
            emb = torch.cat([emb, predict_next(emb)], dim=1)
        emb = torch.cat([emb, predict_next(emb)], dim=1)  # (1, T_act+1, P, C)
        rollout_z = emb.squeeze(0).numpy().copy()

    tensors = [("in.visual", visual.numpy()), ("in.actions", actions.numpy())]
    tensors += sorted(enc_taps.items())
    tensors += [("action_encoder.out", a.numpy().copy())]
    tensors += sorted(pred_taps.items())
    tensors += [("rollout.z", rollout_z)]

    meta = {
        "lewm.dump.model": str(args.model.resolve()),
        "lewm.dump.episode": args.episode,
        "lewm.dump.start": args.start,
        "lewm.dump.frameskip": args.frameskip,
        "lewm.dump.seed": args.seed,
        "lewm.dump.num_hist": num_hist,
        "lewm.dump.rollout_steps": T_act,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    L.write_gguf(args.out, None, tensors, meta)

    print(f"wrote {args.out}  tensors={len(tensors)}")
    for name, arr in tensors:
        arr = np.asarray(arr)
        print(f"  {name:28s} {str(tuple(arr.shape)):22s} |max|={np.abs(arr).max():.4g}  nz={np.count_nonzero(arr)/arr.size:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
