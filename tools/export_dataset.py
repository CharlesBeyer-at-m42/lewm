#!/usr/bin/env python
"""Dataset exporter: Wall or PushT (reference layout) -> frame file + companion GGUF for LeWM training.

Builds the reference datasets exactly as train.py does (same seed, same split, same slice
permutation), so the exported batch plan reproduces the reference's batch order.

Outputs in --out-dir:
  frames.u8      raw uint8 frames (3, H, W) in episode order, C-order. Wall: every episode has
                 frames_per_episode frames. PushT (ragged): episode e has episode_frames[e] frames
                 and lewm.data.frames_per_episode is 0; the mp4 episodes are decoded once here.
                 Pixels as stored by the reference (0..255); LeWM normalises to [-1, 1].
  dataset.gguf   metadata + small tensors:
                   actions          (n_episodes, steps, action_dim) f32, normalised as the reference does
                   episode_frames   (n_episodes,) i32, PushT only: frames per episode (seq_lengths)
                   episodes.train   (n_train,) i32 episode indices of the split
                   episodes.val     (n_val,)   i32
                   plan.train       (n_slices, 2) i32 rows of (episode, start) in the reference's order
                   plan.val         (n_slices, 2) i32
                 lewm.data.* keys: n_episodes, frames_per_episode, steps, height, width, action_dim,
                 frameskip, num_frames, action_mean/std as f32 arrays.

Examples (from lewm/):
  ../lpworldmodel/.venv/bin/python tools/export_dataset.py --data-dir ../data/bmw48/wall_single \\
      --out-dir ../data/lewm/wall --num-hist 3 --frameskip 5 --seed 0
  ../lpworldmodel/.venv/bin/python tools/export_dataset.py --env pusht --n-rollout 2000 \\
      --data-dir ../data/bmw48/pusht_noise --out-dir ../data/lewm/pusht2000 --num-hist 3 --frameskip 5 --seed 0
PushT episodes are the first --n-rollout of train/ (what env.dataset.n_rollout gives the reference)
followed by all of val/; the split is the directory split, not --split-ratio.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import lewm_ref as L  # noqa: F401  (puts the reference on sys.path)
import gguf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", choices=("wall", "pusht"), default="wall")
    ap.add_argument("--n-rollout", type=int, default=None, help="pusht: env.dataset.n_rollout of the reference run")
    ap.add_argument("--data-dir", type=Path, required=True, help="wall_single or pusht_noise directory")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-hist", type=int, default=3)
    ap.add_argument("--num-pred", type=int, default=1)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--split-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0, help="training.seed of the reference run")
    ap.add_argument("--skip-frames", action="store_true", help="only write dataset.gguf")
    args = ap.parse_args(argv)

    from datasets.img_transforms import default_transform
    from utils import seed as ref_seed  # the reference's seed() helper

    ref_seed(args.seed)
    if args.env == "wall":
        from datasets.wall_dset import load_wall_slice_train_val
        datasets, traj = load_wall_slice_train_val(
            transform=default_transform(224),
            n_rollout=None,
            data_path=str(args.data_dir),
            normalize_action=True,
            split_ratio=args.split_ratio,
            split_mode="random",
            num_hist=args.num_hist,
            num_pred=args.num_pred,
            frameskip=args.frameskip,
        )
        base = traj["train"].dataset  # WallDataset under the TrajSubset
        train_sub, val_sub = traj["train"], traj["valid"]
        train_idx = np.asarray(train_sub.indices, dtype=np.int32)
        val_idx = np.asarray(val_sub.indices, dtype=np.int32)

        def plan(slices_dset, subset):
            # TrajSlicerDataset.slices rows are (subset_index, start, end); map to base episode index
            rows = np.asarray(slices_dset.slices, dtype=np.int64)
            ep = np.asarray(subset.indices, dtype=np.int64)[rows[:, 0]]
            return np.stack([ep, rows[:, 1]], axis=1).astype(np.int32)

        plan_train = plan(datasets["train"], train_sub)
        plan_val = plan(datasets["valid"], val_sub)
        actions = base.actions.float().numpy()  # (n_ep, steps, 2), already normalised
        action_mean, action_std = base.action_mean, base.action_std
        episode_frames = None
        frames_per_episode = actions.shape[1]  # the slicer only ever touches frames [0, steps)
    else:
        from datasets.pusht_dset import load_pusht_slice_train_val
        datasets, traj = load_pusht_slice_train_val(
            transform=default_transform(224),
            n_rollout=args.n_rollout,
            data_path=str(args.data_dir),
            normalize_action=True,
            split_ratio=args.split_ratio,
            num_hist=args.num_hist,
            num_pred=args.num_pred,
            frameskip=args.frameskip,
            with_velocity=True,
        )
        train_dset, val_dset = traj["train"], traj["valid"]  # separate PushTDataset objects (train/, val/)
        n_tr, n_va = len(train_dset), len(val_dset)
        train_idx = np.arange(n_tr, dtype=np.int32)
        val_idx = np.arange(n_tr, n_tr + n_va, dtype=np.int32)

        def plan(slices_dset, offset):
            rows = np.asarray(slices_dset.slices, dtype=np.int64)
            return np.stack([rows[:, 0] + offset, rows[:, 1]], axis=1).astype(np.int32)

        plan_train = plan(datasets["train"], 0)
        plan_val = plan(datasets["valid"], n_tr)
        actions = np.concatenate([train_dset.actions.float().numpy(), val_dset.actions.float().numpy()], axis=0)
        action_mean, action_std = train_dset.action_mean, train_dset.action_std
        episode_frames = np.asarray(list(train_dset.seq_lengths) + list(val_dset.seq_lengths), dtype=np.int32)
        frames_per_episode = 0  # ragged

    n_ep, steps, adim = actions.shape

    args.out_dir.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(args.out_dir / "dataset.gguf"), "lewm")
    w.add_uint32("lewm.data.n_episodes", n_ep)
    w.add_uint32("lewm.data.frames_per_episode", frames_per_episode)
    w.add_uint32("lewm.data.steps", steps)
    w.add_uint32("lewm.data.height", 224)
    w.add_uint32("lewm.data.width", 224)
    w.add_uint32("lewm.data.action_dim", adim)
    w.add_uint32("lewm.data.frameskip", args.frameskip)
    w.add_uint32("lewm.data.num_frames", args.num_hist + args.num_pred)
    w.add_uint32("lewm.data.seed", args.seed)
    w.add_string("lewm.data.source", str(args.data_dir.resolve()))
    w.add_string("lewm.data.frames_file", "frames.u8")
    w.add_array("lewm.data.action_mean", [float(v) for v in action_mean])
    w.add_array("lewm.data.action_std", [float(v) for v in action_std])
    w.add_tensor("actions", np.ascontiguousarray(actions, dtype=np.float32))
    if episode_frames is not None:
        w.add_tensor("episode_frames", np.ascontiguousarray(episode_frames))
    w.add_tensor("episodes.train", train_idx)
    w.add_tensor("episodes.val", val_idx)
    w.add_tensor("plan.train", np.ascontiguousarray(plan_train))
    w.add_tensor("plan.val", np.ascontiguousarray(plan_val))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"dataset.gguf: {n_ep} episodes, {steps} steps, split {len(train_idx)}/{len(val_idx)}, "
          f"plan {len(plan_train)}/{len(plan_val)} slices")

    if args.skip_frames:
        return 0

    frame_bytes = 3 * 224 * 224
    path = args.out_dir / "frames.u8"
    t0 = time.time()
    offsets = [0]
    with open(path, "wb") as f:
        for ep in range(n_ep):
            if args.env == "wall":
                obs = torch.load(args.data_dir / "obses" / f"episode_{ep:03d}.pth")  # (T, 3, H, W) float 0..255
                arr = obs[:frames_per_episode].round().clamp(0, 255).to(torch.uint8).numpy()
                assert arr.shape == (frames_per_episode, 3, 224, 224), arr.shape
            else:
                from decord import VideoReader
                dset, local = (train_dset, ep) if ep < n_tr else (val_dset, ep - n_tr)
                reader = VideoReader(str(dset.data_path / "obses" / f"episode_{local:03d}.mp4"), num_threads=2)
                n = int(episode_frames[ep])
                assert len(reader) >= n, (ep, len(reader), n)
                arr = reader.get_batch(list(range(n))).numpy()  # (T, H, W, 3) uint8, the stored pixels
                arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
                assert arr.shape == (n, 3, 224, 224) and arr.dtype == np.uint8, (arr.shape, arr.dtype)
            f.write(arr.tobytes())
            offsets.append(offsets[-1] + arr.shape[0])
            if ep % 100 == 0 or ep == n_ep - 1:
                el = time.time() - t0
                print(f"  episode {ep + 1}/{n_ep}  {el:6.0f}s", flush=True)
    size = path.stat().st_size
    assert size == offsets[-1] * frame_bytes, size
    print(f"frames.u8: {size / 1e9:.2f} GB")

    # Self-check: a few slices read back through the exported layout must equal what the
    # reference's sliced dataset feeds the model (decode, frameskip, transform).
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    rng = np.random.default_rng(0)
    T = args.num_hist + args.num_pred
    for name, slices_dset, plan_rows in (("train", datasets["train"], plan_train), ("val", datasets["valid"], plan_val)):
        for k in rng.choice(len(plan_rows), size=3, replace=False):
            ep, start = int(plan_rows[k, 0]), int(plan_rows[k, 1])
            ref_vis = slices_dset[int(k)][0]["visual"].numpy()  # (T, 3, H, W) normalised
            got = []
            for t in range(T):
                fi = offsets[ep] + start + t * args.frameskip
                raw = np.asarray(mm[fi * frame_bytes:(fi + 1) * frame_bytes]).reshape(3, 224, 224).astype(np.float32)
                got.append((raw / 255.0 - 0.5) / 0.5)
            diff = float(np.abs(np.stack(got) - ref_vis).max())
            print(f"  self-check {name} slice {k}: episode {ep} start {start} max abs diff {diff:.2e}")
            assert diff < 1e-6, diff
    return 0


if __name__ == "__main__":
    sys.exit(main())
