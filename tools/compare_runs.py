#!/usr/bin/env python
"""Compare two training logs (JSON lines with kind == "step") step by step.

  python tools/compare_runs.py reference.jsonl lewm/train.jsonl [--rtol 1e-3] [--atol 1e-5]

Prints a table of the per-step losses and the relative difference, and exits 1 if any step's total
loss differs by more than atol + rtol * |reference|. Chaotic divergence is expected eventually; the
point is that the first tens of steps agree to numerical precision.
"""

from __future__ import annotations

import argparse
import json
import sys


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") == "step":
                out[int(rec["step"])] = rec
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args(argv)
    ref, cand = load(args.reference), load(args.candidate)
    steps = sorted(set(ref) & set(cand))
    if not steps:
        print("no common steps")
        return 2
    print(f"{'step':>5} {'ref loss':>12} {'lewm loss':>12} {'rel diff':>10}   {'ref z':>10} {'lewm z':>10}   {'ref reg':>10} {'lewm reg':>10}  ok")
    worst = 0.0
    fails = 0
    for s in steps:
        r, c = ref[s], cand[s]
        d = abs(r["loss"] - c["loss"])
        rel = d / max(abs(r["loss"]), 1e-12)
        ok = d <= args.atol + args.rtol * abs(r["loss"])
        fails += not ok
        worst = max(worst, rel)
        print(f"{s:>5} {r['loss']:>12.6f} {c['loss']:>12.6f} {rel:>10.2e}   {r['z_loss']:>10.6f} {c['z_loss']:>10.6f}   {r['reg_loss']:>10.6f} {c['reg_loss']:>10.6f}  {'ok' if ok else 'FAIL'}")
    print(f"\n{len(steps)} common steps, {fails} beyond tolerance, worst relative difference {worst:.2e}")
    print("TRAJECTORY PARITY OK" if fails == 0 else "TRAJECTORY PARITY FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
