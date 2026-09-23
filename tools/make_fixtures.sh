#!/usr/bin/env bash
# Regenerate the parity fixtures under testdata/ (model files + reference dumps).
# Requires the reference implementation beside this repo with its .venv, the Wall dataset, and
# the trained checkpoint from the lewm_ref_wall run (skipped with a warning if absent).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=../lpworldmodel/.venv/bin/python
DATA=${DATASET_DIR:-../data/bmw48}/wall_single
CFG=../lpworldmodel/runs/outputs/lewm_ref_wall/hydra.yaml
CKPT=../lpworldmodel/runs/outputs/lewm_ref_wall/checkpoints/model_latest.pth
mkdir -p testdata

# 1. seeded random init, CLS mode
$PY tools/export_model.py --random --seed 0 --config "$CFG" --action-dim 10 --out testdata/random_seed0.gguf
$PY tools/dump_reference.py --model testdata/random_seed0.gguf --out testdata/random_seed0.dump.gguf \
    --data-dir "$DATA" --episode 0 --start 0 --frameskip 5 --seed 0

# 2. seeded random init, patch mode (exercises the block-causal mask and spatial pos-embed)
$PY - "$CFG" <<'EOF'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
cfg.encoder.feature = "patch"
cfg.encoder.name = "vit_scratch_patch"
OmegaConf.save(cfg, "testdata/hydra_patch.yaml")
EOF
$PY tools/export_model.py --random --seed 0 --config testdata/hydra_patch.yaml --action-dim 10 --out testdata/random_patch_seed0.gguf
$PY tools/dump_reference.py --model testdata/random_patch_seed0.gguf --out testdata/random_patch_seed0.dump.gguf \
    --data-dir "$DATA" --episode 3 --start 5 --frameskip 5 --seed 2

# 3. trained Wall checkpoint
if [ -f "$CKPT" ]; then
    $PY tools/export_model.py --checkpoint "$CKPT" --out testdata/wall_ref.gguf
    $PY tools/dump_reference.py --model testdata/wall_ref.gguf --out testdata/wall_ref.dump.gguf \
        --data-dir "$DATA" --episode 7 --start 10 --frameskip 5 --seed 1
else
    echo "warning: no trained checkpoint at $CKPT; skipping wall_ref fixtures" >&2
fi
ls -la testdata/
