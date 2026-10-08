# tools

Python-side tooling. Run with the reference implementation's interpreter,
`../../lpworldmodel/.venv/bin/python`, so the reference `models` package is importable and
pickled checkpoints can be loaded.

- `export_model.py`: reference checkpoint (or seeded random init) -> model file (GGUF)
- `dump_reference.py`: one fixed input through the reference model -> reference dump (GGUF)
- `export_dataset.py`: reference dataset -> flat file LeWM can memory-map (training milestone). `--env wall` (uniform episodes) or `--env pusht --n-rollout N` (ragged episodes decoded from mp4, with a read-back self-check against the reference's sliced dataset)
- `probe_predictor.py`: predictor probe on an exported dataset's validation slices, one open-loop step: next-code MSE of the predictor against the copy-last baseline, from a reference checkpoint (`--checkpoint`) or a model file (`--model`). Ratio near 1.0 means the predictor has not learned dynamics; validation loss alone does not show this
- `import_model.py`: model file -> reference checkpoint + run dir under `../lpworldmodel/runs/outputs/`, so the reference planner (`plan.py`) can evaluate a LeWM-trained model

Regenerate every fixture with `make_fixtures.sh`.
