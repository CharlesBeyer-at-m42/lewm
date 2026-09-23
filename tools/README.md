# tools

Python-side tooling. Run with the reference implementation's interpreter,
`../../lpworldmodel/.venv/bin/python`, so the reference `models` package is importable and
pickled checkpoints can be loaded.

- `export_model.py`: reference checkpoint (or seeded random init) -> model file (GGUF)
- `dump_reference.py`: one fixed input through the reference model -> reference dump (GGUF)
- `export_dataset.py`: reference dataset -> flat file LeWM can memory-map (training milestone)
