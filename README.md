# LeWM

A C++20 implementation, on [ggml](https://github.com/ggml-org/ggml), of the LpWM world model
("A Case for Sparse Representations in World Models", arXiv:2608.22764).

The reference implementation is the PyTorch repository `lpworldmodel`, cloned beside this one at
`../lpworldmodel`. LeWM must compute what it computes; parity tests enforce that.

## Where the design lives

- `ai/CONTEXT.md`: the glossary. Use its terms in code, tests, and commit messages.
- `ai/docs/adr/`: architecture decisions. Read `0001` (why standalone C++20 on ggml) and `0002`
  (scope: LpWM path only) before proposing structural changes.

## Layout

```
ggml/        git submodule, pinned (see .gitmodules)
src/         lewm library and binaries (lewm-parity, lewm-rollout)
tools/       Python: export_model.py, dump_reference.py, export_dataset.py
tests/       doctest test cases, run through CTest
```

## Building

```bash
git submodule update --init
cmake --preset default
cmake --build --preset default
ctest --preset default
```

Python tools run in the reference implementation's environment:

```bash
../lpworldmodel/.venv/bin/python tools/export_model.py --help
```

## Testdata

`testdata/*.gguf` are ignored by git. Regenerate with `tools/make_fixtures.sh`.
