# Contributing

Read [How we build](https://amasat01.github.io/raptor/content/principles.html) first: the twelve rules every pull request in the
family is read against, plus the style, release and license terms.

**This repo:** turns your Python into fast kernels. It traces a function, emits C++ built on aether, compiles it, and differentiates it.
Running a device artifact is eagle's job.

## Build and run the gate

hawk has a compiled half (`hawk._core`) and builds against two sibling checkouts. Clone `aether`, `eagle` and `hawk` side by side, then:

```bash
pip install ../aether/dsc               # aether's sealed header payload, first
pip install -e .[test]                  # builds hawk._core; $HAWK_AETHER_INCLUDE / $HAWK_EAGLE_INCLUDE override the sibling paths
python -m pytest tests -q -m "not gpu"  # the gate (.github/workflows/ci.yml)
```

Most tests also need `eagle` installed as a Python package to run traced kernels. Without it, host artifacts still run through
`hawk.load` and `hawk.run`. The free-threading rows run with `python -m pytest tests -m ft -W error::RuntimeWarning`.

Docs (`cd docs`): `make strict` (zero warnings), `make linkcheck`, and for notebooks `make nbexec` then `make nbcheck`. Keep a new
notebook under a minute with small committed outputs. `make nbcheck-hosted` runs them as a Colab user would; the comments above the
target in `docs/Makefile` say what it needs.

## What a PR here must show

- **Generated code mirrors the user's Python.** The emitted C++ keeps the user's names and structure, so it reads beside their
  source. A change to emit shows the generated code before and after for a small kernel.
- **A new native function touches six files in two repos.** Follow
  [Add a native function](https://amasat01.github.io/hawk/content/contributing/add_a_native_function.html); the gate
  (`tests/test_math_namespace.py`) fails if a step is missed.
- **Limits.** What hawk cannot spell is stated once, with the way out, in
  [when hawk can't spell it](https://amasat01.github.io/hawk/content/vocabulary/when_hawk_cant_spell_it.html).
- **Derivatives and host/device.** A new op has a derivative rule and runs on host and device.
- The family checklist from [How we build](https://amasat01.github.io/raptor/content/principles.html), nothing more.

## Style specifics

`ruff` as configured in `pyproject.toml` (line length 88, pep257 docstrings). Run `ruff check .` before pushing.

## Release

Bump `__version__` in `hawk/__init__.py` and update `CHANGELOG.md` in a pull request. Its green run builds and tests the
wheels and the sdist; merge it up to date, then tag `vX.Y.Z` on that commit. `.github/workflows/publish.yml` uploads the files
the pull request built (`workflow_dispatch` dry-runs it). `aether-dsc` publishes first (see the family guide).
