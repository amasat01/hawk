# Contributing

Contributions — issues and pull requests — are welcome on the
[GitHub repository](https://github.com/amasat01/hawk), under the terms
below.

## Terms

Contributions are accepted under the Apache License 2.0, the license this
project ships under (inbound = outbound). Every contribution must carry a
Developer Certificate of Origin sign-off (`git commit -s`); see
https://developercertificate.org/ for what that certifies. There is no
Contributor License Agreement and no relicensing right.

## Building from source (development)

To work on hawk itself, build it from a clone. hawk has a compiled half (`hawk._core`, a nanobind extension) and depends
on two build-time siblings, `aether` and `eagle`, taken from checkouts for development. Clone all three
next to each other —

```text
workspace/
├── aether/
├── eagle/
└── hawk/
```

— then install aether's sealed header payload before hawk itself:

```bash
pip install ../aether/dsc
pip install -e .[test]
```

`pip install -e .` builds `hawk._core` through `CMakeLists.txt`
(scikit-build-core + nanobind, the same packaging eagle's own binding uses),
resolving the `aether`/`eagle` C++ header roots from the sibling checkouts
above — or from `$HAWK_AETHER_INCLUDE` / `$HAWK_EAGLE_INCLUDE` when they live
somewhere else.

To run a DEVICE artifact end to end (not just author and compile it) you
also need `eagle` installed as a Python package — see the "Python
package" section of [eagle's installation page](https://amasat01.github.io/eagle/content/userguide/installation.html) — since hawk compiles
a device artifact but never launches one itself. A host artifact needs
no `eagle` at all: `hawk.load`/`hawk.run` run it directly, as
[CPU-only](project:#cpu-only) shows.

## Building and testing

From a source checkout (see above):

```bash
pip install ../aether/dsc
pip install -e .[test]
pytest -m "not gpu"
```

`-m "not gpu"` deselects the tests that need a CUDA GPU and `cupy`; most of
the rest additionally need `eagle` installed as a Python package to execute
traced kernels on host or device.

## Building the docs

```bash
cd docs
make nbexec     # execute every tutorial/example notebook in place
make nbcheck    # refuse to proceed unless every notebook ran clean
make strict     # -W --keep-going: zero warnings allowed
make linkcheck  # verify every internal link resolves
```

Notebooks are executed **locally**, with their outputs committed — the
published site never re-executes them (`make strict` renders the outputs
already in the file). Keep a new notebook fast (well under a minute) and
its committed outputs small: no large arrays or plots.

## Using the code

The code is available under the Apache License 2.0 (see `LICENSE`).
Contributions are accepted under the same terms (inbound = outbound) — see
the [README](https://github.com/amasat01/hawk#readme).

## How-to

One guide for contributors:

- [Add a native function](contributing/add_a_native_function): the contributor path for a native op.

```{toctree}
:maxdepth: 1
:hidden:

contributing/add_a_native_function
```
