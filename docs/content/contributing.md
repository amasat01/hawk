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

## Building and testing

From a source checkout (see [installation](installation)):

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

```{toctree}
:maxdepth: 1

contributing/add_a_native_function
```
