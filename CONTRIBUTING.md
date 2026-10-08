# Contributing

Contributions — issues and pull requests — are welcome on GitHub, under the
terms below.

## Terms

Contributions are accepted under the Apache License 2.0, the license this
project ships under (inbound = outbound). Every contribution must carry a
Developer Certificate of Origin sign-off (`git commit -s`); see
https://developercertificate.org/ for what that certifies. There is no
Contributor License Agreement and no relicensing right.

## Using the code

The code is available under the Apache License 2.0 (see `LICENSE`).
Contributions are accepted under the same terms (inbound = outbound).

## Releasing

`raptor-hawk` publishes to PyPI through `.github/workflows/publish.yml`,
triggered by pushing a tag `v<version>` that matches the `__version__`
literal in `hawk/__init__.py` (a mismatch fails the `check` job before
anything is published). `workflow_dispatch` runs the same build, check and
test-wheel steps without publishing.

1. Bump `__version__` in `hawk/__init__.py`, update `CHANGELOG.md`.
2. `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. The workflow builds manylinux wheels for CPython 3.10-3.13 with
   `cibuildwheel` (hawk's extension is host-only: no CUDA toolchain is
   needed to build it, only the sibling `aether`/`eagle` header trees),
   runs `twine check --strict`, tests each wheel together with a freshly
   built `aether-dsc` sibling on every supported CPython, then publishes
   via a PyPI Trusted Publisher (environment `pypi`; no token in this
   repository).

**Publish order across the family:** `raptor-hawk` depends on `aether-dsc`
(in the `aether` repo), which — together with `raptor-core` — has no family
dependencies and publishes first. Release `aether-dsc` before `raptor-hawk`.
