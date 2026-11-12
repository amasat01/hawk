# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Install smoke: hawk found from INSTALLED distributions only. Run ``python -I tools/w25_install_smoke.py`` from a folder that holds
no source tree, with no ``PYTHONPATH``, in an environment where the wheels are installed. It prints where hawk was imported from,
resolves every first-party variant and component through its entry point, builds a host kernel, runs it (when eagle is importable) and
checks the value against NumPy. Exit code 0 only on success."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile


def main() -> int:
    work = tempfile.mkdtemp(prefix="hawk_smoke_")
    os.environ["HAWK_CACHE_DIR"] = os.path.join(work, "cache")
    import numpy as np

    import hawk
    from hawk import Param, Scalar, components, registry, variants

    print("hawk.__file__", hawk.__file__)
    if os.path.sep + "site-packages" + os.path.sep not in hawk.__file__:
        print("INSTALL SMOKE FAILED: hawk is not imported from an installed distribution")
        return 1
    for name in ("arm", "cpu", "cuda", "simt", "x86"):
        data = variants.load(name)
        print(f"variant {name}: {variants.location(name)} (max_dim {data['linalg']['max_dim']})")
    for name in ("lowering", "simt"):
        print(f"component {name}: {components.component(name).__file__}")
    print("backends (default set):", registry.backend_ids(default=True), "refused:", registry.refused())

    @hawk.kernel
    def k(x: Scalar, a: Param) -> Scalar:
        return a * x + 1.0

    hawk.build([k], os.path.join(work, "out"), targets=("host",), cache_dir=os.path.join(work, "cache"))
    print("host kernel built")
    x = np.arange(4.0)
    try:
        y = np.asarray(k(x, 2.0))
    except ImportError as e:
        print(f"run skipped: eagle is not importable ({e})")
    else:
        if not np.allclose(y, 2.0 * x + 1.0):
            print("INSTALL SMOKE FAILED: the value differs from NumPy:", y)
            return 1
        print("value equals NumPy:", y)
    print("INSTALL SMOKE OK host")
    if shutil.which("nvcc"):
        hawk.build([k], os.path.join(work, "cuda"), targets=("cuda",), cache_dir=os.path.join(work, "cache"))
        print("INSTALL SMOKE OK cuda")
    else:
        print("INSTALL SMOKE cuda: toolchain absent")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
