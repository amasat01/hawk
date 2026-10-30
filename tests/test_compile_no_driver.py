"""A machine with cuda-bindings but no NVIDIA driver (a CI runner, a build box)
answers "no device", never an exception from inside the binding. The binding
imports fine there; it is the first driver call that fails to load libcuda."""
import sys
import types

import pytest

from hawk.compile import nvrtc, toolchain
from hawk.ir import HawkError


class _DriverLibraryMissing(RuntimeError):
    """Stands in for the binding's own error when libcuda cannot be found."""


def _fake_bindings(monkeypatch, *, fail_on_import: bool):
    driver = types.ModuleType("cuda.bindings.driver")

    def cuInit(_flags):
        raise _DriverLibraryMissing('"cuda" is an NVIDIA driver library')

    driver.cuInit = cuInit
    bindings = types.ModuleType("cuda.bindings")
    if fail_on_import:
        def __getattr__(name):
            raise _DriverLibraryMissing('"cuda" is an NVIDIA driver library')
        bindings.__getattr__ = __getattr__
    else:
        bindings.driver = driver
    cuda = types.ModuleType("cuda")
    cuda.bindings = bindings
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    if not fail_on_import:
        monkeypatch.setitem(sys.modules, "cuda.bindings.driver", driver)


@pytest.mark.parametrize("fail_on_import", [False, True], ids=["first-call", "import"])
def test_a_driver_library_that_cannot_load_is_no_reachable_device(monkeypatch, fail_on_import):
    _fake_bindings(monkeypatch, fail_on_import=fail_on_import)
    with pytest.raises(HawkError, match="no CUDA device reachable"):
        nvrtc._current_device()


def test_the_device_probe_answers_none_without_a_driver(monkeypatch):
    _fake_bindings(monkeypatch, fail_on_import=False)
    monkeypatch.setattr(toolchain, "_DEVICE_ARCH_MEMO", [])
    monkeypatch.setattr(toolchain, "_driver_device_arch", lambda: None)   # the ctypes probe finds no libcuda either
    assert toolchain._probed_device_arch() is None
