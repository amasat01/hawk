"""A machine with cuda-bindings but no NVIDIA driver (a CI runner, a build box)
answers "no device", never an exception from inside the binding."""
import builtins

import pytest

from hawk.compile import nvrtc, toolchain
from hawk.ir import HawkError


class _DriverLibraryMissing(RuntimeError):
    """Stands in for the binding's own error when libcuda cannot be found."""


@pytest.fixture
def no_driver_library(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cuda.bindings" and fromlist and "driver" in fromlist:
            raise _DriverLibraryMissing('"cuda" is an NVIDIA driver library')
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_the_driver_binding_failing_to_load_is_no_reachable_device(no_driver_library):
    with pytest.raises(HawkError, match="no CUDA device reachable"):
        nvrtc._current_device()


def test_the_device_probe_answers_none_without_a_driver(no_driver_library, monkeypatch):
    monkeypatch.setattr(toolchain, "_DEVICE_ARCH_MEMO", [])
    monkeypatch.setattr(toolchain, "_driver_device_arch", lambda: None)   # the ctypes probe finds no libcuda either
    monkeypatch.setitem(__import__("sys").modules, "cuda.bindings.driver", object())  # skip the ctypes-first branch
    assert toolchain._probed_device_arch() is None
