"""Backend selection must not hide an explicitly requested GPU failure."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ndgpu.backend import device_name, get_backend, is_rocm


def fake_cupy(monkeypatch, *, hip, count=1, name=b"test GPU"):
    runtime = SimpleNamespace(
        is_hip=hip, getDeviceCount=lambda: count, getDevice=lambda: 0,
        getDeviceProperties=lambda _: {"name": name})
    xp = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))
    monkeypatch.setitem(sys.modules, "cupy", xp)
    return xp


@pytest.mark.parametrize("hip,family", [(False, "cuda"), (True, "rocm")])
@pytest.mark.parametrize("name", [b"test GPU", "test GPU"])
def test_gpu_family_and_label(monkeypatch, hip, family, name):
    xp = fake_cupy(monkeypatch, hip=hip, name=name)
    assert get_backend("gpu") is xp
    assert get_backend("auto") is xp
    assert get_backend(family) is xp
    assert is_rocm(xp) is hip
    assert device_name(xp) == f"{family} (cupy): test GPU"
    wrong = "cuda" if hip else "rocm"
    with pytest.raises(RuntimeError, match="build of CuPy"):
        get_backend(wrong)


def test_explicit_gpu_never_falls_back(monkeypatch):
    fake_cupy(monkeypatch, hip=True, count=0)
    assert get_backend("auto") is np
    with pytest.raises(RuntimeError, match="no GPU device"):
        get_backend("gpu")


def test_runtime_failure_only_auto_falls_back(monkeypatch):
    xp = fake_cupy(monkeypatch, hip=True)

    def unavailable():
        raise RuntimeError("driver inaccessible")

    xp.cuda.runtime.getDeviceCount = unavailable
    assert get_backend("auto") is np
    with pytest.raises(RuntimeError, match="driver inaccessible"):
        get_backend("gpu")


def test_cpu_does_not_import_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    assert get_backend("cpu") is np
    assert not is_rocm(np)
    assert device_name(np) == "cpu (numpy)"
