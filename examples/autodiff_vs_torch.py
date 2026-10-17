# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Autodiff demo: hawk-generated derivatives of one RK4 gravity step, checked
against ``torch.autograd``/``torch.func.jvp`` and against a float64 finite
difference.

The primal is the SAME planar two-body RK4 step as ``early_termination.py``
(no termination test here): ``x_next = step(x; dt)``, ``x`` a per-sample
``Vector[4]``, float32. hawk differentiates it w.r.t. ``x`` alone in both
directions -- ``hawk.diff.vjp`` (a random per-sample cotangent ``v`` ->
``bar_x``) and ``hawk.diff.jvp`` (a random tangent ``t`` -> ``dot_x_next``) --
and publishes the primal and BOTH derivative kernels in one
``hawk.build`` call, the way a producer that ships a forward
kernel alongside its own gradient is expected to (see
``hawk/tests/test_artifact_derivative_per_kernel.py``). Every plane the
derivative kernels declare (``bar_x``, ``bar_x_next``, ``dot_x``,
``dot_x_next``) is named through ``hawk.diff.ADJOINT_PREFIX``/
``TANGENT_PREFIX`` and then confirmed against the built bundle's own sidecars
-- never assumed.

The device arm (``--device``) checks hawk's vjp/jvp against
``torch.autograd.grad``/``torch.func.jvp`` on the same physics written
idiomatically in torch, PLUS a float64 central-difference Jacobian on a
handful of samples so the check does not rest on torch alone; it also feeds
torch CUDA tensors through ``eagle.to_cupy`` (zero-copy, as in
``zero_copy_interop.py``) for every input and output plane and reads the
gradients back from torch's own storage, asserting pointer identity the same
way. The host arm (``--host``) needs no GPU, cupy or torch: its reference is
the float64 finite difference alone (torch, if importable, adds an optional
CPU cross-check).

``--device`` also times, per call: hawk's vjp kernel against
``torch.autograd.grad`` (forward + backward, which is what torch must do for
a gradient), hawk's jvp kernel against ``torch.func.jvp``, and each side's
own primal alone (so the reader can see the derivative's overhead relative to
its own primal) -- warm-up excluded, median of ``--reps`` INTERLEAVED reps,
at the default batch (65536). ``--sweep`` prints the same table over batch
sizes 1, 32, 4096, 65536 and 1048576 (reps=1 per point, comfortably under two
minutes). The measured numbers come from the committed card
(``benchmarks/autodiff_vs_torch_card.md``); this sweep is a demo
measurement, not a benchmark, so run ``--sweep`` for the actual numbers on
your own card.

Run it with ``python examples/autodiff_vs_torch.py --host`` (CPU) or
``--device`` (GPU, needs torch); add ``--sweep`` for the timing table over
batch sizes.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import statistics
import tempfile
import time
from types import SimpleNamespace

import numpy as np

#: Gravitational parameter of the central body, in the demo's own units --
#: the same physics constant as ``early_termination.py``.
MU = 1.0
#: The primal's own plane names (this module's choice, not a guess): a
#: ``Vector[4]`` state in, the RK4-advanced state out.
PRIMAL_INPUT = "x"
PRIMAL_OUTPUT = "x_next"
#: Differentiate w.r.t. the state alone -- no Param gradients.
WRT = (PRIMAL_INPUT,)
#: The kernel names, and so the names every file the bundle publishes them
#: under carries.
KERNEL_NAME = "rk4_step"
VJP_NAME = f"{KERNEL_NAME}_vjp"
JVP_NAME = f"{KERNEL_NAME}_jvp"
#: Acceptance: hawk's vjp/jvp against torch (float32, so not bit-exact) and
#: against a float64 central-difference Jacobian, both directions, checked as
#: a per-sample vector-norm relative error -- the SAME metric and value as
#: early_termination.py's STATE_REL_TOL. Measured on a Quadro
#: P2000, host and device arms alike: the largest of the four checks this
#: module runs (vjp vs torch, jvp vs torch, vjp vs FD, jvp vs FD) is on the
#: order of 1e-7, comfortably two orders of magnitude under this tolerance --
#: so that value carries over UNCHANGED rather than being fitted to a larger
#: observed error.
REL_TOL = 1e-5
#: Finite-difference sanity: a central-difference Jacobian (float64) is built
#: for only the first FD_SAMPLES columns of a batch, at step FD_H -- a sanity
#: check does not need the whole batch, only enough samples to catch a
#: sign/order mistake the rule table itself already gates against (see
#: hawk/tests/test_diff_finite_differences.py).
FD_SAMPLES = 8
FD_H = 1e-5
#: Sweep batch sizes for ``--sweep`` / :func:`run_sweep`.
SWEEP_BATCHES = (1, 32, 4096, 65536, 1048576)


# --------------------------------------------------------------------------- #
# Seeding: well-conditioned planar orbit states (no termination here).
# --------------------------------------------------------------------------- #
def seed_states(batch: int, seed: int) -> np.ndarray:
    """A ``(4, batch)`` float64 plane of ``(rx, ry, vx, vy)`` states, every
    one comfortably inside a non-degenerate orbit (radius in ``(0.6, 1.4)``,
    tangential speed in ``(0.6, 1.2)``) -- large enough to keep ``rsqrt``
    well away from either a pole or an overflow."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.6, 1.4, batch)
    angle = rng.uniform(0.0, 2.0 * np.pi, batch)
    speed = rng.uniform(0.6, 1.2, batch)
    return np.stack([
        r * np.cos(angle), r * np.sin(angle),
        -speed * np.sin(angle), speed * np.cos(angle),
    ])


# --------------------------------------------------------------------------- #
# The primal kernel and its two hawk-generated derivatives.
# --------------------------------------------------------------------------- #
def step_kernel():
    """Trace the primal RK4 step -- the same physics as
    ``early_termination.py``'s ``et_step``, with no termination test."""
    import hawk
    import hawk.math as hm

    def rhs(s):
        r2 = s[0] * s[0] + s[1] * s[1]
        inv_r3 = hm.rsqrt(r2 * r2 * r2)
        return hm.vec(s[2], s[3], -MU * s[0] * inv_r3, -MU * s[1] * inv_r3)

    @hawk.kernel
    def rk4_step(x: hawk.Vector[4], dt: hawk.Param,
                x_next: hawk.Mutable[hawk.Vector[4]]):
        k1 = rhs(x)
        k2 = rhs(x + (0.5 * dt) * k1)
        k3 = rhs(x + (0.5 * dt) * k2)
        k4 = rhs(x + dt * k3)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return rk4_step


def derivative_kernels(primal) -> tuple:
    r"""The reverse (``vjp``) and forward (``jvp``) derivative kernels of
    ``primal`` w.r.t. :data:`WRT`, wrapped as ordinary
    :class:`hawk.trace.Kernel`\ s the way a caller who goes through no
    bundle-composing frontend must (a bare ``Derived`` sink tuple is not
    itself buildable) -- see
    ``hawk/tests/test_artifact_derivative_per_kernel.py`` lines 25-60."""
    from hawk import Kernel
    from hawk.diff import jvp, vjp

    return (Kernel(VJP_NAME, vjp(primal, wrt=WRT)),
           Kernel(JVP_NAME, jvp(primal, wrt=WRT)))


def build_artifact(directory, targets) -> object:
    """Emit, compile and publish the primal AND both its derivatives into
    ONE bundle -- the same manifest, the way a producer that ships a forward
    kernel alongside its own gradient is expected to."""
    import hawk

    primal = step_kernel()
    vjp_kernel, jvp_kernel = derivative_kernels(primal)
    return hawk.build([primal, vjp_kernel, jvp_kernel], pathlib.Path(directory),
                        mode="float32", targets=tuple(targets))


def check_manifest(bundle) -> dict:
    """Read the published manifest back and validate its shape through raptor."""
    from raptor.schema import validate_manifest

    doc = json.loads(pathlib.Path(bundle.manifest_path).read_text())
    validate_manifest(doc)
    return doc


# --------------------------------------------------------------------------- #
# The consumer side: load the artifact through eagle and run it.
# --------------------------------------------------------------------------- #
def _sidecar(bundle, name: str) -> dict:
    return next(a.sidecar for a in bundle.artifacts if a.name == name)


def host_plugin(bundle, name: str):
    """The plan-able view of the bundle's HOST entry (mirrors
    early_termination.py/zero_copy_interop.py)."""
    from hawk.artifact import plan_view

    library = ctypes.CDLL(str(pathlib.Path(bundle.directory) / f"{name}.so"))
    view = plan_view(_sidecar(bundle, name))
    entry = getattr(library, view.pop("host_entry"))
    return SimpleNamespace(
        host_entry=ctypes.cast(entry, ctypes.c_void_p).value,
        _keepalive=library, **view)


def device_plugin(bundle, name: str):
    """The plan-able view of the bundle's DEVICE entry (mirrors
    early_termination.py/zero_copy_interop.py)."""
    from eagle.registry import load_manifest
    from hawk.artifact import plan_view

    registry = load_manifest(pathlib.Path(bundle.manifest_path))
    loaded = registry[name]
    view = plan_view(_sidecar(bundle, name))
    view.pop("host_entry", None)
    return SimpleNamespace(
        device_function=loaded.fn.kernel.ptr,
        _keepalive=(registry, loaded), **view)


def _plane_names() -> SimpleNamespace:
    """The vjp/jvp plane names the naming RULE mints --
    ``<prefix>_<primal plane>`` via ``hawk.diff.ADJOINT_PREFIX``/
    ``TANGENT_PREFIX`` -- never a bare string literal. Every caller still
    confirms the ROLE each one actually got by reading it off the built
    bundle's own sidecar (see :func:`_assert_plane_roles`), rather than
    trusting the naming rule alone."""
    from hawk.diff import ADJOINT_PREFIX, TANGENT_PREFIX

    return SimpleNamespace(
        bar_out=f"{ADJOINT_PREFIX}_{PRIMAL_OUTPUT}",  # the vjp's cotangent seed, in
        bar_wrt=f"{ADJOINT_PREFIX}_{PRIMAL_INPUT}",   # the vjp's gradient, out
        dot_wrt=f"{TANGENT_PREFIX}_{PRIMAL_INPUT}",   # the jvp's tangent seed, in
        # the jvp's directional derivative, out
        dot_out=f"{TANGENT_PREFIX}_{PRIMAL_OUTPUT}",
    )


def _assert_plane_roles(plugin, expected: dict) -> None:
    """Read ``plugin.arg_spec`` (projected straight from the sidecar by
    ``plan_view``) and confirm every expected plane is there with the role
    named -- the "read from sidecars, never guess" check applied to the
    example's OWN bindings, not only to the test."""
    roles = {name: role for role, name in plugin.arg_spec}
    for name, role in expected.items():
        assert roles.get(name) == role, (
            f"expected plane {name!r} to carry role {role!r}, got "
            f"{roles.get(name)!r} (arg_spec={plugin.arg_spec})")


# --------------------------------------------------------------------------- #
# The float64 numpy reference physics -- for the finite-difference check.
# --------------------------------------------------------------------------- #
def _np_rhs(s: np.ndarray) -> np.ndarray:
    r2 = s[0] * s[0] + s[1] * s[1]
    inv_r3 = r2 ** -1.5
    return np.stack([s[2], s[3], -MU * s[0] * inv_r3, -MU * s[1] * inv_r3])


def _np_step(x: np.ndarray, dt: float) -> np.ndarray:
    k1 = _np_rhs(x)
    k2 = _np_rhs(x + 0.5 * dt * k1)
    k3 = _np_rhs(x + 0.5 * dt * k2)
    k4 = _np_rhs(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _fd_jacobian(x_sample: np.ndarray, dt: float, h: float = FD_H) -> np.ndarray:
    """The 4x4 central-difference Jacobian of :func:`_np_step` at one float64
    sample."""
    jac = np.zeros((4, 4))
    for j in range(4):
        e = np.zeros(4)
        e[j] = h
        jac[:, j] = (_np_step(x_sample + e, dt) - _np_step(x_sample - e, dt)) / (2 * h)
    return jac


def _fd_directional_errors(x64: np.ndarray, dt: float, v32: np.ndarray,
                           t32: np.ndarray, bar_x: np.ndarray,
                           dot_next: np.ndarray, n_samples: int = FD_SAMPLES
                           ) -> tuple:
    """Max relative error, over the first ``n_samples`` columns, of hawk's
    ``bar_x``/``dot_next`` against ``J.T @ v``/``J @ t`` for the SAME
    per-sample float64 central-difference Jacobian ``J`` -- one Jacobian per
    sample serves BOTH directions, which is what makes this the check that
    does not rest on torch alone."""
    max_vjp = 0.0
    max_jvp = 0.0
    for i in range(min(n_samples, x64.shape[1])):
        jac = _fd_jacobian(x64[:, i], dt)
        vjp_fd = jac.T @ v32[:, i].astype(np.float64)
        jvp_fd = jac @ t32[:, i].astype(np.float64)
        got_vjp = bar_x[:, i].astype(np.float64)
        got_jvp = dot_next[:, i].astype(np.float64)
        max_vjp = max(max_vjp, float(np.linalg.norm(got_vjp - vjp_fd)
                                     / max(np.linalg.norm(vjp_fd), 1e-9)))
        max_jvp = max(max_jvp, float(np.linalg.norm(got_jvp - jvp_fd)
                                     / max(np.linalg.norm(jvp_fd), 1e-9)))
    return max_vjp, max_jvp


def _max_rel_norm(a: np.ndarray, ref: np.ndarray) -> float:
    """Per-sample vector-norm relative error (the SAME metric early_termination.py uses
    for cross-implementation agreement), maximised over the batch."""
    diff_norm = np.linalg.norm(a - ref, axis=0)
    ref_norm = np.maximum(np.linalg.norm(ref, axis=0), 1e-6)
    return float(np.max(diff_norm / ref_norm))


# --------------------------------------------------------------------------- #
# The host arm: numpy in/out, no cupy or torch required.
# --------------------------------------------------------------------------- #
def run_host(bundle, state: np.ndarray, dt: float, seed: int = 1) -> dict:
    """Bind and run the primal, vjp and jvp kernels on CPU threads through
    ``eagle.exec.HostTeam``. The reference is the float64 finite difference
    alone; if torch happens to be importable, an OPTIONAL torch-CPU
    cross-check is added, but nothing here requires it (or cupy)."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    names = _plane_names()
    batch = state.shape[1]
    rng = np.random.default_rng(seed)
    x32 = np.ascontiguousarray(state, dtype=np.float32)
    v32 = rng.normal(size=(4, batch)).astype(np.float32)
    t32 = rng.normal(size=(4, batch)).astype(np.float32)

    p_plugin = host_plugin(bundle, KERNEL_NAME)
    v_plugin = host_plugin(bundle, VJP_NAME)
    j_plugin = host_plugin(bundle, JVP_NAME)
    _assert_plane_roles(v_plugin, {names.bar_out: "vec_in", names.bar_wrt: "mutable"})
    _assert_plane_roles(j_plugin, {names.dot_wrt: "vec_in", names.dot_out: "mutable"})

    plan_p = eplan.plan(p_plugin, structure=eexec.HostTeam)
    plan_v = eplan.plan(v_plugin, structure=eexec.HostTeam)
    plan_j = eplan.plan(j_plugin, structure=eexec.HostTeam)

    x_next = np.zeros_like(x32)
    plan_p.bind(x=x32, dt=dt, x_next=x_next).launch()

    bar_x = np.zeros_like(x32)
    plan_v.bind(
        **{"x": x32, "dt": dt, names.bar_out: v32, names.bar_wrt: bar_x}
    ).launch()

    dot_next = np.zeros_like(x32)
    plan_j.bind(
        **{"x": x32, "dt": dt, names.dot_wrt: t32, names.dot_out: dot_next}
    ).launch()

    vjp_fd_rel_err, jvp_fd_rel_err = _fd_directional_errors(
        state.astype(np.float64), dt, v32, t32, bar_x, dot_next)

    result = {
        "batch": batch, "dt": dt, "state": state, "v32": v32, "t32": t32,
        "bar_x": bar_x, "dot_next": dot_next,
        "vjp_fd_rel_err": vjp_fd_rel_err, "jvp_fd_rel_err": jvp_fd_rel_err,
        "torch_available": False,
    }

    try:
        import torch
    except ImportError:
        return result

    def rhs_t(s, dt_):
        r2 = s[0] * s[0] + s[1] * s[1]
        inv_r3 = torch.rsqrt(r2 * r2 * r2)
        return torch.stack((s[2], s[3], -MU * s[0] * inv_r3, -MU * s[1] * inv_r3))

    def step_t(x, dt_):
        k1 = rhs_t(x, dt_)
        k2 = rhs_t(x + 0.5 * dt_ * k1, dt_)
        k3 = rhs_t(x + 0.5 * dt_ * k2, dt_)
        k4 = rhs_t(x + dt_ * k3, dt_)
        return x + (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    xg = torch.from_numpy(x32.copy()).requires_grad_(True)
    vt = torch.from_numpy(v32.copy())
    tt = torch.from_numpy(t32.copy())
    out = step_t(xg, dt)
    torch_bar_x = torch.autograd.grad(out, xg, grad_outputs=vt)[0].detach().numpy()
    with torch.no_grad():
        x0 = torch.from_numpy(x32.copy())
        _out2, torch_dot_next = torch.func.jvp(lambda xx: step_t(xx, dt), (x0,), (tt,))
    result.update({
        "torch_available": True,
        "vjp_torch_rel_err": _max_rel_norm(bar_x, torch_bar_x),
        "jvp_torch_rel_err": _max_rel_norm(dot_next, torch_dot_next.numpy()),
    })
    return result


# --------------------------------------------------------------------------- #
# The device arm: torch CUDA tensors, zero-copy into hawk's kernels, checked
# against torch.autograd.grad / torch.func.jvp (and the same float64 FD).
# --------------------------------------------------------------------------- #
def run_device(bundle, state: np.ndarray, dt: float, seed: int = 1) -> dict:
    """Needs cupy and torch (a GPU-capable install). Every plane -- ``x``,
    ``v``, ``t``, and the primal/vjp/jvp OUTPUTS alike -- is a torch CUDA
    tensor viewed through ``eagle.to_cupy`` with no copy (as in
    ``zero_copy_interop.py``); the results are read back from torch's own
    storage, and one behavioural check (mutate ``x`` in place, relaunch the
    SAME bound plan with no rebind) proves the kernel reads that memory
    rather than a snapshot taken when the view was built."""
    import cupy as cp
    import eagle
    import eagle.exec as eexec
    import torch
    from eagle import plan as eplan

    names = _plane_names()
    batch = state.shape[1]
    rng = np.random.default_rng(seed)
    x32 = np.ascontiguousarray(state, dtype=np.float32)
    v32 = rng.normal(size=(4, batch)).astype(np.float32)
    t32 = rng.normal(size=(4, batch)).astype(np.float32)

    p_plugin = device_plugin(bundle, KERNEL_NAME)
    v_plugin = device_plugin(bundle, VJP_NAME)
    j_plugin = device_plugin(bundle, JVP_NAME)
    _assert_plane_roles(v_plugin, {names.bar_out: "vec_in", names.bar_wrt: "mutable"})
    _assert_plane_roles(j_plugin, {names.dot_wrt: "vec_in", names.dot_out: "mutable"})

    plan_p = eplan.plan(p_plugin, structure=eexec.DeviceKernel)
    plan_v = eplan.plan(v_plugin, structure=eexec.DeviceKernel)
    plan_j = eplan.plan(j_plugin, structure=eexec.DeviceKernel)

    # ---- zero-copy: torch CUDA tensors, viewed as cupy, no copy (as in
    # zero_copy_interop.py's T-IN-CUDA-ALIAS / T-OUT-CUDA-ALIAS). ----
    t_x = torch.from_numpy(x32.copy()).cuda().contiguous()
    t_v = torch.from_numpy(v32.copy()).cuda().contiguous()
    t_t = torch.from_numpy(t32.copy()).cuda().contiguous()
    t_x_next = torch.empty_like(t_x)
    t_bar_x = torch.empty_like(t_x)
    t_dot_next = torch.empty_like(t_x)

    cu_x = eagle.to_cupy(t_x)
    cu_v = eagle.to_cupy(t_v)
    cu_t = eagle.to_cupy(t_t)
    cu_x_next = eagle.to_cupy(t_x_next)
    cu_bar_x = eagle.to_cupy(t_bar_x)
    cu_dot_next = eagle.to_cupy(t_dot_next)

    aliases = {
        "in_alias_x": bool(cu_x.data.ptr == t_x.data_ptr()),
        "in_alias_v": bool(cu_v.data.ptr == t_v.data_ptr()),
        "in_alias_t": bool(cu_t.data.ptr == t_t.data_ptr()),
        "out_alias_x_next": bool(cu_x_next.data.ptr == t_x_next.data_ptr()),
        "out_alias_bar_x": bool(cu_bar_x.data.ptr == t_bar_x.data_ptr()),
        "out_alias_dot_next": bool(cu_dot_next.data.ptr == t_dot_next.data_ptr()),
    }

    bound_p = plan_p.bind(x=cu_x, dt=dt, x_next=cu_x_next)
    bound_v = plan_v.bind(
        **{"x": cu_x, "dt": dt, names.bar_out: cu_v, names.bar_wrt: cu_bar_x}
    )
    bound_j = plan_j.bind(
        **{"x": cu_x, "dt": dt, names.dot_wrt: cu_t, names.dot_out: cu_dot_next}
    )

    bound_p.launch()
    bound_v.launch()
    bound_j.launch()
    cp.cuda.runtime.deviceSynchronize()

    # the pointers never moved: every kernel wrote INTO the torch buffers'
    # own storage.
    aliases["out_alias_x_next_after"] = bool(cu_x_next.data.ptr == t_x_next.data_ptr())
    aliases["out_alias_bar_x_after"] = bool(cu_bar_x.data.ptr == t_bar_x.data_ptr())
    aliases["out_alias_dot_next_after"] = bool(
        cu_dot_next.data.ptr == t_dot_next.data_ptr()
    )

    # read back from TORCH's OWN storage, not eagle's.
    hawk_x_next = t_x_next.cpu().numpy()
    hawk_bar_x = t_bar_x.cpu().numpy()
    hawk_dot_next = t_dot_next.cpu().numpy()

    # ---- torch reference: reverse via autograd.grad, forward via
    # torch.func.jvp, on the SAME (still un-mutated) t_x/t_v/t_t. ----
    def rhs_t(s, dt_):
        r2 = s[0] * s[0] + s[1] * s[1]
        inv_r3 = torch.rsqrt(r2 * r2 * r2)
        return torch.stack((s[2], s[3], -MU * s[0] * inv_r3, -MU * s[1] * inv_r3))

    def step_t(x, dt_):
        k1 = rhs_t(x, dt_)
        k2 = rhs_t(x + 0.5 * dt_ * k1, dt_)
        k3 = rhs_t(x + 0.5 * dt_ * k2, dt_)
        k4 = rhs_t(x + dt_ * k3, dt_)
        return x + (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    xg = t_x.clone().requires_grad_(True)
    out = step_t(xg, dt)
    torch_bar_x = (
        torch.autograd.grad(out, xg, grad_outputs=t_v)[0].detach().cpu().numpy()
    )
    with torch.no_grad():
        torch_x_next, torch_dot_next = torch.func.jvp(
            lambda xx: step_t(xx, dt), (t_x,), (t_t,)
        )
    torch_x_next = torch_x_next.detach().cpu().numpy()
    torch_dot_next = torch_dot_next.detach().cpu().numpy()

    primal_torch_rel_err = _max_rel_norm(hawk_x_next, torch_x_next)
    vjp_torch_rel_err = _max_rel_norm(hawk_bar_x, torch_bar_x)
    jvp_torch_rel_err = _max_rel_norm(hawk_dot_next, torch_dot_next)
    vjp_fd_rel_err, jvp_fd_rel_err = _fd_directional_errors(
        state.astype(np.float64), dt, v32, t32, hawk_bar_x, hawk_dot_next)

    # ---- BEHAVIOURAL check (as in zero_copy_interop.py's T-IN-CUDA-ALIAS row):
    # mutate the input IN PLACE, relaunch the SAME bound plan with no
    # rebind -- the vjp output must track the mutation, proving the kernel
    # reads THIS memory rather than a snapshot taken when cu_x was built. ----
    t_x.mul_(1.01)
    bound_v.launch()
    cp.cuda.runtime.deviceSynchronize()
    mutated_bar_x = t_bar_x.cpu().numpy()
    behavioural_ok = bool(
        not np.allclose(mutated_bar_x, hawk_bar_x, rtol=1e-6, atol=1e-9)
    )

    del cu_x, cu_v, cu_t, cu_x_next, cu_bar_x, cu_dot_next
    del t_x, t_v, t_t, t_x_next, t_bar_x, t_dot_next
    cp.get_default_memory_pool().free_all_blocks()

    return {
        "batch": batch, "dt": dt, "state": state, "v32": v32, "t32": t32,
        "bar_x": hawk_bar_x, "dot_next": hawk_dot_next,
        **aliases, "behavioural_ok": behavioural_ok,
        "primal_torch_rel_err": primal_torch_rel_err,
        "vjp_torch_rel_err": vjp_torch_rel_err,
        "jvp_torch_rel_err": jvp_torch_rel_err,
        "vjp_fd_rel_err": vjp_fd_rel_err, "jvp_fd_rel_err": jvp_fd_rel_err,
    }


# --------------------------------------------------------------------------- #
# Timing: hawk's vjp/jvp (and primal) against torch's, device only.
# --------------------------------------------------------------------------- #
#: Consecutive calls per timed sample (see :func:`_time_derivatives`).
TIMING_BLOCK = 10


def _time_derivatives(bundle, batch: int, dt: float, reps: int, seed: int = 0) -> dict:
    """Time hawk primal/vjp/jvp against torch primal/``autograd.grad``/
    ``torch.func.jvp`` -- build once, warm up once (excluded), then time
    ``reps`` INTERLEAVED reps and return the per-arm MEDIAN wall time (demo
    measurement, not a benchmark)."""
    import cupy as cp
    import eagle.exec as eexec
    import torch
    from eagle import plan as eplan

    names = _plane_names()
    state = seed_states(batch, seed)
    rng = np.random.default_rng(seed + 1)
    x32 = np.ascontiguousarray(state, dtype=np.float32)
    v32 = rng.normal(size=(4, batch)).astype(np.float32)
    t32 = rng.normal(size=(4, batch)).astype(np.float32)

    plan_p = eplan.plan(
        device_plugin(bundle, KERNEL_NAME), structure=eexec.DeviceKernel
    )
    plan_v = eplan.plan(device_plugin(bundle, VJP_NAME), structure=eexec.DeviceKernel)
    plan_j = eplan.plan(device_plugin(bundle, JVP_NAME), structure=eexec.DeviceKernel)

    cx, cv, ct = cp.asarray(x32), cp.asarray(v32), cp.asarray(t32)
    cx_next, cbar_x, cdot_next = cp.zeros_like(cx), cp.zeros_like(cx), cp.zeros_like(cx)

    bound_p = plan_p.bind(x=cx, dt=dt, x_next=cx_next)
    bound_v = plan_v.bind(
        **{"x": cx, "dt": dt, names.bar_out: cv, names.bar_wrt: cbar_x}
    )
    bound_j = plan_j.bind(
        **{"x": cx, "dt": dt, names.dot_wrt: ct, names.dot_out: cdot_next}
    )

    def hawk_primal():
        bound_p.launch()
        cp.cuda.runtime.deviceSynchronize()

    def hawk_vjp():
        bound_v.launch()
        cp.cuda.runtime.deviceSynchronize()

    def hawk_jvp():
        bound_j.launch()
        cp.cuda.runtime.deviceSynchronize()

    xt = torch.from_numpy(x32.copy()).cuda()
    vt = torch.from_numpy(v32.copy()).cuda()
    tt = torch.from_numpy(t32.copy()).cuda()

    def rhs_t(s, dt_):
        r2 = s[0] * s[0] + s[1] * s[1]
        inv_r3 = torch.rsqrt(r2 * r2 * r2)
        return torch.stack((s[2], s[3], -MU * s[0] * inv_r3, -MU * s[1] * inv_r3))

    def step_t(x, dt_):
        k1 = rhs_t(x, dt_)
        k2 = rhs_t(x + 0.5 * dt_ * k1, dt_)
        k3 = rhs_t(x + 0.5 * dt_ * k2, dt_)
        k4 = rhs_t(x + dt_ * k3, dt_)
        return x + (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def torch_primal():
        with torch.no_grad():
            step_t(xt, dt)
        torch.cuda.synchronize()

    def torch_vjp():
        xg = xt.clone().requires_grad_(True)
        out = step_t(xg, dt)
        torch.autograd.grad(out, xg, grad_outputs=vt)
        torch.cuda.synchronize()

    def torch_jvp():
        with torch.no_grad():
            torch.func.jvp(lambda xx: step_t(xx, dt), (xt,), (tt,))
        torch.cuda.synchronize()

    arms = {"hawk_primal": hawk_primal, "hawk_vjp": hawk_vjp, "hawk_jvp": hawk_jvp,
           "torch_primal": torch_primal, "torch_vjp": torch_vjp, "torch_jvp": torch_jvp}

    for fn in arms.values():
        fn()  # warm-up: build/first-launch/JIT cost, excluded

    # Each timed sample is a BLOCK of consecutive calls of one arm, divided by
    # the block size: a single call timed right after a switch of arm picks up
    # whatever the previous arm left behind (measured: hawk's primal read 3x
    # slow when timed alone straight after torch's jvp).
    times = {name: [] for name in arms}
    for _ in range(max(1, reps)):
        for name, fn in arms.items():
            fn()  # absorb the switch from the previous arm, untimed
            start = time.perf_counter()
            for _ in range(TIMING_BLOCK):
                fn()
            times[name].append((time.perf_counter() - start) / TIMING_BLOCK)

    wall = {name: statistics.median(v) for name, v in times.items()}
    cx = cv = ct = cx_next = cbar_x = cdot_next = xt = vt = tt = None
    cp.get_default_memory_pool().free_all_blocks()
    return wall


def run_sweep(dt: float = 0.01, seed: int = 0, build_dir=None, reps: int = 1) -> list:
    """The hawk-vs-torch primal/vjp/jvp wall-per-call table over
    :data:`SWEEP_BATCHES`, ``reps=1`` by default. Builds the device artifact
    once and reuses it across every batch. NOT run by the test."""
    import cupy as cp

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        where = pathlib.Path(build_dir or tmpdir) / "autodiff_vs_torch_sweep"
        bundle = build_artifact(where, ("cuda",))
        for batch in SWEEP_BATCHES:
            wall = _time_derivatives(bundle, batch, dt, reps, seed=seed)
            ratio_vjp = (
                wall["torch_vjp"] / wall["hawk_vjp"]
                if wall["hawk_vjp"] > 0 else float("nan")
            )
            ratio_jvp = (
                wall["torch_jvp"] / wall["hawk_jvp"]
                if wall["hawk_jvp"] > 0 else float("nan")
            )
            rows.append(
                {"batch": batch, **wall, "ratio_vjp": ratio_vjp, "ratio_jvp": ratio_jvp}
            )
            print(f"batch={batch:>8}  us/call: "
                 f"primal hawk={wall['hawk_primal'] * 1e6:7.2f} "
                 f"torch={wall['torch_primal'] * 1e6:9.2f}  "
                 f"vjp hawk={wall['hawk_vjp'] * 1e6:7.2f} "
                 f"torch={wall['torch_vjp'] * 1e6:10.2f} "
                 f"({ratio_vjp:7.1f}x)  "
                 f"jvp hawk={wall['hawk_jvp'] * 1e6:7.2f} "
                 f"torch={wall['torch_jvp'] * 1e6:10.2f} "
                 f"({ratio_jvp:7.1f}x)", flush=True)
            cp.get_default_memory_pool().free_all_blocks()
    return rows


# --------------------------------------------------------------------------- #
# The demo itself.
# --------------------------------------------------------------------------- #
def run_demo(batch: int | None = None, dt: float = 0.01, seed: int = 0,
            target: str = "host", reps: int = 3, build_dir=None,
            verbose: bool = True) -> dict:
    """Build, run and check. Returns every measured number."""
    if batch is None:
        batch = 65536 if target == "device" else 1024

    state = seed_states(batch, seed)
    targets = ("host",) if target == "host" else ("cuda",)
    with tempfile.TemporaryDirectory() as tmpdir:
        where = pathlib.Path(build_dir or tmpdir) / "autodiff_vs_torch"
        bundle = build_artifact(where, targets)
        manifest = check_manifest(bundle)
        if target == "host":
            arm = run_host(bundle, state, dt)
            timing = None
        else:
            arm = run_device(bundle, state, dt)
            timing = _time_derivatives(bundle, batch, dt, reps, seed=seed)

    results = {"target": target, "batch": batch, "dt": dt,
              "exec_targets": manifest["exec_targets"], "timing": timing, **arm}
    if verbose:
        report(results)

    assert results["vjp_fd_rel_err"] <= REL_TOL, (
        f"vjp vs float64 FD: max relative error {results['vjp_fd_rel_err']:.3e}, "
        f"tolerance {REL_TOL:.0e}")
    assert results["jvp_fd_rel_err"] <= REL_TOL, (
        f"jvp vs float64 FD: max relative error {results['jvp_fd_rel_err']:.3e}, "
        f"tolerance {REL_TOL:.0e}")
    if target == "host":
        if results["torch_available"]:
            assert results["vjp_torch_rel_err"] <= REL_TOL
            assert results["jvp_torch_rel_err"] <= REL_TOL
    else:
        for key in ("in_alias_x", "in_alias_v", "in_alias_t", "out_alias_x_next",
                   "out_alias_bar_x", "out_alias_dot_next", "out_alias_x_next_after",
                   "out_alias_bar_x_after", "out_alias_dot_next_after"):
            assert results[key], f"{key}: expected a zero-copy alias, measured a copy"
        assert results["behavioural_ok"], (
            "mutating x in place and relaunching (no rebind) did not change the "
            "vjp output -- the zero-copy check is proving nothing")
        assert results["primal_torch_rel_err"] <= REL_TOL, (
            f"primal vs torch: max relative error "
            f"{results['primal_torch_rel_err']:.3e}")
        assert results["vjp_torch_rel_err"] <= REL_TOL, (
            f"vjp vs torch.autograd.grad: max relative error "
            f"{results['vjp_torch_rel_err']:.3e}")
        assert results["jvp_torch_rel_err"] <= REL_TOL, (
            f"jvp vs torch.func.jvp: max relative error "
            f"{results['jvp_torch_rel_err']:.3e}")
    return results


def report(results: dict) -> None:
    """Print the agreement lines, the zero-copy lines and the timing table."""
    print(f"target            : {results['target']} "
          f"(manifest exec_targets {results['exec_targets']})")
    print(f"batch x dt        : {results['batch']} x {results['dt']}")
    print(f"vjp vs float64 FD ({FD_SAMPLES} samples): max relative error = "
          f"{results['vjp_fd_rel_err']:.3e} (tolerance {REL_TOL:.0e})")
    print(f"jvp vs float64 FD ({FD_SAMPLES} samples): max relative error = "
          f"{results['jvp_fd_rel_err']:.3e} (tolerance {REL_TOL:.0e})")
    if results["target"] == "host":
        if results["torch_available"]:
            print(f"vjp vs torch.autograd.grad (CPU): max relative error = "
                  f"{results['vjp_torch_rel_err']:.3e} (tolerance {REL_TOL:.0e})")
            print(f"jvp vs torch.func.jvp (CPU)     : max relative error = "
                  f"{results['jvp_torch_rel_err']:.3e} (tolerance {REL_TOL:.0e})")
        else:
            print("torch not installed: skipping the optional CPU cross-check")
    else:
        print(f"primal vs torch   : max relative error = "
              f"{results['primal_torch_rel_err']:.3e} (tolerance {REL_TOL:.0e})")
        print(f"vjp vs torch.autograd.grad: max relative error = "
              f"{results['vjp_torch_rel_err']:.3e} (tolerance {REL_TOL:.0e})")
        print(f"jvp vs torch.func.jvp     : max relative error = "
              f"{results['jvp_torch_rel_err']:.3e} (tolerance {REL_TOL:.0e})")
        print("zero-copy (torch CUDA tensor -> cupy view via eagle.to_cupy, "
              "every plane in and out):")
        for key in ("in_alias_x", "in_alias_v", "in_alias_t", "out_alias_x_next",
                   "out_alias_bar_x", "out_alias_dot_next"):
            print(f"  {key:<20} data_ptr equal = {results[key]}")
        print(f"  behavioural (mutate x in place + relaunch, no rebind): "
              f"vjp output changed = {results['behavioural_ok']}")
        if results["timing"] is not None:
            report_timing(results["timing"])


def report_timing(wall: dict) -> None:
    """Print the timing table (demo measurement, not a benchmark)."""
    print("timing (median of interleaved reps, warm-up excluded):")
    for kind, hawk_key, torch_key in (("primal", "hawk_primal", "torch_primal"),
                                      ("vjp   ", "hawk_vjp", "torch_vjp"),
                                      ("jvp   ", "hawk_jvp", "torch_jvp")):
        h, t = wall[hawk_key], wall[torch_key]
        ratio = t / h if h > 0 else float("nan")
        print(f"  {kind} hawk = {h * 1e6:9.2f} us   torch = {t * 1e6:10.2f} us   "
              f"torch/hawk = {ratio:7.1f}x")
    for side, primal_key, vjp_key, jvp_key in (
        ("hawk ", "hawk_primal", "hawk_vjp", "hawk_jvp"),
        ("torch", "torch_primal", "torch_vjp", "torch_jvp"),
    ):
        p = wall[primal_key]
        vjp_over = wall[vjp_key] / p if p > 0 else float("nan")
        jvp_over = wall[jvp_key] / p if p > 0 else float("nan")
        print(f"  {side} derivative overhead vs its own primal: "
              f"vjp = {vjp_over:.2f}x, jvp = {jvp_over:.2f}x")


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=int, default=None,
                        help="independent samples (default: 65536 for "
                             "--device, 1024 for --host)")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="RK4 step size (default: 0.01)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the states/cotangent/tangent (default: 0)")
    parser.add_argument("--reps", type=int, default=3,
                        help="interleaved timing reps for --device, median "
                             "reported (default: 3)")
    parser.add_argument("--build-dir", default=None,
                        help="where to publish the artifact (default: a "
                             "temporary directory)")
    parser.add_argument("--sweep", action="store_true",
                        help="print the hawk-vs-torch primal/vjp/jvp table "
                             "over a range of batch sizes, then exit "
                             "(reps=1 per point, well under two minutes)")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--host", dest="target", action="store_const",
                       const="host", help="run on CPU threads (the default)")
    where.add_argument("--device", dest="target", action="store_const",
                       const="device", help="run on the GPU, checked against "
                       "torch.autograd.grad / torch.func.jvp")
    parser.set_defaults(target="host")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Entry point."""
    args = parse_args(argv)
    if args.sweep:
        run_sweep(dt=args.dt, seed=args.seed, build_dir=args.build_dir)
        return 0
    run_demo(batch=args.batch, dt=args.dt, seed=args.seed, target=args.target,
             reps=args.reps, build_dir=args.build_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
