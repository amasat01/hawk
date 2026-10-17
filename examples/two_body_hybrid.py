# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Hybrid orbit demo: a hand-written physics kernel with a learned correction.

Every trajectory is a planar orbit around a point mass. The real world in this
demo also has drag, which the textbook two-body formula does not know about, so
a small tanh network is trained in torch to predict exactly the missing piece of
the acceleration. hawk then traces ONE kernel that performs a whole RK4 step
with that network evaluated inline, compiles it, and eagle runs the compiled
kernel over the whole batch -- on CPU threads, or as a replayed CUDA graph.

Run it with ``python examples/two_body_hybrid.py --host`` (CPU) or
``--device`` (GPU).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import tempfile
import time
from types import SimpleNamespace

import numpy as np

#: Gravitational parameter of the central body, in the demo's own units.
MU = 1.0
#: Drag coefficient of the "true" dynamics: the acceleration the two-body
#: formula misses is ``-DRAG_K * |v| * v``. Large enough that the orbit visibly
#: decays over a few hundred steps, small enough to stay a correction.
DRAG_K = 0.05
#: Width of the correction network's single hidden layer.
HIDDEN = 32
#: Acceptance (a): the compiled rollout must track a torch eager rollout of the
#: same arithmetic to this many absolute units.
MATCH_TOL = 1e-4
#: Acceptance (b): the corrected rollout must land this much closer to the true
#: final state than an uncorrected two-body rollout does.
ERROR_RATIO = 0.5
#: Substeps per demo step used by the float64 reference rollout.
REFERENCE_SUBSTEPS = 16
#: The kernel's name, and so the name of every file it is published under.
KERNEL_NAME = "two_body_hybrid_step"


# --------------------------------------------------------------------------- #
# The physics, in numpy: the model we can write down, the part we cannot.
# --------------------------------------------------------------------------- #
def seed_states(batch: int, seed: int) -> np.ndarray:
    """A ``(4, batch)`` float64 plane of perturbed near-circular orbit seeds.

    A column is one trajectory, ``(rx, ry, vx, vy)``. Radius and speed each
    carry a small random perturbation, so no two trajectories are the same.
    """
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, 2.0 * np.pi, batch)
    radius = 1.0 + 0.02 * rng.standard_normal(batch)
    speed = np.sqrt(MU / radius) * (1.0 + 0.02 * rng.standard_normal(batch))
    return np.stack([
        radius * np.cos(angle),
        radius * np.sin(angle),
        -speed * np.sin(angle),
        speed * np.cos(angle),
    ])


def two_body_rhs(state: np.ndarray) -> np.ndarray:
    """The textbook derivative of ``(rx, ry, vx, vy)``: gravity alone."""
    r3 = (state[0] ** 2 + state[1] ** 2) ** 1.5
    return np.stack([
        state[2],
        state[3],
        -MU * state[0] / r3,
        -MU * state[1] / r3,
    ])


def residual_acceleration(state: np.ndarray) -> np.ndarray:
    """The ``(2, batch)`` acceleration the two-body formula misses: drag."""
    speed = np.sqrt(state[2] ** 2 + state[3] ** 2)
    return np.stack([-DRAG_K * speed * state[2], -DRAG_K * speed * state[3]])


def true_rhs(state: np.ndarray) -> np.ndarray:
    """Gravity plus drag: the dynamics the training data is generated from."""
    out = two_body_rhs(state)
    out[2:] = out[2:] + residual_acceleration(state)
    return out


def rk4_rollout(rhs, state: np.ndarray, dt: float, steps: int,
                substeps: int = 1) -> np.ndarray:
    """Classical RK4 in float64 numpy, ``steps * substeps`` steps of ``dt``."""
    h = dt / substeps
    for _ in range(steps * substeps):
        k1 = rhs(state)
        k2 = rhs(state + (0.5 * h) * k1)
        k3 = rhs(state + (0.5 * h) * k2)
        k4 = rhs(state + h * k3)
        state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


# --------------------------------------------------------------------------- #
# The learned half: a 4 -> HIDDEN -> 2 tanh MLP fitted to the residual only.
# --------------------------------------------------------------------------- #
def train_correction(seed: int, hidden: int = HIDDEN, samples: int = 8192,
                     iterations: int = 1500) -> dict:
    """Fit the residual acceleration in torch on CPU; return float32 weights.

    The network never sees gravity: its target is ``true - two_body`` and
    nothing else, so the analytic half of the model stays analytic.
    """
    import torch

    torch.manual_seed(seed)
    states = seed_states(samples, seed + 1)
    inputs = torch.from_numpy(np.ascontiguousarray(states.T, dtype=np.float32))
    target = torch.from_numpy(
        np.ascontiguousarray(residual_acceleration(states).T, dtype=np.float32))

    model = torch.nn.Sequential(
        torch.nn.Linear(4, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 2),
    ).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # The last thirds of the fit are run at a smaller step, which is what takes
    # the residual from roughly fitted to accurately fitted.
    schedule = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, iterations // 3), gamma=0.2)
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), target)
        loss.backward()
        optimizer.step()
        schedule.step()

    with torch.no_grad():
        return {
            "w1": model[0].weight.numpy().copy(),
            "b1": model[0].bias.numpy().copy(),
            "w2": model[2].weight.numpy().copy(),
            "b2": model[2].bias.numpy().copy(),
            "loss": float(loss),
        }


def torch_rollout(state: np.ndarray, weights: dict, dt: float,
                  steps: int) -> np.ndarray:
    """The same hybrid RK4, written straightforwardly in eager float32 torch.

    This is the arm acceptance (a) compares against: same weights, same
    arithmetic, no compilation and no eagle.
    """
    import torch

    w1 = torch.from_numpy(weights["w1"])
    b1 = torch.from_numpy(weights["b1"])
    w2 = torch.from_numpy(weights["w2"])
    b2 = torch.from_numpy(weights["b2"])

    def rhs(x):
        r3 = (x[0] ** 2 + x[1] ** 2) ** 1.5
        hidden = torch.tanh(x.T @ w1.T + b1)
        correction = (hidden @ w2.T + b2).T
        return torch.stack([
            x[2],
            x[3],
            -MU * x[0] / r3 + correction[0],
            -MU * x[1] / r3 + correction[1],
        ])

    x = torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32))
    h = float(np.float32(dt))
    for _ in range(steps):
        k1 = rhs(x)
        k2 = rhs(x + (0.5 * h) * k1)
        k3 = rhs(x + (0.5 * h) * k2)
        k4 = rhs(x + h * k3)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x.numpy()


# --------------------------------------------------------------------------- #
# The kernel: ONE traced hawk kernel = one full RK4 step of the hybrid model.
# --------------------------------------------------------------------------- #
def step_kernel(hidden: int = HIDDEN):
    """Trace the RK4 step. The MLP is evaluated inline, four times per step.

    The weights are four ``Table`` arguments: one copy shared by every
    trajectory, not one per sample. The kernel stays a pure function of
    ``(state, weights, dt)`` and carries no state of its own.
    """
    import hawk
    import hawk.math as hm

    def hybrid_rhs(s, w1, b1, w2, b2):
        """Gravity plus the network's correction, as one traced expression."""
        r2 = s[0] * s[0] + s[1] * s[1]
        inv_r3 = hm.rsqrt(r2 * r2 * r2)
        act = []
        for h in range(hidden):
            pre = b1.at(h=h)
            for i in range(4):
                pre = pre + w1.at(h=h, i=i) * s[i]
            act.append(hm.tanh(pre))
        corr = []
        for o in range(2):
            out = b2.at(o=o)
            for h in range(hidden):
                out = out + w2.at(o=o, h=h) * act[h]
            corr.append(out)
        return hm.vec(
            s[2],
            s[3],
            -MU * s[0] * inv_r3 + corr[0],
            -MU * s[1] * inv_r3 + corr[1],
        )

    @hawk.kernel
    def two_body_hybrid_step(
        x: hawk.Vector[4],
        w1: hawk.Table["h":hidden, "i":4],
        b1: hawk.Table["h":hidden],
        w2: hawk.Table["o":2, "h":hidden],
        b2: hawk.Table["o":2],
        dt: hawk.Param,
        x_next: hawk.Mutable[hawk.Vector[4]],
    ):
        k1 = hybrid_rhs(x, w1, b1, w2, b2)
        k2 = hybrid_rhs(x + (0.5 * dt) * k1, w1, b1, w2, b2)
        k3 = hybrid_rhs(x + (0.5 * dt) * k2, w1, b1, w2, b2)
        k4 = hybrid_rhs(x + dt * k3, w1, b1, w2, b2)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return two_body_hybrid_step


def build_artifact(directory, targets) -> object:
    """Emit, compile and publish the step kernel into ``directory``."""
    import hawk

    return hawk.build([step_kernel()], pathlib.Path(directory),
                        mode="float32", targets=tuple(targets))


def check_manifest(bundle) -> dict:
    """Read the published manifest back and validate its shape through raptor.

    raptor is the protocol spine both halves agree on: hawk writes the
    document, eagle reads it, and this is the schema both are checked against.
    """
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
    """The plan-able view of the bundle's HOST entry.

    hawk publishes the shared object and names its entry symbol in the
    sidecar; ``hawk.artifact.plan_view`` projects that sidecar into the
    declaration ``eagle.plan`` reads, and the symbol address is what the plan
    is handed. hawk itself never imports eagle.
    """
    from hawk.artifact import plan_view

    library = ctypes.CDLL(str(pathlib.Path(bundle.directory) / f"{name}.so"))
    view = plan_view(_sidecar(bundle, name))
    entry = getattr(library, view.pop("host_entry"))
    return SimpleNamespace(
        host_entry=ctypes.cast(entry, ctypes.c_void_p).value,
        _keepalive=library, **view)


def device_plugin(bundle, name: str):
    """The plan-able view of the bundle's DEVICE entry.

    ``eagle.registry.load_manifest`` applies the manifest-level gates (schema
    version, execution axis, ABI tag), validates each sidecar and loads the
    compiled module; the loaded entry point is then attached to the same
    projected declaration the host path uses.
    """
    from eagle.registry import load_manifest
    from hawk.artifact import plan_view

    registry = load_manifest(pathlib.Path(bundle.manifest_path))
    loaded = registry[name]
    view = plan_view(_sidecar(bundle, name))
    view.pop("host_entry", None)
    return SimpleNamespace(
        device_function=loaded.fn.kernel.ptr,
        _keepalive=(registry, loaded), **view)


def weight_tables(weights: dict, xp) -> dict:
    """The four weight arrays as the ``Table`` buffers eagle binds.

    A Table is one buffer shared by every trajectory, so the network's weights
    are stored once however large the batch is.
    """
    return {name: xp.ascontiguousarray(xp.asarray(
        np.asarray(weights[name], dtype=np.float32))) for name in ("w1", "b1", "w2", "b2")}


def run_host(bundle, state: np.ndarray, weights: dict, dt: float,
             steps: int) -> tuple:
    """Run the rollout on CPU threads through ``eagle.exec.HostTeam``.

    Two plans are bound once, one per ping-pong direction, and then launched
    alternately: binding packs the argument block, launching does nothing but
    issue the work.
    """
    import eagle.exec as eexec
    from eagle import plan as eplan

    batch = state.shape[1]
    buffers = [np.ascontiguousarray(state, dtype=np.float32),
               np.zeros((4, batch), dtype=np.float32)]
    tables = weight_tables(weights, np)
    plan = eplan.plan(host_plugin(bundle, KERNEL_NAME), structure=eexec.HostTeam)
    bound = [plan.bind(x=buffers[0], x_next=buffers[1], dt=dt, **tables),
             plan.bind(x=buffers[1], x_next=buffers[0], dt=dt, **tables)]

    start = time.perf_counter()
    for step in range(steps):
        bound[step % 2].launch()
    elapsed = time.perf_counter() - start
    return buffers[steps % 2].copy(), elapsed


def run_device(bundle, state: np.ndarray, weights: dict, dt: float,
               steps: int) -> tuple:
    """Run the rollout on the GPU as ONE captured CUDA graph, replayed.

    The graph holds both ping-pong directions, so a single replay advances two
    steps and ``steps`` must be even. Nothing is re-packed and no kernel is
    re-launched from Python between steps: the whole rollout is replays.
    """
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan
    from eagle.pipeline import GraphPipeline

    if steps % 2:
        raise ValueError(
            f"--device replays a two-step graph, so --steps must be even; "
            f"got {steps}")

    batch = state.shape[1]
    buffers = [cp.asarray(np.ascontiguousarray(state, dtype=np.float32)),
               cp.zeros((4, batch), dtype=cp.float32)]
    tables = weight_tables(weights, cp)
    plan = eplan.plan(device_plugin(bundle, KERNEL_NAME),
                      structure=eexec.DeviceKernel)
    bound = [plan.bind(x=buffers[0], x_next=buffers[1], dt=dt, **tables),
             plan.bind(x=buffers[1], x_next=buffers[0], dt=dt, **tables)]

    pipeline = GraphPipeline()
    pipeline.add(lambda: bound[0].launch(), name="step_a")
    pipeline.add(lambda: bound[1].launch(), name="step_b")
    pipeline.build()

    # One replay first, so the timed loop measures a warm graph. ``launch``
    # replays on the pipeline's own stream and waits for it. The seed state is
    # then written back, because the warm-up steps are not part of the rollout.
    pipeline.launch(1)
    buffers[0][:] = cp.asarray(np.ascontiguousarray(state, dtype=np.float32))

    start = time.perf_counter()
    pipeline.launch(steps // 2)
    elapsed = time.perf_counter() - start
    return cp.asnumpy(buffers[0]), elapsed


# --------------------------------------------------------------------------- #
# The demo itself.
# --------------------------------------------------------------------------- #
def run_demo(steps: int = 200, batch: int = 1024, dt: float = 0.01,
             seed: int = 0, target: str = "host",
             build_dir=None, verbose: bool = True) -> dict:
    """Train, build, run and check. Returns every measured number."""
    state = seed_states(batch, seed)
    weights = train_correction(seed)

    targets = ("host",) if target == "host" else ("cuda",)
    with tempfile.TemporaryDirectory() as tmpdir:
        where = pathlib.Path(build_dir or tmpdir) / "two_body_hybrid"
        bundle = build_artifact(where, targets)
        manifest = check_manifest(bundle)
        if target == "host":
            final, elapsed = run_host(bundle, state, weights, dt, steps)
        else:
            final, elapsed = run_device(bundle, state, weights, dt, steps)

    start = time.perf_counter()
    eager = torch_rollout(state, weights, dt, steps)
    eager_elapsed = time.perf_counter() - start

    match = float(np.max(np.abs(final.astype(np.float64) - eager)))

    truth = rk4_rollout(true_rhs, state, dt, steps, REFERENCE_SUBSTEPS)
    plain = rk4_rollout(two_body_rhs, state, dt, steps)
    hybrid_error = float(np.mean(
        np.linalg.norm(final.astype(np.float64) - truth, axis=0)))
    plain_error = float(np.mean(np.linalg.norm(plain - truth, axis=0)))

    results = {
        "target": target, "steps": steps, "batch": batch, "dt": dt,
        "exec_targets": manifest["exec_targets"],
        "training_loss": weights["loss"],
        "match": match,
        "hybrid_error": hybrid_error,
        "two_body_error": plain_error,
        "compiled_wall_per_step": elapsed / steps,
        "eager_wall_per_step": eager_elapsed / steps,
    }
    if verbose:
        report(results)

    assert match <= MATCH_TOL, (
        f"(a) compiled rollout differs from the torch eager rollout by "
        f"{match:.3e}, tolerance {MATCH_TOL:.0e}")
    assert hybrid_error < ERROR_RATIO * plain_error, (
        f"(b) corrected error {hybrid_error:.3e} is not below "
        f"{ERROR_RATIO} x the two-body error {plain_error:.3e}")
    return results


def report(results: dict) -> None:
    """Print the acceptance numbers and the timing line."""
    print(f"target            : {results['target']} "
          f"(manifest exec_targets {results['exec_targets']})")
    print(f"batch x steps     : {results['batch']} x {results['steps']} "
          f"at dt = {results['dt']}")
    print(f"training loss     : {results['training_loss']:.3e}")
    print(f"(a) max |compiled - torch eager| : {results['match']:.3e} "
          f"(tolerance {MATCH_TOL:.0e})")
    print(f"(b) final-state error vs truth   : corrected "
          f"{results['hybrid_error']:.3e} vs two-body only "
          f"{results['two_body_error']:.3e}")
    print(f"(c) wall per step (demo measurement, not a benchmark): compiled "
          f"{results['compiled_wall_per_step'] * 1e3:.3f} ms, torch eager "
          f"{results['eager_wall_per_step'] * 1e3:.3f} ms")


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=200,
                        help="RK4 steps to roll out (default: 200)")
    parser.add_argument("--batch", type=int, default=1024,
                        help="independent trajectories (default: 1024)")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="step size (default: 0.01)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the orbits and the training (default: 0)")
    parser.add_argument("--build-dir", default=None,
                        help="where to publish the artifact (default: a "
                             "temporary directory)")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--host", dest="target", action="store_const",
                       const="host", help="run on CPU threads (the default)")
    where.add_argument("--device", dest="target", action="store_const",
                       const="device", help="run on the GPU as a captured graph")
    parser.set_defaults(target="host")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Entry point."""
    args = parse_args(argv)
    run_demo(steps=args.steps, batch=args.batch, dt=args.dt, seed=args.seed,
             target=args.target, build_dir=args.build_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
