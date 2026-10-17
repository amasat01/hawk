#!/usr/bin/env python
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Card writer: hawk's vjp/jvp vs torch's ``autograd.grad``/``torch.func.jvp``.

Backs the example notebook's hawk-vs-torch speed-up statement with a
committed, reproducible measurement.

Real run (GPU, done separately from this script's ``--dry`` path, never on this check's own machine time):
wraps ``examples/autodiff_vs_torch.py``'s own ``run_sweep`` -- the hawk-vs-
torch wall-per-call sweep over ``SWEEP_BATCHES`` on a CUDA device artifact --
into the eagle card format (JSON + MD, script md5, hardware/driver/commit,
raw per-repetition timings, no absolute paths) and writes
``autodiff_vs_torch_card.json``/``.md`` next to this script.

``--dry`` (CPU, tiny batch, no GPU, no cupy required): exercises the SAME
card-writing mechanics -- facts, md5s, the no-absolute-path gate, JSON + MD
rendering -- with a host-only timing arm this script builds itself, because
the example's own timing path (``_time_derivatives``) is unconditionally
cupy/CUDA. Never authoritative (the card carries ``"dry": true``), and never
written to the committed filename -- see ``--out``.

Usage::

    python autodiff_vs_torch_card.py --dry
    python autodiff_vs_torch_card.py --device              # needs a visible GPU
"""

from __future__ import annotations

import argparse
import inspect
import pathlib
import sys
import tempfile
import time

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for _p in (_HERE, _REPO_ROOT / "examples"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _card_common as cc           # noqa: E402
import autodiff_vs_torch as example  # noqa: E402

CARD_NAME = "autodiff_vs_torch"
DRY_BATCH = 8
DRY_REPS = 3


# --------------------------------------------------------------------------- #
# The host-team twin of the example's own (GPU-only) ``_time_derivatives``:
# same arms, same warm-up/interleaving discipline, host team + torch-CPU
# instead of a CUDA device + torch-CUDA. Lives here, not in the example,
# because the example's kernel-migration scope is the traced bodies only.
# --------------------------------------------------------------------------- #
def _time_derivatives_host(bundle, batch: int, dt: float, reps: int, seed: int = 0) -> dict:
    import numpy as np
    import eagle.exec as eexec
    from eagle import plan as eplan

    names = example._plane_names()
    state = example.seed_states(batch, seed)
    rng = np.random.default_rng(seed + 1)
    x32 = np.ascontiguousarray(state, dtype=np.float32)
    v32 = rng.normal(size=(4, batch)).astype(np.float32)
    t32 = rng.normal(size=(4, batch)).astype(np.float32)

    plan_p = eplan.plan(example.host_plugin(bundle, example.KERNEL_NAME),
                        structure=eexec.HostTeam)
    plan_v = eplan.plan(example.host_plugin(bundle, example.VJP_NAME),
                        structure=eexec.HostTeam)
    plan_j = eplan.plan(example.host_plugin(bundle, example.JVP_NAME),
                        structure=eexec.HostTeam)

    x_next, bar_x, dot_next = np.zeros_like(x32), np.zeros_like(x32), np.zeros_like(x32)
    bound_p = plan_p.bind(x=x32, dt=dt, x_next=x_next)
    bound_v = plan_v.bind(**{"x": x32, "dt": dt, names.bar_out: v32, names.bar_wrt: bar_x})
    bound_j = plan_j.bind(**{"x": x32, "dt": dt, names.dot_wrt: t32, names.dot_out: dot_next})

    arms = {"hawk_primal": bound_p.launch, "hawk_vjp": bound_v.launch,
           "hawk_jvp": bound_j.launch}

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        xt = torch.from_numpy(x32.copy())
        vt = torch.from_numpy(v32.copy())
        tt = torch.from_numpy(t32.copy())

        def rhs_t(s, dt_):
            r2 = s[0] * s[0] + s[1] * s[1]
            inv_r3 = torch.rsqrt(r2 * r2 * r2)
            return torch.stack(
                (s[2], s[3], -example.MU * s[0] * inv_r3, -example.MU * s[1] * inv_r3))

        def step_t(x, dt_):
            k1 = rhs_t(x, dt_)
            k2 = rhs_t(x + 0.5 * dt_ * k1, dt_)
            k3 = rhs_t(x + 0.5 * dt_ * k2, dt_)
            k4 = rhs_t(x + dt_ * k3, dt_)
            return x + (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        def torch_primal():
            with torch.no_grad():
                step_t(xt, dt)

        def torch_vjp():
            xg = xt.clone().requires_grad_(True)
            out = step_t(xg, dt)
            torch.autograd.grad(out, xg, grad_outputs=vt)

        def torch_jvp():
            with torch.no_grad():
                torch.func.jvp(lambda xx: step_t(xx, dt), (xt,), (tt,))

        arms.update(torch_primal=torch_primal, torch_vjp=torch_vjp, torch_jvp=torch_jvp)

    for fn in arms.values():
        fn()  # warm-up: first-launch/JIT cost, excluded

    times = {name: [] for name in arms}
    for _ in range(max(1, reps)):
        for name, fn in arms.items():
            fn()  # absorb the switch from the previous arm, untimed
            start = time.perf_counter()
            fn()
            times[name].append(time.perf_counter() - start)
    return {name: cc.median_iqr(v) for name, v in times.items()}


def _facts() -> dict:
    device, peak = cc.device_facts()
    return {
        "device": device, "peak": peak, "software": cc.software(),
        "script_md5": cc.md5(__file__),
        "example_script_md5": cc.md5(example.__file__),
        "example_script": "examples/autodiff_vs_torch.py",
        "generated_utc": cc.utc_now_iso(),
    }


def _code() -> dict:
    return {
        "kernel_source": inspect.getsource(example.step_kernel),
        "derivative_source": inspect.getsource(example.derivative_kernels),
    }


def render_md(card: dict) -> str:
    lines = [f"# {CARD_NAME} card", ""]
    if card.get("dry"):
        lines += ["**DRY RUN -- not authoritative.** CPU, tiny batch "
                  f"({card['batch_or_batches']}), host-only arms.", ""]
    lines += [f"Schema: `{card['schema']}`  ·  generated {card['facts']['generated_utc']}",
             "", "## Facts", "",
             "```json", _json_block(card["facts"]), "```", "",
             "## Results", "", "```json", _json_block(card["results"]), "```", ""]
    return "\n".join(lines)


def _json_block(obj) -> str:
    import json

    return json.dumps(obj, indent=2, sort_keys=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry", action="store_true",
                   help="CPU, tiny batch, no GPU -- smoke-test the card writer, "
                        "never the real measurement")
    p.add_argument("--device", action="store_true",
                   help="real GPU run (needs a visible CUDA device): wraps run_sweep")
    p.add_argument("--out", default=None,
                   help="output directory (default: a scratch tmp dir for --dry, "
                        "this script's own directory for --device)")
    p.add_argument("--reps", type=int, default=None)
    args = p.parse_args(argv)

    if args.dry == args.device:
        p.error("pass exactly one of --dry / --device")

    facts = _facts()
    if args.dry:
        reps = args.reps or DRY_REPS
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = example.build_artifact(
                pathlib.Path(tmpdir) / "autodiff_vs_torch_card_dry", ("host",))
            results = _time_derivatives_host(bundle, DRY_BATCH, dt=0.01, reps=reps)
        card = {
            "schema": cc.SCHEMA, "name": CARD_NAME, "dry": True,
            "batch_or_batches": DRY_BATCH, "reps": reps,
            "facts": facts, "code": _code(), "results": results,
        }
        out_dir = pathlib.Path(args.out) if args.out else pathlib.Path(tempfile.mkdtemp(
            prefix="demo_card_dry_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{CARD_NAME}_card.DRY.json"
        md_path = out_dir / f"{CARD_NAME}_card.DRY.md"
        cc.write_card(card, json_path, render_md(card), md_path)
        print(f"[dry] wrote {json_path} and {md_path} (not committed)")
        return 0

    # --device: the real run, GPU only.
    reps = args.reps or 1
    rows = example.run_sweep(dt=0.01, seed=0, reps=reps)
    card = {
        "schema": cc.SCHEMA, "name": CARD_NAME, "dry": False,
        "batches": list(example.SWEEP_BATCHES), "reps": reps,
        "facts": facts, "code": _code(), "results": rows,
    }
    out_dir = pathlib.Path(args.out) if args.out else _HERE
    json_path, md_path = out_dir / f"{CARD_NAME}_card.json", out_dir / f"{CARD_NAME}_card.md"
    cc.write_card(card, json_path, render_md(card), md_path)
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
