# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A scratch numeric interpreter over HAWK IR — TEST SCAFFOLD, not a product
module.

The finite-difference row needs a known answer to check a derivative rule
against: this walks a sink set with
numpy and returns each sink's value, so a primal kernel and the IR its VJP/JVP
transform produced can both be evaluated at the same point and compared with a
central difference. It is deliberately in ``hawk/tests/`` and deliberately
naive — no caching, no fusion, no CSE — because a fast evaluator would start to
share machinery with the emitter it is supposed to be independent of.
"""

from __future__ import annotations

import math

import numpy as np

from hawk.ir import (
    AccumWrite,
    At,
    Const,
    Dispatch,
    Leaf,
    Loop,
    LoopCount,
    LoopValue,
    MapreducePartial,
    Node,
    Primitive,
    SampleIndex,
    Select,
    TapeRead,
)

_BINARY = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
    "pow": lambda a, b: a ** b, "min": np.minimum, "max": np.maximum,
    "atan2": np.arctan2, "lt": np.less, "le": np.less_equal, "gt": np.greater,
    "ge": np.greater_equal, "eq": np.equal, "ne": np.not_equal,
    "land": np.logical_and, "lor": np.logical_or,
    "dot": lambda a, b: np.dot(a, b), "cross": np.cross,
    "mv": lambda a, b: a @ b, "mm": lambda a, b: a @ b, "outer": np.outer,
}
_UNARY = {
    "neg": np.negative, "abs": np.abs, "sqrt": np.sqrt,
    "rsqrt": lambda x: 1.0 / np.sqrt(x), "exp": np.exp, "log": np.log,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "tanh": np.tanh,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "lnot": np.logical_not, "norm": lambda v: np.linalg.norm(v),
    "sum": np.sum, "transpose": lambda m: m.T, "floor": np.floor,
    "exp2": np.exp2, "expm1": np.expm1, "log2": np.log2, "log10": np.log10,
    "log1p": np.log1p, "cbrt": np.cbrt, "sinh": np.sinh, "cosh": np.cosh,
    "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh,
    "ceil": np.ceil, "trunc": np.trunc, "rint": np.rint, "sign": np.sign,
    "round": lambda x: c_round(x),
    "erf": np.vectorize(math.erf, otypes=[float]),
    "erfc": np.vectorize(math.erfc, otypes=[float]),
    "isnan": np.isnan, "isinf": np.isinf, "isfinite": np.isfinite,
}
_BINARY.update({
    "hypot": np.hypot, "copysign": np.copysign, "fmod": np.fmod,
    "remainder": np.remainder,
    "fdim": lambda a, b: np.where(a > b, a - b,
                                  np.where(np.isnan(a) | np.isnan(b), np.nan, 0.0)),
})
_TERNARY = {
    "fma": lambda a, b, c: a * b + c,
    "clip": lambda x, lo, hi: np.minimum(np.maximum(x, lo), hi),
}


def c_round(x):
    """C ``round``: halves away from zero (``np.round`` rounds them to even)."""
    x = np.asarray(x, dtype=float)
    r = np.trunc(x)
    with np.errstate(invalid="ignore"):
        bump = np.where(np.abs(x - r) >= 0.5, np.sign(x), 0.0)
    return np.where(np.isfinite(x) & (bump != 0), r + bump, r)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


_QUAT = {
    "quat_mul": _quat_mul,
    "quat_conj": _quat_conj,
    "quat_recip": lambda q: _quat_conj(q) / float(np.dot(q, q)),
    "quat_rotate": lambda q, v: _quat_mul(_quat_mul(q, np.array([0.0, *v])),
                                          _quat_conj(q))[1:],
    "as_pure": lambda v: np.array([0.0, *v]),
    "as_vec3": lambda q: np.asarray(q)[1:],
}

#: `random_uniform`/`random_normal`: NOT aether's actual Philox
#: stream (the real thing is only reachable through a compiled artifact,
#: tested against aether's own golden oracle in `tests/test_random_op.py`'s
#: row) -- ANY deterministic function of `(seed, counter)` alone is a
#: legitimate stand-in HERE, because the one property the FD/VJP row needs is
#: that a draw stays FIXED while `x`/`y` perturb around it (`seed`/`counter`
#: never change during a finite difference), and `np.random.default_rng`
#: seeded from the two integers is exactly that: reproducible, and
#: independent of everything else in the primal.
_RANDOM = {
    "random_uniform": lambda seed, counter: float(
        np.random.default_rng((int(seed), int(counter))).uniform(0.0, 1.0)),
    "random_normal": lambda seed, counter: float(
        np.random.default_rng((int(seed), int(counter))).standard_normal()),
}


def evaluate(sinks, env: dict, lanes: int = 1, lane: int = 0) -> dict:
    """Evaluate ``sinks`` at ``env`` (leaf name -> value); returns sink values.

    ``lane`` is the sample this evaluation stands at, and it is what a
    :class:`~hawk.ir.nodes.SampleIndex` node reads: the interpreter walks ONE
    sample, so the lane index is a parameter of the walk rather than a plane in
    ``env``."""
    memo: dict[int, object] = {}
    env = {**env, "__lane__": lane}
    out: dict[str, object] = {}
    flat: list = []
    for sink in sinks:
        if isinstance(sink, Loop):
            # a lowered loop that COMMITS once per iteration: the
            # interpreter runs it and REPLAYS its body sinks, one commit per
            # iteration, which is exactly what the emitted scope does.
            for k, bindings in _iterations(sink, env, memo):
                for inner in sink.body_sinks:
                    flat.append((inner, dict(bindings)))
                del k
            continue
        flat.append((sink, None))
    for sink, bindings in flat:
        # a loop body sink is evaluated in ITS iteration's memo alone, for the
        # reason :func:`_iterations` states.
        memo_here = memo if bindings is None else dict(bindings)
        value = _value(sink.value, env, memo_here)
        if isinstance(sink, AccumWrite) and sink.index is not None:
            target = np.asarray(out.get(sink.name, np.zeros(lanes)), dtype=float)
            target[int(_value(sink.index, env, memo_here))] += value
            out[sink.name] = target
        elif isinstance(sink, MapreducePartial):
            out[sink.name] = value
        else:
            out[sink.name] = value
    return out


def _value(node: Node, env: dict, memo: dict):
    hit = memo.get(id(node))
    if hit is not None:
        return hit
    memo[id(node)] = value = _compute(node, env, memo)
    return value


def _iterations(loop, env: dict, memo: dict):
    """Run ``loop`` and yield ``(index value, per-iteration memo bindings)``.

    The bindings seed a FRESH memo per iteration with the loop's boundary values
    (computed once, outside — the emitter does the same), its index and its
    carried values, so a body node is recomputed each iteration exactly as the
    emitted scope recomputes it."""
    # a Loop OPERAND that is itself a loop is a forward loop this one tapes: it
    # is a statement, not a value, and it is reached through TapeRead instead.
    outside = {id(operand): _value(operand, env, memo)
               for operand in loop.operands if not isinstance(operand, Loop)}
    carry = [_value(init, env, memo) for init in loop.inits]
    ks = list(range(loop.start, loop.stop, loop.step))
    if loop.count is not None:
        # a derived loop of a loop with a `break`: the first (or, reversed, the
        # last) `count` iterations of the static range.
        n = int(outside[id(loop.count)])
        ks = ks[len(ks) - n:] if loop.count_tail else ks[:n]
    form = loop.exit_form
    count = 0
    for k in ks:
        bindings = dict(outside)
        bindings[id(loop.index)] = k
        for slot, placeholder in enumerate(loop.carries):
            bindings[id(placeholder)] = carry[slot]
        yield k, bindings
        # a FRESH memo per iteration, seeded ONLY with what crosses the
        # boundary. It may NOT inherit the caller's: a derived loop reuses the
        # primal body's own nodes (that is what keeps the two structurally
        # shared), so a value memoised for the reverse loop's iteration k would
        # be returned again while the forward loop is being replayed to fill its
        # tape — one scope's answer read in another's, which is a wrong number.
        per_iter = dict(bindings)
        if form is not None and bool(_value(loop.exit_cond, env, per_iter)):
            # Python's own `break`: the carried names keep their values AT it.
            carry = [_value(v, env, per_iter) for v in loop.exit_values]
            count += form == "tail"
            break
        carry = [_value(nxt, env, per_iter) for nxt in loop.nexts]
        count += 1
    memo[("loop_final", id(loop))] = carry
    memo[("loop_count", id(loop))] = count


def _loop_state(loop, env: dict, memo: dict):
    """``(final carried values, per-slot tape)`` for one lowered loop, memoised.

    The TAPE is the value each slot held at the START of every iteration — the
    same array the emitted forward loop writes when a reverse loop reads it
    through a :class:`~hawk.ir.loop_nodes.TapeRead`."""
    hit = memo.get(("loop", id(loop)))
    if hit is not None:
        return hit
    tape = [[] for _ in loop.carries]
    carry = None
    for _k, bindings in _iterations(loop, env, memo):
        for slot, placeholder in enumerate(loop.carries):
            tape[slot].append(bindings[id(placeholder)])
    carry = memo[("loop_final", id(loop))]
    memo[("loop", id(loop))] = state = (carry, tape)
    return state


def _compute(node: Node, env: dict, memo: dict):
    if isinstance(node, Const):
        return node.literal
    if isinstance(node, LoopValue):
        return _loop_state(node.loop, env, memo)[0][node.slot]
    if isinstance(node, LoopCount):
        _loop_state(node.loop, env, memo)
        return memo[("loop_count", id(node.loop))]
    if isinstance(node, TapeRead):
        loop = node.loop
        index = int(_value(node.index, env, memo))
        ordinal = (index - loop.start) // loop.step
        return _loop_state(loop, env, memo)[1][node.slot][ordinal]
    if isinstance(node, SampleIndex):
        return env["__lane__"]
    if isinstance(node, Primitive):
        # a primitive's FORWARD is its inlined subgraph; the boundary node adds
        # no arithmetic, which is exactly what the emitter renders too.
        return _value(node.forward, env, memo)
    if isinstance(node, Leaf):
        if node.name not in env:
            raise KeyError(f"no value bound for leaf {node.name!r} ({node.role})")
        return env[node.name]
    if isinstance(node, At):
        plane = _value(node.plane, env, memo)
        return np.asarray(plane)[int(_value(node.index, env, memo))]
    if isinstance(node, Select):
        cond, a, b = (_value(c, env, memo) for c in node.operands)
        return np.where(cond, a, b)
    if isinstance(node, Dispatch):
        # The serial oracle: the SAME clamp every emitted policy shares —
        # this interpreter does not model predicated/switch/segmented at all,
        # exactly as it models neither of `select`'s two aether spellings, so
        # a policy that lowers the clamp differently is a defect the RENDERED
        # artifact must be compared against, not this scratch evaluator.
        k = int(_value(node.selector, env, memo))
        k = min(max(k, 0), node.K - 1)
        return _value(node.branches[k], env, memo)
    args = [_value(c, env, memo) for c in node.operands]
    kind = node.kind
    if kind in _QUAT:
        return _QUAT[kind](*args)
    if kind == "component":
        return args[0][node.literal]
    if kind == "set_component_at":
        # `v` with ONE component replaced. A COPY, because the IR node
        # is a value: the vector it was given keeps the value it had.
        out = np.array(args[0], dtype=float)
        out[int(args[1])] = args[2]
        return out
    if kind == "component_at":
        # the RUNTIME-indexed component: the same read `component`
        # does, at a row this evaluation computed rather than one the type
        # carried. `int()` because the index node is i32/i64 and numpy hands
        # back a 0-d array.
        return np.asarray(args[0])[int(args[1])]
    if kind == "vec":
        conv = bool if node.ttype.dtype == "bool" else float
        flat = np.array([conv(a) for a in args])
        return flat if node.literal is None else flat.reshape(node.literal)
    if kind == "splat":
        return np.full(node.literal, float(args[0]))
    if kind in _UNARY:
        return _UNARY[kind](args[0])
    if kind in _RANDOM:
        return _RANDOM[kind](*args)
    if kind == "div" and node.ttype.dtype in ("i32", "i64"):
        # an integer quotient, because that is what the emitted `Int / Int` is;
        # numpy's `/` would give a float and the split-index identity
        # `lane - (lane / m) * m` would stop being the remainder.
        return int(args[0]) // int(args[1])
    if kind in _BINARY:
        return _BINARY[kind](*args)
    if kind in _TERNARY:
        return _TERNARY[kind](*args)
    raise NotImplementedError(f"the scratch evaluator has no rule for {kind!r}")
