"""Writes `broadcast_table_cases.json`: the dumps of three kernels over the lane axes a (3) and b (4), as the current front end
builds them, beside their requests' lane space and broadcast rows.

Run from a folder that holds no Python module named like the standard library, with the front end on the path:
`python regen_broadcast_table_cases.py`. The lane space and the broadcast rows are the expectations of the test that reads the
file (`build_over_libraries.rs`); this script keeps them as they are in the file and replaces only the dumps.

- `bcast_two`: `out = 2 * x + 1`, `x` a table of scalars the caller gives along `a` only (pattern [true, false]).
- `vec_two`: `out = v[0] * v[1] + 2 * v[2]`, `v` a table of three components given the same way.
- `shared_two`: `out = x + table[1]`, one table for every lane (no pattern).
- `own_axis_two`: `out = t[1]`, `t` a table with an axis of its own (`elem`), which is no lane axis: a row for it gathers nothing.
"""

import json
import pathlib

from hawk import Grid, Mutable, Scalar, Table, Vector, _broadcast, _ir, kernel

DATA = pathlib.Path(__file__).with_name("broadcast_table_cases.json")
GIVEN_ALONG_A = (True, False)


@kernel
def bcast_two(lane: Grid["a":3, "b":4], x: Scalar, out: Mutable[Scalar]):
    out = 2.0 * x + 1.0


@kernel
def vec_two(lane: Grid["a":3, "b":4], v: Vector[3], out: Mutable[Scalar]):
    out = v[0] * v[1] + 2.0 * v[2]


@kernel
def shared_two(table: Table[Scalar], x: Scalar, out: Mutable[Scalar]):
    out = x + table[1]


@kernel
def own_axis_two(lane: Grid["a":3, "b":4], t: Table["elem":3], out: Mutable[Scalar]):
    out = t[1]


def dumps() -> dict:
    """The dump of each kernel, the table kernels as the twin that reads their compact plane."""
    twin = lambda k, name: _broadcast.twin(k, {name: GIVEN_ALONG_A}).sinks
    return {
        "bcast_two": _ir.dump_text(twin(bcast_two, "x")),
        "vec_two": _ir.dump_text(twin(vec_two, "v")),
        "shared_two": _ir.dump_text(shared_two.sinks),
        "own_axis_two": _ir.dump_text(own_axis_two.sinks),
    }


if __name__ == "__main__":
    cases = json.loads(DATA.read_text())
    for name, text in dumps().items():
        cases[name]["dump"] = text
    DATA.write_text(json.dumps(cases, indent=1) + "\n")
