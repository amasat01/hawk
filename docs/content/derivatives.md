# Derivatives: step_jvp, step_vjp, jump= and loops

Every kernel gets forward (`jvp`) and reverse (`vjp`) derivatives from `hawk.diff`. This page names the three options that need a word of explanation. The tutorials teach them in order: [backward](tutorials/04_gradients_backward), [forward](tutorials/05_gradients_forward) and [through branches and stops](tutorials/07_gradients_through_branches_and_stops).

## step_jvp and step_vjp

A kernel that updates its planes in place is one step of a rollout. `hawk.diff.step_vjp` is the `vjp` of one such step: it reads each plane's value before the step, takes `bar_<plane>_next` (the gradient with respect to the value after the step) and returns `bar_<plane>` (the gradient with respect to the value before it). `step_jvp` is the forward counterpart. `hawk.diff.step_function(k)` is the step itself as a pure map: an ordinary kernel `<k>_step` that reads every carried plane as it was before the step and writes `<plane>_next`, so `jvp` and `vjp` differentiate it like any other kernel. To record a whole rollout and reverse it, see [tapes and stop rules](vocabulary/05_tapes_and_events).

`step_vjp(hawk.steps(k, K), steps_reverse="fused")` reverses all `K` steps of a launch in one launch. It re-runs the `K` steps from the state before the launch, so it needs no tape; state gradients match `K` calls of the per-step form bit for bit, parameter gradients to a few ULP. `"per_step"` is the default, and `jvp` of a `hawk.steps(k, K)` kernel is `hawk.steps(step_jvp(k), K)`.

## A recorded program: jvp and vjp of a builder

`hawk.diff.jvp` and `hawk.diff.vjp` also take a graph builder (a function decorated with `@hawk.Graph.builder`); a kernel behaves as before, and anything else that is callable is refused by name.

```python
tangent_step = hawk.diff.jvp(program)                        # a builder; call it as program, plus tangents={"x": dx}
graph = tangent_step(x, y, s, t, tangents={"x": dx})         # built; graph() replays the program with its tangents
graph.tangent_of(t)                                          # the tangent of any array of the program
gradients = hawk.diff.vjp(program)(x, y, s, t, cotangents={"t": seed}, wrt="x")   # replays once, then reverses it
```

The forward derivative re-records the program: a kernel launch becomes the launch that computes its outputs and tangents together (planes `dot_<plane>`), and a whole-plane node gains its tangent node. A scan or reduce sum, a reorder (the same order), a copy and a transform apply to the tangent planes; a product is `dA B` then `A dB` added; a one-axis minimum or maximum takes the tangent at the winning lane, and a running minimum or maximum the tangent at the winner of each prefix (zero where a prefix has none). A column with no winner, one of only NaN, has a zero derivative, forward and backward. A sort, a compaction, `all` and `any` have no tangent, and conditionals, loops and forks keep their structure and their tests. A tangent array is the caller's, given by the parameter's name in `tangents=`; every other float array of the program gets zeros. A minimum over a plane of two axes and a host step are refused by name.

The reverse derivative is the one mechanism `hawk.reverse` uses, so the two give the same bytes. Use `hawk.reverse(record, ...)` when you hold a graph you have already replayed; use `hawk.diff.vjp(builder)` to run the builder and get the gradients in one call, with `cotangents` and `wrt` naming the builder's array parameters. The builder must make `hawk.Graph(record=True)`.

## with_derivative

`hawk.diff.with_derivative(k, wrt=("x",))` is one kernel that computes `k`'s outputs and their forward derivatives along `wrt`, in one launch. Pass `dot_x` (the direction) beside `k`'s own inputs; it writes `k`'s outputs and `dot_<output>`. It is a plain kernel, so it is a step of an `eagle.Graph` builder like any other. A `hawk.steps(k, K)` kernel is refused: take `jvp` of it instead.

## jump=

`@hawk.kernel(jump=<width>)` differentiates an `if`, a conditional expression or `where` that compares a differentiated value (`x < t`, `<=`, `>`, `>=`, combined with `&`, `|`, `~` or `and`, `or`, `not`) as a smoothed switch. A sample within `sqrt(5) * width` of the switch passes on the jump between the two branches, weighted by an Epanechnikov kernel of standard deviation `width`; farther samples cost one comparison.

- `jump="h"` reads the width from a `Param[Scalar]`, so changing it needs no rebuild.
- `jump="auto"` picks each comparison's width from the batch.
- `jvp` and `vjp` run the other side of one drawn switch, chosen by the integer input `jump_seed` (default `0`), so a switched value that reaches another comparison, a `break` or a nonlinear function is differentiated correctly.
- Refused by name with `jump=`: `==` and `!=` on a differentiated value, a `break` that depends on one, a rounding or integer cast before the comparison, `step_vjp` and rollouts over launches, and a nested `for`.

A kernel without `jump=` derives as before. The glossary has a line for each term: [jump width](glossary), [step vjp](glossary).

## Loops and filled planes

A `for` differentiates in both directions, a nested `for` included: reverse mode recomputes the inner loop inside each reversed outer iteration (its carried values in a local array of the inner trip count), forward mode carries the inner tangents in the inner loop. Refused by name: an inner `break`, `jump=` through a nested loop, and, in reverse, an inner loop that gathers a plane being differentiated (read the plane in the outer loop, or flatten the two loops). A gather inside a `dispatch` branch of a loop is allowed, but not inside a `dispatch` nested in a `dispatch` branch.

A plane filled element by element (`out[0] = ...`, `out[3 * i + d] = ...` in loops, a store on both arms of an `if`) differentiates when the body stores every element before it reads one. A partial fill, a read of an element not yet stored (`out[j + 1]` in a loop over `j`), a store under a one-sided `if` or an index the trace cannot compute keeps the launch-start value, and the derivative is refused with "read before it is assigned".

## Groups of lanes

`g.scan(x)` is the inclusive running sum of `x` over the lanes of a group in rank order, beside `g.sum(x)`; `g.scan(x, exclusive=True)` is refused by name.

```python
@hawk.kernel
def running(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        y = g.scan(x)  # lane r gets x[0] + ... + x[r] of its group
```

`jvp` of a kernel with a `hawk.group` is a kernel with the same group: the same lanes, phases and barriers. Each floating scratch plane gets a tangent plane `dot_<name>` of the same size, stored and read wherever the scratch is. `g.sum` and a scan are linear, so their tangent is the sum or scan of the tangents. `g.min` and `g.max` take the tangent of the lane holding the result, the first in rank order among equals; a lane holding NaN never wins, and a group of NaN has a zero tangent. `g.any`, `g.all`, `g.rank`, `g.size` and integer scratch carry none. `vjp` of a kernel with a `hawk.group` is one kernel with the same group. Its body first recomputes the scratch stores, then runs the phases in reverse order, each separated by a barrier; each floating scratch plane gets an adjoint plane `bar_<name>` beside it. A scratch read by several lanes is gathered, not added to by every lane: the reading lane stores its contribution, and after a barrier the slot's own lane sums what the lanes that read it sent. A read at `rank + k` is sent back by the inverse of its edge rule: `wrap` and `zero` read the sent plane at `rank - k`, and `clamp` also adds to the first or last slot the lanes it folded onto it. `g.sum` gives every lane the group sum of the adjoints of its result, a scan gives lane `r` the sum of the adjoints of lanes `r` to the end, and `g.min` and `g.max` give the summed adjoint to the winner lane (first in rank order among equals; a lane holding NaN never wins, and a group of NaN gives zero). A keyed add or store is differentiated as outside a group. `g.any`, `g.all`, `g.rank`, `g.size` and integer scratch carry none. A scratch plane stored more than once, a scratch read at an index that is not `rank + k`, and a group beside a sink outside it are refused by name. `step_jvp`, `step_vjp` and `with_derivative` of a kernel with a group are refused by name.

`hawk.diff.CELL_PLANE` is the name of the per-cell parameter plane a promoted kernel reads, beside `ADJOINT_PREFIX` and `TANGENT_PREFIX`. `hawk.runtime.check_scatter_status(name, team, status, cells)` refuses, by name, a run whose ordered scatter dropped a key outside its plane; `team` is the sidecar's team record and `status` the status plane after the run.
