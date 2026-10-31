# Derivatives: step_jvp, step_vjp, jump= and loops

Every kernel gets forward (`jvp`) and reverse (`vjp`) derivatives from `hawk.diff`. This page names the three options that need a word of explanation. The tutorials teach them in order: [backward](tutorials/04_gradients_backward), [forward](tutorials/05_gradients_forward) and [through branches and stops](tutorials/07_gradients_through_branches_and_stops).

## step_jvp and step_vjp

A kernel that updates its planes in place is one step of a rollout. `hawk.diff.step_vjp` is the `vjp` of one such step: it reads each plane's value before the step, takes `bar_<plane>_next` (the gradient with respect to the value after the step) and returns `bar_<plane>` (the gradient with respect to the value before it). `step_jvp` is the forward counterpart. To record a whole rollout and reverse it, see [tapes and stop rules](vocabulary/05_tapes_and_events).

`step_vjp(hawk.steps(k, K), steps_reverse="fused")` reverses all `K` steps of a launch in one launch. It re-runs the `K` steps from the state before the launch, so it needs no tape; state gradients match `K` calls of the per-step form bit for bit, parameter gradients to a few ULP. `"per_step"` is the default, and `jvp` of a `hawk.steps(k, K)` kernel is `hawk.steps(step_jvp(k), K)`.

## jump=

`@hawk.kernel(jump=<width>)` differentiates an `if`, a conditional expression or `where` that compares a differentiated value (`x < t`, `<=`, `>`, `>=`, combined with `land`, `lor` and `not`) as a smoothed switch. A sample within `sqrt(5) * width` of the switch passes on the jump between the two branches, weighted by an Epanechnikov kernel of standard deviation `width`; farther samples cost one comparison.

- `jump="h"` reads the width from a `Param[Scalar]`, so changing it needs no rebuild.
- `jump="auto"` picks each comparison's width from the batch.
- `jvp` and `vjp` run the other side of one drawn switch, chosen by the integer input `jump_seed` (default `0`), so a switched value that reaches another comparison, a `break` or a nonlinear function is differentiated correctly.
- Refused by name with `jump=`: `==` and `!=` on a differentiated value, a `break` that depends on one, a rounding or integer cast before the comparison, `step_vjp` and rollouts over launches, and a nested `for`.

A kernel without `jump=` derives as before. The glossary has a line for each term: [jump width](glossary), [step vjp](glossary).

## Loops and filled planes

A `for` differentiates in both directions, a nested `for` included: reverse mode recomputes the inner loop inside each reversed outer iteration (its carried values in a local array of the inner trip count), forward mode carries the inner tangents in the inner loop. Refused by name: an inner `break`, `jump=` through a nested loop, and, in reverse, an inner loop that gathers a plane being differentiated (read the plane in the outer loop, or flatten the two loops). A gather inside a `dispatch` branch of a loop is allowed, but not inside a `dispatch` nested in a `dispatch` branch.

A plane filled element by element (`out[0] = ...`, `out[3 * i + d] = ...` in loops, a store on both arms of an `if`) differentiates when the body stores every element before it reads one. A partial fill, a read of an element not yet stored (`out[j + 1]` in a loop over `j`), a store under a one-sided `if` or an index the trace cannot compute keeps the launch-start value, and the derivative is refused with "read before it is assigned".
