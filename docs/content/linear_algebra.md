# Small linear algebra, per lane

Every lane solves its own small system, and the answer is differentiable.

```python
import hawk
import hawk.math as m
from hawk import Matrix, Mutable, Vector, Int32

@hawk.kernel
def newton_step(J: Matrix[3, 3], f: Vector[3], x: Vector[3],
                y: Mutable[Vector[3]], info: Mutable[Int32]):
    dx, info = m.solve(J, f)
    y = x - dx
```

The matrices are square, their size is fixed at trace time, and one lane owns
all of its matrices. Nothing is allocated, and the same kernel runs on the host,
on CUDA and, as plain C, on the embedded target.

## The names

Results come back as tuples, as in NumPy, with the status last.

| Name | Returns | What it does |
|---|---|---|
| `m.solve(A, b)` | `x, info` | `A x = b`, LU with partial pivoting; `b` is a vector or a matrix of right-hand sides |
| `m.lu(A)` | `lu, info` | the factor, to solve several times with `m.lu_solve(lu, b)` |
| `m.inv(A)` | `Ainv, info` | the inverse |
| `m.det(A)` | `d` | the determinant; zero for a singular matrix |
| `m.cholesky(A)` | `L, info` | `L L^T` is the symmetric part of `A`; `L` is the lower triangular matrix |
| `m.chol_from(L)` | `factor, info` | a lower triangular `L` you built, taken as the factor of `L L^T`, for `m.cho_solve` |
| `m.cho_solve(factor, b)` | `x` | `A x = b` from the factor that `m.cholesky` or `m.chol_from` returned |
| `m.eigh(A)` | `w, V, info` | eigenvalues ascending, eigenvectors as columns of `V` |
| `m.lstsq(J, r)` | `x, info` | least squares through the normal equations |
| `m.lm_step(J, r, mu)` | `dx` | the damped Gauss-Newton step, `-(J^T J + mu I)^-1 J^T r` |
| `m.mvn_sample(seed, counter, mean, cov)` | `y` | a draw from a multivariate normal, `mean + L z` |
| `m.eye(n)`, `m.diag(v)`, `m.diagonal(A)`, `m.trace(A)`, `m.tril(A)`, `m.mat(rows)` | | building and reading matrices |

`m.cho_solve` takes the factor that `m.cholesky` or `m.chol_from` gave you, not a
matrix you built: a plain matrix carries no status to fail on, so it is refused by
name. Pass a lower triangular matrix through `m.chol_from(L)` (below), factor with
`m.cholesky`, or solve with `m.solve`. An LU or eigen factor is read by its own
solves and fields, and is refused if you use it as a matrix.

## A factor you built

When the lower triangular factor `L` comes from your own formulas (a square-root
filter, a covariance you propagate as `L`), `m.chol_from(L)` takes it as the factor
of `L L^T` and `m.cho_solve` solves against it, with no factoring of its own:

```python
@hawk.kernel
def gain(L: Matrix[3, 3], r: Vector[3], k: Mutable[Vector[3]], info: Mutable[hawk.Int32]):
    factor, info = m.chol_from(L)
    k = m.cho_solve(factor, r)
```

Only the lower triangle is read. `info` comes from the diagonal alone: it is the first
column whose entry is zero, negative, NaN or infinite, and `0` when every one is
fine. A failed lane's solve is zeros, as for `m.cholesky`, and its derivatives are
zero. An entry below the diagonal is not checked, so a NaN there reaches the solve.
`vjp` and `jvp` are exact in `L` (its lower triangle; the part above carries no
derivative) and in `b`.

## Building a matrix in the kernel

`m.mat([[a, b], [c, d]])` builds a matrix from traced scalars, row by row. A list of
vectors gives the rows (`m.mat([u, v, w])`), or the columns with `columns=True`. A
ragged or empty argument is refused by name. `A[i, j]` reads one entry, and
`m.vec(u, v)` joins vectors. The result is an ordinary matrix: it feeds `m.solve`,
`m.cholesky` and `m.eigh`, and its derivative is the matrix of the derivatives.
Entries are written one by one, so a 22 by 22 Jacobian never sits in registers whole.
Write the entries in a plain Python helper, since a comprehension is not a kernel form:

```python
def jac(C, x, n):
    return m.mat([[3.0 * x[i] * x[i] + C[i, j] if i == j else C[i, j] for j in range(n)]
                  for i in range(n)])

@hawk.kernel
def newton_step(C: Matrix[22, 22], x: Vector[22], f: Vector[22], dx: Mutable[Vector[22]]):
    dx, info = m.solve(jac(C, x, 22), f)
```

## Past 32 by 32: a team per matrix

Up to `n = 32` a lane holds its matrix itself, one thread per lane. Bigger matrices (up to `4096 x 4096`) use the
same calls, `m.solve`, `m.lu`, `m.cholesky`, `m.eigh`, `m.lstsq` and the rest, and nothing in the kernel changes. hawk
splits the launch where such a call sits: a stage before it stores each lane's matrix in scratch, a team of threads
works one matrix in place, and a stage after reads the answer back. eagle owns the scratch and launches the stages in
order, in one captured graph, so run these kernels through `eagle.deploy(...).run` or `.bind`.

```python
@hawk.kernel
def fit(J: Matrix[71, 71], r: Vector[71], x: Mutable[Vector[71]], info: Mutable[Int32]):
    x, info = m.solve(J, r)            # 71 by 71 per lane: a team per matrix
```

Size alone picks the form: `n <= 32` is the one-thread form, bit for bit as before, and `n > 32` is a team. To
override it, pass `linalg_form=` to `hawk.build_bundle`: `"auto"` (the default), `"team"` (a team at every size) or
`"thread"` (refuses a matrix above 32 by name). The override is a build option; kernel code never changes.

Least squares with more than 32 unknowns (`m.lstsq` of an `m x n` matrix, `n > 32`) uses Householder QR, which does
not square the conditioning. A product summed over rows into a dense block, `hawk.sum(m.outer(j, j), over=...)`
and `hawk.sum(j * r, over=...)`, is built as one ordered block sum instead of one sink per entry.

What to expect:

- The team result is the same bits for any team size. Against the one-thread form (run both with `linalg_form=`) it
  agrees to rounding, not bit for bit.
- A team call inside a loop body, or under a `launch.box`, is refused by name. Iterate by replaying the plan.
- Scratch is memory, not registers: each lane needs about `n*n` elements of the kernel's width for the matrix, and an
  eigen-decomposition about three times that. A bind whose scratch is larger than the free device memory is refused,
  with the byte sum and the stage. 130 lanes of `1800 x 1800` doubles need 3.4 GB for the matrices alone; one lane of
  `800 x 800` needs 5 MB. Per-routine figures are on aether's team linear algebra page.

## `info`

`info` is an `Int32` per lane: zero when the factorisation held. A positive value
is the column that failed, numbered as LAPACK numbers it (the first zero pivot for
LU, the first minor that is not positive for Cholesky). For `m.eigh` it is a bit
mask: 1 means the sweeps did not converge, 2 means two neighbouring eigenvalues
are too close for their vectors to be told apart.

A failed lane returns zeros, never a half answer, and its derivatives are zero too.
The neighbouring lanes do not notice. If your kernel drops `info`
(`x, _ = m.solve(A, b)`), the C target's report lists the factorisation under
`linalg.status_dropped`, so a silent zero is at least a recorded one.

## Derivatives

`hawk.diff.vjp` and `jvp` differentiate all of the above, in loops and past a
`break` too. A solve is factored once, and the derivative reuses that factor.
The Cholesky factor itself is differentiable, so `m.mvn_sample` is too: the draw
is `mean + L z` with `z` fixed, which gives gradients with respect to the mean and
the covariance. Eigenvector derivatives are zeroed on a lane whose `info` has bit 2
set, because they are not defined there.

## Precision

The factorisations use plain multiply-add order in the kernel's scalar width, with
no compensation, so host, CUDA and the C unit agree on the bits. In `float32` the
answers are only as good as the conditioning allows: a Cholesky factor or its
gradient of a nearly singular matrix can be far off, or fail outright. Compare a
`float32` gradient against the `float64` one before trusting it, and note that
`m.lstsq` squares the conditioning.

## On the embedded C target

The C unit spells each of these as a looped helper in the same order of operations,
so a lane's values and `info` are bit-identical to the host build. The helpers
follow the MISRA C:2012 rules the rest of the target does, and the profile's
`max_stack_bytes` counts each factor at its true size, so a matrix that does not
fit is refused rather than overflowed. See {doc}`Embedded C target <embedded>`.

## Limits

- Sizes are static and capped at 4096. <!-- limit: la-size --> The bind refuses a larger matrix and names the cap; split the problem into blocks, or solve it outside the kernel.
- No complex, general (non-symmetric) eigenproblem, SVD or pseudo-inverse yet, and QR only as the least-squares solve past 32 unknowns. <!-- limit: la-more --> For these, factor in the caller and pass the result in as a plane; they are on the list.
- A matrix input binds with the exact lane shape. <!-- limit: la-broadcast --> Only vectors and scalars broadcast along a `Grid`; expand the matrix first.
