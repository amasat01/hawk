# Solve a small system in every lane

Give each lane its own matrix and call `m.solve`. The result is a tuple with the status last, so a singular matrix is a value you test, not an exception.

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

To solve several right-hand sides against one matrix, factor once with `m.lu(A)` and call `m.lu_solve(lu, b)`. For a symmetric positive matrix use `m.cholesky(A)` and `m.cho_solve(L, b)`.

The full list of names, their limits and the lanes they run on is the [linear algebra reference](linear_algebra).
