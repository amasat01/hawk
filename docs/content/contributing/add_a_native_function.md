# Add a native function end to end

A contributor how-to, not a newcomer tutorial: how a brand-new math
function — call it `foo` — becomes available to every hawk kernel, with a
host implementation, a device implementation, and a derivative rule. Six
files across two repositories. Cross-linked from
[aether's own "adding a function" tutorial](https://amasat01.github.io/aether/).

No public seam skips this today: [`hawk.ext.primitive`](../vocabulary/03_your_own_derivative)
only *composes* existing ops, and [`hawk.raw_device`](../vocabulary/when_hawk_cant_spell_it)
cannot be differentiated. A genuinely new native op is this six-step edit.

## 1. aether: the host/device dispatch

`aether/math/*.h` (see `special.h` for the pattern): one macro line adds
`foo` to `aether::math`, dispatching to libm on the host and the matching
CUDA builtin on the device:

```cpp
AETHER_MATH_UNARY(foo, foof, ::foo, std::foo(x))
```

Add a `cwiseFoo` alongside it if a rank &ge; 1 value should vectorise
without scalarising first (see `_CWISE1` in step 3).

## 2. hawk: declare the op kind

`hawk/ir/ops.py` — add `"foo"` to `UNARY_MATH` (which folds into
`OP_KINDS`, the vocabulary the walk and the derivative-rule table both key
on).

## 3. hawk: the aether spelling

`hawk/emit/spelling.py` — add a `"foo": "foo"` entry to `_MATH1` (the
rank-0 `aether::math::foo` spelling), and to `_CWISE1` too if step 1 added
a `cwiseFoo`.

## 4. hawk: the derivative rule

`hawk/diff/rules.py` — add `foo`'s rule to the table (`erf`'s entry is a
model: `"erf": lambda n, x: _mul(_c(_TWO_OVER_SQRT_PI), _o("exp", _neg(_mul(x, x))))`).
This is the rule every kernel calling `foo` gets automatically — the whole
point of a native op over a primitive is that *no caller* has to supply one.

## 5. hawk: the authoring name

`hawk/trace/value.py` — `foo = _lifted("foo", "<one-line docstring>")`
exports it as a free function; `hawk/math.py` — add `"foo"` to `__all__`
so `from hawk.math import foo` works.

## 6. Regenerate the sealed payload, and let the gate prove it

```bash
python -m aether_dsc.seal
```

reseals the aether payload hawk compiles against. `tests/test_math_namespace.py`
then traces and emits *every* name in `hawk.math.__all__` — `foo` lands in
its coverage automatically, and the gate fails loudly if any of the five
steps above was missed (traces but has no aether spelling, has a spelling
but no rule, and so on).

## Next

Back to [when hawk can't spell it](../vocabulary/when_hawk_cant_spell_it) for the
decision between a primitive, `raw_device`, and this path.
