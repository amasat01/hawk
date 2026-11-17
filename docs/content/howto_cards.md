# Build under other numbers

The lowering decides on numbers it ships in data documents: team sizes, thresholds, register and cost rows. A build that wants
other numbers (a tuning sweep, a test of a threshold, a kit for another card) states them in `build_spec`. The documents are
`build.json` and `arm.json`/`x86.json` for the host, `cuda_build.json` and `cuda.json` for a device.

## Change one number

`shipped_card` returns the document the library ships, parsed. Edit the number, then pass the whole document back:

```python
from hawk.artifact import emit_source
from hawk.targets import build_spec, shipped_card

card = shipped_card("build.json")
card["lanes"]["team_threads"] = 64            # the matrix order up to which the smaller team serves

spec = build_spec(("host",), cards={"build.json": card})
unit = emit_source(kernel, targets=("host",), spec=spec)
```

A document is a whole document: it replaces the shipped one for this build, so a key that is missing from it is refused. It is given
as a mapping; text is refused by name. hawk checks nothing in it: the library reads it with the reader of the shipped file, and a wrong
key or an out-of-range number is refused by name there.

## What changes

The same documents reach the library's facts answer and its request, so every decision that reads a number sees yours. A build under
other documents is another unit: its unit key differs from the build under the shipped documents, and a cached unit of one is never
taken for the other.
