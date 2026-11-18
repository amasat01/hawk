# Add a target

A target is served by an OSPREY: one compiled library with its own facts, compilers and toolchains. hawk lists what
the libraries it finds serve in `hawk.targets.TARGETS` and asks the serving library for everything else, so a new
target is a new library (or a new row in one) and no hawk code. A test fails if `"cuda"` or `"host"` appears as a
string anywhere outside `hawk/targets`.

## What hawk needs to find it

An OSPREY is a folder `osprey_<name>/` holding the library `libosprey_<name>` and a manifest, `osprey.json`:

| Member | States |
|---|---|
| `name`, `version` | the library's name and version |
| `door` | the version of the door the library speaks |
| `targets` | one row per target the library serves |

A row of `targets` states `request` (the id users write: `host`, `cuda`), `target` (the library's own name for it, the
spelling its requests and answers use), `about` (a phrase), `device` (whether it builds for a device), `entry`
(`"host"` or `"device"`: how eagle enters a built kernel) and `format` (the manifest's word for its object).

hawk searches the folders that installed distributions declare under the entry-point group `hawk.ospreys` (the value is
a data path; nothing is imported) and the folders of the environment variable `HAWK_OSPREY_PATH`. A library that is
missing its manifest members, or that speaks another door, is refused by name.

## What the library answers

hawk calls the library's door and nothing else: the machine, the lowering of a kernel's request, one unit's build, the
link and the state of the store. The facts a number comes from (a fold threshold, a maximum dimension, a warp size), the
flags and the compilers all live inside the library; a missing one is refused by the library with its own code, and
hawk relays the refusal.

## The family side

A built target also reaches the other repos: `entry` and `format` are written to the manifest, and eagle runs the
kernel through an execution backend for that device kind. See
[Add an execution backend](https://amasat01.github.io/eagle/content/devguide/add_an_execution_backend.html).
