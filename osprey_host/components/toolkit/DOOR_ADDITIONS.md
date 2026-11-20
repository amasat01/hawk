# Door additions for the tool side (crate `components/toolkit`)

Authority: handover to the lead | Consumed by: the lead, when wiring the toolkit into each OSPREY after the split | Archive condition: the four
entries are in `osprey.h`, the loader, the binding and both OSPREYs, and the gate is green.

The call shapes a CALLER uses are those of `include/osprey.h` (the caller states a target, options, a level and a cache root, and hands the machine answer back as given); the requests described below are what the library composes from them for the toolkit. The engine is `toolkit::Toolkit` (crate `components/toolkit`, dependency `rcore` only, which is already in the tree). An OSPREY makes
one `Toolkit` from its tool table (JSON), its embedded header trees and the process environment, and its four door entries call
`machine`, `build`, `link` and `state` on it. Every difference between OSPREYs is in the three inputs.

## 1. C declarations (header `osprey.h`; bumps `OB_ABI` by one)

Four fields are appended to `ob_table`, in this order, after `card`:

```c
    int (*machine)(rx_ctx*, const char*, char*, size_t, size_t*);
    int (*build)(rx_ctx*, const char*, size_t, const char*, char*, size_t, size_t*);
    int (*link)(rx_ctx*, const char*, size_t, const char*, char*, size_t, size_t*);
    int (*state)(rx_ctx*, const char*, const char*, char*, size_t, size_t*);
```

and four loader functions, written like `ob_set_text` (JSON in and out, `RX_TRUNC` when `cap` is too small with `len` filled in; a
refusal reads in `rx_last_error(ctx)`, its text starting with the refusal kind name and a colon):

```c
/* The machine the OSPREY's tools run on. ask is JSON: {"key_terms": [text...], "host": bool, "root": <store root>?, "device_arch":
   <pin>?, "reserved_names": {"prelude": <the unit's include block>}?, "witnesses": [<witness>...]?}. key_terms are the texts the key
   must cover beyond what the machine states (the OSPREY's facts identity and flags); host asks for the host's own facts
   (architecture, vector instruction set, cache sizes, a CPU digest); root is the store root, needed by the facts that serve a header
   tree or keep a verdict (reserved_names, witnesses); device_arch is a pin. A witness is {"name", "source", "source_suffix"?,
   "compile": [argv template], "link": [argv template], "exit_verdicts": {"<exit code>": "<verdict word>"}, "timeout": seconds}.
   Answer: {"key": <opaque>, "stated": {"tools": {name: {"path", "identity", <probe facts>}}, "headers": {tree: digest}, "host":
   {...}?, "runtime": {"library", "version", "archs"}? (when a runtime compile library stands in for a tool), "device": {"arch",
   "why", "gpu_arch", "driver_release", "driver_encoded", "toolkit_archs"}? (when the table has a device member), "reserved_names":
   [sorted names]? (when asked and every tool is found), "witnesses": {name: {"verdict": word | "unrun", "why"?, "hit"}}? (when
   asked and every tool is found)}, "tools_absent": null | "<reason that names every missing tool, the rules tried in order and how
   to fix it>"}. The machine is ALWAYS stated: a missing tool is not another answer shape. The tools found, the header digests, the
   host facts and the device policy (with the fallback arch when the toolkit's archs are unknown) are stated; emitting source needs
   no compiler. ob_build, ob_link and the witnesses refuse with the name tool_absent and the same reason when a tool they need is
   missing. A pin the toolkit cannot serve, a reserved-name
   preprocess that fails and a missing root are refusals. */
int ob_machine(rx_ctx* ctx, ob_backend* lib, const char* ask, char* buf, size_t cap, size_t* len);

/* Builds one unit: runs the OSPREY's tool on the text under the instructions the request states, through the store under root (a
   folder; created when missing). unit is JSON (section 3). Answer: {"files": {"output": <path>}, "key": <hex>, "hit": bool,
   "figures": {...}, "report": {"argv": [...], "seconds": n, "elf": bool}}. A
   second call for the same unit on the same machine runs no tool and answers "hit": true. */
int ob_build(rx_ctx* ctx, ob_backend* lib, const char* unit, size_t n_unit, const char* root, char* buf,
    size_t cap, size_t* len);

/* Links the library of a set: parts are the answers of ob_build, in order; the set text is the "set" member (a unit request, section
   3, compiled with its own template and linked last); each entry of "images" ({"image": <path>, "symbol", "size_symbol"}) is assembled into an object by the
   request's "embed" recipe and linked too. request is JSON: {"name": <file name>, "parts": [<ob_build answer>...],
   "set": <unit>?, "link": {"linux": [argv...], "macos": [argv...]}, "embed": <recipe>?, "as": {...}?, "key_terms": [...]?}.
   Answer: {"library": <path>, "key": <hex>, "hit": bool, "report": {"argv": [...], "seconds": n}}. */
int ob_link(rx_ctx* ctx, ob_backend* lib, const char* request, size_t n_request, const char* root, char* buf,
    size_t cap, size_t* len);

/* "stats" -> {"hits", "misses", "tool_runs"} of this process; "reset" -> {} (zeroes them and forgets memoised tool probes; the
   store on disk stays). Any other op is refused by name. */
int ob_state(rx_ctx* ctx, ob_backend* lib, const char* op, const char* root, char* buf, size_t cap, size_t* len);
```

The door's `set_text` argument of the `ob_link` of the spec is the request's `"set"` member: it is a build request (name, text,
template), so the set text is compiled and cached like any unit.

`loader.c`: each function is the `ob_set_text` pattern: `inner = table->ctx_new()`, call the table entry, `forward` a refusal,
`ctx_free`. `ob_build` and `ob_link` take a length with the JSON because the request carries source text; `ob_machine` and `ob_state`
take NUL-terminated text. The binding's generator adds the four names to its list, as for the existing `ob_*` functions.

## 2. Glue in each OSPREY (`src/lib.rs`, the same text in both)

```rust
mod trees { include!(concat!(env!("OUT_DIR"), "/trees.rs")); }   // pub const TREES, from build.rs (section 4)
const TOOL_TABLE: &str = include_str!("../facts/tools.json");    // the OSPREY's own table (section 5)

static TOOLKIT: OnceLock<Result<Toolkit, Refusal>> = OnceLock::new();

fn toolkit() -> Result<&'static Toolkit, Refusal> {
    TOOLKIT
        .get_or_init(|| Toolkit::new(TOOL_TABLE, trees::TREES, Env::process()))
        .as_ref()
        .map_err(Clone::clone)
}

/// Runs one of the toolkit's functions behind the door's calling convention.
fn answer(ctx: *mut Ctx, buf: *mut c_char, cap: usize, len: *mut usize, call: impl FnOnce(&Toolkit) -> Result<String, Refusal>) -> c_int {
    if ctx.is_null() || len.is_null() { return RX_BADARG; }
    let done = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| toolkit().and_then(call)))
        .unwrap_or_else(|_| Err(Refusal::new(toolkit::Kind::Io, "panic in the tool side")));
    match done {
        Ok(text) => match put(buf, cap, len, &text) { Ok(()) => RX_OK, Err(Fault::Truncated) => RX_TRUNC, Err(_) => RX_BADARG },
        Err(refusal) => {
            // SAFETY: `ctx` is the live context the caller holds.
            unsafe { &mut *ctx }.set_err(&refusal.to_string());     // "<kind name>: <message>"
            if refusal.kind == toolkit::Kind::BadRequest { RX_BADARG } else { RX_REFUSED }
        }
    }
}

extern "C" fn machine(ctx: *mut Ctx, ask: *const c_char, buf: *mut c_char, cap: usize, len: *mut usize) -> c_int {
    answer(ctx, buf, cap, len, |tk| tk.machine(text_arg(ask, "ask").map_err(|e| Refusal::bad(e))?))
}
extern "C" fn build(ctx: *mut Ctx, unit: *const c_char, n: usize, root: *const c_char, buf: *mut c_char, cap: usize, len: *mut usize) -> c_int {
    answer(ctx, buf, cap, len, |tk| tk.build(bytes_arg(unit, n)?, Path::new(text_arg(root, "root").map_err(|e| Refusal::bad(e))?)))
}
// link: the same with tk.link(bytes_arg(request, n)?, root); state: tk.state(text_arg(op)?, root)
```

`bytes_arg` is the existing `set_text` slice-to-UTF-8 step made a function (`std::slice::from_raw_parts` over `n` bytes, refused when
not UTF-8). The four entries are listed in the `OspreyV1` table the library returns. `Refusal` is `Clone`, so the one-time failure of
`Toolkit::new` (a table that does not parse) is the answer of every call.

## 3. The unit request (`ob_build`) and the places a template may use

JSON members: `name` (names the files), `text`, `template` (the argv, as the facts state it), `lists` ({"{defines}": [..]}: a list
place that is a whole element of the template stands for one element per item), `extra` (appended after the template), `key_terms`
(texts the key covers: target, mode, profile, optimisation level, C library, CPU digest; the library's answers state them),
`keep_source` (publish the source text beside the output; the answer's `files.source`), `source_suffix` (default `.cpp`), `output_suffix` (default `.o`), `figures` (the grammar of the figures to read from the tool's
output; `ptxas_info` is the one that exists), `places` ({"{include:facts}": "<text>"} for a place the OSPREY fills itself),
`as` ({"compiler": "host_compiler"}: the place `{compiler}` is the tool the table calls `host_compiler`; a link of a device OSPREY
needs it), and `runtime` ({"options": [text with places...], "target": <image kind>, "headers": [tree names]}): the in-process compile used when the machine has no offline compiler (`stated.runtime` present); the unit's `template` is used when the tool exists. `figures` also knows `image_resources`: the register and frame figures read off an ELF image (no tool, no device).

Places: `{<tool>}` for each tool of the table, `{source}`, `{output}` / `{object}`, `{depfile}`, `{include:<tree>}` (the served folder
of the embedded tree of that name) and the request's own. A place nobody fills is refused by name (`bad_request`). The key covers the
text, the template as given, the lists, the extras, the key terms, the identity of each tool the template names and the digest of
each tree it includes; it does not cover the paths the argv is filled with.

The header shape of the facts maps to trees: `roots` -> two trees named `aether` and `eagle`; `served` -> one tree named `served`;
`{include:facts}` -> a tree named `facts`; `{include:door}` -> a tree named `door`. The OSPREY embeds the trees its templates name.

## 4. Embedding the trees (`build.rs` of each OSPREY)

```toml
[build-dependencies]
toolkit = { path = "../toolkit" }          # components/toolkit
```

```rust
fn main() {
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("trees.rs");
    toolkit::embed::write_table(
        &[("aether", "OSPREY_AETHER_HEADERS"), ("eagle", "OSPREY_EAGLE_HEADERS"), ("facts", "OSPREY_FACTS_HEADERS")],
        &out,
    ).unwrap_or_else(|e| panic!("{e}"));
}
```

The environment variables name the folders at ITS build; the build refuses by name when one is unset. The generated source holds
the bytes (`include_bytes!`); a served tree is written once under `<root>/headers/<digest>` and the digest is a key term.

## 5. The tool table (`facts/tools.json` of each OSPREY; data only)

`{"tools": {name: {"rules": [...], "fix": "<how to put the tool in place>", "identity_args": ["--version"], "probes": [...]}},
"process_env": {...}, "limits": {...}}`; tools are found in the table's order and a tool may use an earlier one.

- Rules, tried in order: `{"env": "VAR", "required": bool, "first_word": bool}` (required: an explicit override, refuses when it does
  not resolve; first_word: the first word of the value, as `$CXX` is), `{"path": "name"}` (on `PATH`), `{"env_flag": {"var", "flags":
  [...]}}` (the value after a flag inside a variable that holds flags), `{"tool": "other"}`; any rule may add `"dir_child": "g++"`
  (a folder resolves to that file inside it).
- Probes: `{"name", "args", "after": "<marker>", "as": "version" | "int_list" | "gcc_family"}`; the answers are stated beside the
  tool's `path` and `identity`. `gcc_family` reads `-dM -E` output and states `{"gcc": bool, "gcc_release": "major.minor" | null}`.
- `process_env`: `strip_flags` {vars, flags}, `remove` [vars], `path_prepend` [tools], `root_of` {VAR: tool} (the folder above the
  tool's folder).
- `limits`: `address_cap`, `source_budget`, `timeout`, each `{"env": "VAR", "default": n}`; 0 disables. Without the member the
  defaults are 12 GiB, 8 MiB and 1800 s, and no variable is read.

cpu, as the Python states it today: `compiler` = rules `[{"env":"HAWK_CXX","required":true},{"env":"CXX","first_word":true},
{"path":"g++"}]`, probes `[{"name":"family","args":["-dM","-E","-x","c++","/dev/null"],"as":"gcc_family"}]`; limits
`HAWK_COMPILE_ADDRESS_CAP`, `HAWK_COMPILE_SOURCE_BUDGET`, `HAWK_COMPILE_TIMEOUT`. cuda: `compiler` = `[{"env":"HAWK_NVCC","required":
true},{"path":"nvcc"}]` with probes `release` (`--version`, after `release `, version) and `archs` (`--list-gpu-arch`, after
`compute_`, int_list); `host_compiler` = `[{"env":"HAWK_NVCC_CCBIN","dir_child":"g++"},{"env_flag":{"var":"NVCC_PREPEND_FLAGS","flags":
["-ccbin","--compiler-bindir"]},"dir_child":"g++"},{"env_flag":{"var":"NVCC_APPEND_FLAGS","flags":["-ccbin","--compiler-bindir"]},
"dir_child":"g++"},{"env":"NVCC_CCBIN","dir_child":"g++"},{"path":"g++"}]`; `process_env`: `strip_flags` over both flag variables with
the two ccbin flags, `remove` [`NVCC_CCBIN`], `path_prepend` [`compiler`], `root_of` {`CUDA_PATH`: `compiler`}.

## 6. The device, reserved-name, witness and runtime members of the table

```
"device": {
  "driver": {"libraries": [names tried in order], "calls": {"init", "version", "current_device", "device_by_ordinal", "attribute"},
             "version_decode": [major divisor, minor divisor], "attributes": {"major": <n>, "minor": <n>}, "arch_format": "sm_{major}{minor}"},
  "arch":   {"pin_env": "VAR", "fallback": "compute_61", "portable_min": 60, "toolkit": {"tool": "compiler", "probe": "archs"}}},
"reserved_names": {"tool": "compiler", "args": [argv after the tool: places {source}, {include:<tree>}], "system_headers": [names]},
"runtime_compile": {"replaces": "compiler", "libraries": [names], "ok_status": 0, "force": {"env": "VAR", "tool": <word>, "runtime": <word>},
  "calls": {"version", "arch_count", "arch_list", "create", "compile", "log_size", "log", "destroy",
            "images": {<kind>: {"size": fn, "get": fn, "text": bool}}}}
```
- Driver call signatures are fixed by the mechanism: `init(unsigned)`, `version(int*)`, `current_device(int*)`,
  `device_by_ordinal(int*, int)`, `attribute(int*, int attribute, int device)`, all returning `int` (0 = success). The arch is the
  pin of the ask, else the pin in `pin_env`, else the running GPU; with none, the first toolkit arch at least `portable_min` as a
  virtual `compute_<n>` (or `fallback` when the toolkit's archs are unknown). An arch above the toolkit's highest is that highest,
  virtual; below its lowest is refused naming who asked. The toolkit's archs come from the runtime library when it is in use, else
  from the probe named in `device.arch.toolkit`.
- Runtime call signatures: `create(program*, source, name, header count, header texts, header names)`, `compile(program, option
  count, options)`, `log_size(program, size_t*)`, `log(program, char*)`, image `size(program, size_t*)` and `get(program, char*)`,
  `destroy(program*)`, `version(int*, int*)`, `arch_count(int*)`, `arch_list(int*)`; headers are the files of the trees the request
  names. A tool replaced by the runtime library is not listed in `stated.tools`; a build whose request has a `runtime` member then
  compiles in process, through the same store and key, with no address-space cap or timeout (it is in this process).
  `force.env` lets one variable choose: its value equal to `tool` never uses the library, equal to `runtime` never looks for the
  tool, any other value is refused.
- The flag variables of the environment (`env_flag` rules and `process_env.strip_flags`) are read as a POSIX shell reads them: white
  space separates, single quotes keep everything, double quotes keep everything but `\"` and `\\`, a backslash outside quotes escapes
  the next character; an unclosed quote means no value.

cuda, as the Python states it today (data the OSPREY's `tools.json` carries): `device.driver`: libraries `["libcuda.so.1",
"libcuda.so"]` (none opens on macOS, so the arch there is the portable virtual one), calls `cuInit`, `cuDriverGetVersion`, `cuCtxGetDevice`,
`cuDeviceGet`, `cuDeviceGetAttribute`, `version_decode` `[1000, 10]`, attributes major 75 and minor 76, `arch_format` `sm_{major}{minor}`;
`device.arch`: `pin_env` `HAWK_CUDA_ARCH`, `fallback` `compute_61`, `portable_min` 60, `toolkit` `{"tool": "compiler", "probe":
"archs"}`; `runtime_compile`: `replaces` `compiler`, libraries `["libnvrtc.so", "libnvrtc.so.12"]`, `force` `{"env":
"HAWK_DEVICE_COMPILER", "tool": "nvcc", "runtime": "nvrtc"}`, calls `nvrtcVersion`, `nvrtcGetNumSupportedArchs`,
`nvrtcGetSupportedArchs`, `nvrtcCreateProgram`, `nvrtcCompileProgram`, `nvrtcGetProgramLogSize`, `nvrtcGetProgramLog`,
`nvrtcDestroyProgram`, images `ptx` (`nvrtcGetPTXSize`, `nvrtcGetPTX`, text) and `cubin` (`nvrtcGetCUBINSize`, `nvrtcGetCUBIN`).
The runtime options the Python builds are, as a request's `runtime.options`: `-std=c++20`, `-arch={arch}`, `-default-device`,
`-D__builtin_inf()=__longlong_as_double(0x7ff0000000000000LL)` and any `-I` of a header root, then the caller's defines and extras;
`{arch}` is a request place (`compute_<n>` for a PTX target, `sm_<n>` for a CUBIN). `reserved_names` for cpu: `tool` `compiler`, `args`
`["-E","-P","-x","c++","-std=c++23","-DAETHER_CPP_MODE","-I{include:aether}","-I{include:eagle}","{source}"]` (the header shape of the
facts picks the trees), `system_headers` `["cstdlib","cstdio","cstring","cmath","unistd.h"]`; the ask's `prelude` is the writers'
include block. The masked-call witness for cpu is the ask's `witnesses` record: `source` the witness file, `compile` the unit's own
`compile_unit` template (its `{include:*}` places become one empty folder, `{defines}` is dropped), `link` `["{compiler}", "{object}",
"-o", "{output}"]`, `exit_verdicts` `{"0": "fixed", "1": "miscompile"}`, `timeout` 120; a verdict `unrun` is not kept.

## 6b. The cap on tool processes

At most N tool processes run at once under one store root, across threads and processes (a held `flock` on one of N files under
`<root>/jobs`; a dead process frees its slot). N is the request's `jobs` member (build and link requests), else the table's
`limits.jobs` (`{"env": "HAWK_COMPILE_JOBS", "default": 2}`), else 2; 0 lifts the cap. The cap applies to builds, links and image
assembly, not to the probes of `machine`.

## 7. Not in the toolkit

- The zig-wheel compiler fallback is a Python package and is not ported; the `tools_absent` reason names every compiler tried in order.
- The reserved names and the verdicts are kept in this process (reserved names) or in the store under the root (witnesses); the
  reserved names are not persisted across processes.
- The real runtime compile and driver libraries are not on the machine this was written on: the mechanism is proven against fake
  libraries with the same C entry points; the real libraries are proven on the Linux machine.
