//! kreplay <corpus dir> [--show k] [--accepted list]: streams a kernel corpus (per build call: the kernel before lowering, the units the old lowering
//! made and each unit's recorded host text) through the lowering entry of the library's table. For every kernel the old lowering
//! left as one plain unit it loads the kernel's IR, builds the request from the record, asks the facts entry for the machine the
//! recording states, calls `lower`, and compares the walk digest, the text byte for byte and the meta. Kernels of any other shape
//! are counted by shape (the work lists of the later slices). Files are read one line at a time and never held whole.
//!
//! What the front end's request must state itself: this tool adapts the request where the record
//! does not say it, and each
//! adaptation is counted in its own printed category, never in `equal`:
//! - the wide form: a call with no split record is asked with the thread form (`wide form`);
//! - the kernel name: leading underscores are left off, as the units were built (`name as built`);
//! - the box switch: a unit the old build wrote unboxed on purpose is listed, not compared (`box
//! switch`);
//! - patched calls: kernels of a call whose test replaced the old renderer are not compared
//!   (`patched`);
//! - own-card calls: kernels built under a card of the test's own are not compared (`own card`).
//!
//! A unit is `equal` only when it equals the last recorded render and the request needed none of
//! the first two.
use aether_writers::contract::types::{Access, Guard, Scalar, SinkPolicy, Target};
use aether_writers::contract::types::{Caches, MapBits};
use osprey_host::facts;
use osprey_host::facts::FusionRow;
use osprey_host::lower::request::{
    Body, DeclaredSlot, IndexSet, KernelAnswer, KernelRequest, LaneAxis, Member, RequestKind,
    Settings, SinkAxes, Steps, UnitAnswer, WideForm, ACCESSES, DTYPES, FOLD_FORMS, LINALG_FORMS,
    ROLES, SCALARS, WIDE_FORMS,
};
use osprey_host::lower::unit_meta as meta;
use raptor_ir::door::{RX_OK, RX_TRUNC};
use raptor_support::corpus::{declare, err, make_nodes, manifests, Vals};
use rcore::json::{self, J};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::c_char;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[path = "kreplay/meta.rs"]
mod oracle;

/// The context a call's nodes are loaded into, with the handles the record's node numbers stand for; none before the first load.
type Loaded = Option<(*mut raptor_ir::record::Ctx, Vec<u32>)>;

extern crate raptor_wings; // links the wing analyses the writers call

/// The compiler identity of a clang and of a GCC, as the recording's build had.
const COMPILERS: [&str; 2] = [
    "c++ || Apple clang version 21.0.0 (clang-2100.0.123.102) || 0",
    "g++ || g++ (GCC) 13.2.0 || 0",
];

/// The record under a tagged record's `v`, or the value itself.
fn rec(j: &J) -> &J {
    j.get("v").unwrap_or(j)
}
/// The text of a tagged string.
fn text(j: Option<&J>) -> Option<&str> {
    j.and_then(|x| x.get("s")).and_then(J::str)
}
/// The integer of a tagged integer.
fn num(j: Option<&J>) -> Option<i64> {
    j.and_then(|x| x.get("i"))
        .and_then(J::str)
        .and_then(|s| s.parse().ok())
}
/// The elements of a tagged list or tuple.
fn items(j: Option<&J>) -> &[J] {
    j.and_then(|x| x.get("l").or(x.get("t")))
        .map_or(&[], J::arr)
}
/// The value under a key of a tagged dict, seen through a frozen wrapper.
fn entry<'a>(dict: &'a J, key: &str) -> Option<&'a J> {
    let dict = dict.get("of").unwrap_or(dict);
    dict.get("d")?
        .arr()
        .iter()
        .find(|pair| text(pair.arr().first()) == Some(key))
        .and_then(|pair| pair.arr().get(1))
}
/// The reference a recorded value names (`{"fn": "module:name"}`).
fn reference(j: &J) -> Option<&str> {
    j.get("fn").and_then(J::str)
}

/// The kind the record states for a kernel.
fn kind_of(kind: &J) -> Result<RequestKind, String> {
    let plain = |guard| RequestKind {
        guard,
        active_set: false,
        lanes: None,
        sink: SinkPolicy::Plain,
    };
    if matches!(kind, J::Null) {
        return Ok(plain(Guard::Default));
    }
    if let Some(path) = reference(kind) {
        return match path {
            "hawk.ext:DEFAULT_KIND" => Ok(plain(Guard::Default)),
            other => Err(format!("kernel kind {other}")),
        };
    }
    let kind = rec(kind);
    let guard = kind.get("guard").ok_or("kind without a guard")?;
    let (guard, active_set) = match reference(guard) {
        Some("hawk.ext:DEFAULT_GUARD") => (Guard::Default, false),
        Some("hawk.ext:DATA_ONLY") => (Guard::DataOnly, false),
        Some(other) => return Err(format!("guard {other}")),
        None => {
            let g = rec(guard);
            let masks: Vec<String> = items(g.get("masks"))
                .iter()
                .filter_map(|m| text(Some(m)).map(String::from))
                .collect();
            let masks = if masks.is_empty() {
                text(g.get("mask"))
                    .map(|m| vec![m.to_string()])
                    .unwrap_or_default()
            } else {
                masks
            };
            (
                Guard::Masks(masks),
                matches!(g.get("active_set"), Some(J::Bool(true))),
            )
        }
    };
    let sink = kind.get("sink").ok_or("kind without a sink")?;
    let sink = if reference(sink) == Some("hawk.ext:PLAIN") {
        SinkPolicy::Plain
    } else {
        let s = rec(sink);
        let output = kind.get("output").map(rec);
        match text(s.get("compensation")) {
            Some(plane) => SinkPolicy::Compensated {
                plane: plane.to_string(),
                output_slot: output.and_then(|o| text(o.get("slot"))).map(String::from),
                output_names: items(output.and_then(|o| o.get("names")))
                    .iter()
                    .filter_map(|n| text(Some(n)).map(String::from))
                    .collect(),
            },
            None => return Err("a sink policy of another form".into()),
        }
    };
    Ok(RequestKind {
        guard,
        active_set,
        lanes: lanes_of(kind),
        sink,
    })
}

/// The index set of lanes the record's kind declares, when it does.
fn lanes_of(kind: &J) -> Option<IndexSet> {
    let set = rec(kind.get("lanes").filter(|l| !matches!(l, J::Null))?);
    Some(IndexSet {
        bits: if num(set.get("width")) == Some(64) {
            MapBits::B64
        } else {
            MapBits::B32
        },
        prefix: matches!(set.get("prefix"), Some(J::Bool(true))),
    })
}

/// A declared slot of the record.
fn declared_of(slot: &J) -> Result<DeclaredSlot, String> {
    let slot = rec(slot);
    let ttype = rec(slot.get("ttype").ok_or("a declared slot without a type")?);
    let (role, dtype) = (text(slot.get("role")), text(ttype.get("dtype")));
    Ok(DeclaredSlot {
        role: *ROLES
            .iter()
            .find(|r| Some(r.name()) == role)
            .ok_or("a declared role")?,
        name: text(slot.get("name")).ok_or("a declared name")?.to_string(),
        dtype: *DTYPES
            .iter()
            .find(|d| Some(d.name()) == dtype)
            .ok_or("a declared dtype")?,
        shape: items(ttype.get("shape"))
            .iter()
            .filter_map(|n| num(Some(n)))
            .collect(),
        tag: text(ttype.get("tag")).map(String::from),
    })
}

/// The settings the record's spec states.
fn settings_of(spec: &J) -> Result<Settings, String> {
    let s = rec(spec.get("settings").ok_or("a spec without settings")?);
    let form = text(s.get("form")).unwrap_or("auto");
    let bounds = items(s.get("launch_bounds"));
    Ok(Settings {
        fold_form: *FOLD_FORMS
            .iter()
            .find(|f| format!("{f:?}").to_lowercase() == form)
            .ok_or("a fold form")?,
        launch_bounds: match bounds {
            [a, b] => Some((num(Some(a)).unwrap_or(0), num(Some(b)).unwrap_or(0))),
            _ => None,
        },
        wide_form: *WIDE_FORMS
            .iter()
            .find(|f| format!("{f:?}").to_lowercase() == text(s.get("wide_form")).unwrap_or("auto"))
            .ok_or("a wide form")?,
        linalg_form: *LINALG_FORMS
            .iter()
            .find(|f| {
                format!("{f:?}").to_lowercase() == text(s.get("linalg_form")).unwrap_or("auto")
            })
            .ok_or("a linalg form")?,
        route_to_ordered: matches!(s.get("route_to_ordered"), Some(J::Bool(true))),
        fold_k: num(s.get("fold_k")).filter(|k| *k != 0),
        warps: num(s.get("warps")).unwrap_or(1),
        team_threads: num(s.get("team_threads")).unwrap_or(0),
        exports: None,
        literal_table: false,
    })
}

/// The facts answer for the machine the record's spec states (the caller's side of the facts entry), by instruction set, caches,
/// host profile and compiler: asked once each.
fn facts_answer(spec: &J, cache: &mut HashMap<String, String>) -> Result<String, String> {
    let machine = rec(spec.get("machine").ok_or("a spec without a machine")?);
    let mp = machine
        .get("facts")
        .and_then(|f| f.get("mp"))
        .ok_or("a machine without facts")?;
    let get = |key: &str| entry(mp, key);
    let isa = text(machine.get("isa")).map(String::from);
    let profile = get("host_profile")
        .and_then(|p| text(Some(p)))
        .map(String::from);
    let gcc = matches!(get("arm64_gcc"), Some(J::Bool(true)))
        || (isa.as_deref() == Some("x86_64")
            && profile.as_deref().is_some_and(|p| p.contains("vector")));
    let caches = match (num(get("l1d_cache_bytes")), num(get("l2_cache_bytes"))) {
        (Some(l1d_bytes), Some(l2_bytes)) => Some(Caches {
            l1d_bytes,
            l2_bytes,
            source: text(get("cache_source")).unwrap_or("").to_string(),
        }),
        _ => None,
    };
    let vector = text(get("vector_isa"));
    let key = format!("{isa:?}|{vector:?}|{caches:?}|{profile:?}|{gcc}");
    if let Some(found) = cache.get(&key) {
        return Ok(found.clone());
    }
    let stated = facts::Stated {
        isa,
        vector_isa: text(get("vector_isa")).map(String::from),
        caches,
        host_profile: profile,
        device_arch: None,
    };
    let dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/facts"));
    let answer = facts::answer(
        dir,
        "host",
        COMPILERS[gcc as usize],
        &facts::probe(),
        &stated,
    )?;
    cache.insert(key, answer.clone());
    Ok(answer)
}

/// The shape of a kernel none of whose recorded units carries its name: the record holds no unit of it (a call that failed before
/// the kernel was rendered) or the units under another name.
const UNRECORDED: &str = "a kernel whose recorded units carry another name";

/// What the old lowering made of one kernel, from the call's units.
enum Shape {
    /// One plain unit of the kernel's own walk.
    Plain,
    /// A kernel split by a segmented dispatch into one unit per branch (and a classifying unit).
    Segmented,
    /// A kernel cut into stages, or placed with a stage block of its own.
    Staged,
    /// A chain of kernels, one unit or one per piece of a cut.
    Chain,
    /// One unit written for the device alone as a warp per lane.
    Warp,
    /// A kernel built for a target the library does not serve (named by the unit's targets).
    Unserved(String),
    /// Anything else, named by the first shape that applies.
    Other(&'static str),
}

/// The targets the kernel's own unit was built for.
fn unit_targets<'a>(kernel: &J, units: &'a [J]) -> Vec<&'a str> {
    let name = kernel.get("name").and_then(J::str).unwrap_or("");
    let first = units
        .iter()
        .find(|u| u.get("unit").and_then(J::str) == Some(name))
        .or(units.first());
    first
        .and_then(|u| u.get("targets"))
        .map_or(&[][..], J::arr)
        .iter()
        .filter_map(J::str)
        .collect()
}

/// The shape of a kernel by its units' names, flags and classes.
fn host_shape(kernel: &J, units: &[J], kernels: &[J]) -> Shape {
    let name = kernel.get("name").and_then(J::str).unwrap_or("");
    let targets = unit_targets(kernel, units);
    if let Some(other) = targets.iter().find(|t| !matches!(**t, "host" | "cuda")) {
        return Shape::Unserved((*other).to_string());
    }
    if !targets.is_empty() && !targets.contains(&"host") && targets != ["cuda"] {
        return Shape::Other("no host target");
    }
    if matches!(kernel.get("segmented"), Some(J::Bool(true))) {
        return Shape::Segmented;
    }
    if is_chain(kernel) {
        return Shape::Chain;
    }
    let Some(unit) = units
        .iter()
        .find(|u| u.get("unit").and_then(J::str) == Some(name))
    else {
        return if own_units(name, units, kernels).is_empty() {
            Shape::Other(UNRECORDED)
        } else {
            Shape::Staged
        };
    };
    if unit.get("cls").and_then(J::str) == Some("WarpStage") {
        return Shape::Warp;
    }
    let set = |key: &str| !matches!(unit.get(key), None | Some(J::Null) | Some(J::Bool(false)));
    if set("segment") {
        return Shape::Other("segment");
    }
    if unit.get("cls").and_then(J::str) == Some("FusedKernel") {
        return Shape::Other("fused lanes");
    }
    if unit.get("members").is_some_and(|m| m.int() > 0) {
        return Shape::Other("chain");
    }
    if set("team")
        || (set("stage_block")
            && unit
                .get("stage_block")
                .is_some_and(|block| !one_stage(block)))
    {
        return Shape::Staged;
    }
    match unit.get("cls").and_then(J::str) {
        Some("Kernel" | "_ActiveUnit" | "SimpleNamespace") | None => {}
        Some(_) => return Shape::Other("another stage class"),
    }
    if own_units(name, units, kernels).len() > 1 {
        return Shape::Staged;
    }
    Shape::Plain
}

/// The recorded units that are this kernel's own, in order: its name, and its stages `<name>_<n>` unless another kernel of the call
/// has that name.
fn own_units<'a>(name: &str, units: &'a [J], kernels: &[J]) -> Vec<&'a J> {
    let named = |n: &str| {
        kernels
            .iter()
            .any(|k| k.get("name").and_then(J::str) == Some(n))
    };
    units
        .iter()
        .filter(|record| record.get("unit").and_then(J::str).is_some())
        .filter(|record| {
            let u = record.get("unit").and_then(J::str).unwrap_or("");
            u == name
                || (u
                    .strip_prefix(name)
                    .and_then(|r| r.strip_prefix('_'))
                    .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
                    && !named(u))
        })
        .collect()
}

/// Whether a stage block is the kernel as one stage with no cells, team or keyed planes: the kernel placed whole.
fn one_stage(block: &J) -> bool {
    let block = block.get("v").unwrap_or(block);
    items(entry(block, "stages")).len() == 1
        && ["team_planes", "keyed"]
            .iter()
            .all(|key| entry(block, key).is_none())
}

/// The side (`host` or `cuda`) a target's records are named by.
fn side_of(target: Target) -> &'static str {
    SIDES.iter().find(|s| s.0 == target).map_or("host", |s| s.1)
}

/// The recorded units built under a name for a side, in the call's order: the units of that name
/// whose targets include the
/// side (a unit that names no target counts for every side). The last one is the build the record
/// ends with.
fn built_as<'a>(units: &[&'a J], name: &str, side: &str) -> Vec<&'a J> {
    let for_side = |u: &J| {
        let targets = u.get("targets").map_or(&[][..], J::arr);
        targets.is_empty() || targets.iter().any(|t| t.str() == Some(side))
    };
    units
        .iter()
        .copied()
        .filter(|u| u.get("unit").and_then(J::str) == Some(name) && for_side(u))
        .collect()
}

/// The walk digest a recorded unit states.
fn digest_of(unit: Option<&J>) -> Option<&str> {
    unit?.get("digest").and_then(J::str)
}

/// Whether the record cannot say which build the answer is: the last build of the name has another
/// digest than the answer and an
/// earlier build of the same name has the answer's. The recorder would have to state, on each unit
/// and text, the index of the
/// build it came from.
fn ambiguous_record(same: &[&J], answer: &str) -> bool {
    let Some((last, earlier)) = same.split_last() else {
        return false;
    };
    digest_of(Some(last)) != Some(answer)
        && earlier.iter().any(|u| digest_of(Some(u)) == Some(answer))
}

/// What the tool adapts in the request of a kernel, so that the record can pass (see the header of
/// this file).
fn adaptations(call: &J, kernel: &J) -> Vec<&'static str> {
    let mut found = vec![];
    let spec = rec(call.get("spec").unwrap_or(call));
    let stated = settings_of(spec).map(|s| s.wide_form);
    if !has_split(call) && stated.is_ok_and(|form| form != WideForm::Thread) {
        found.push("wide form");
    }
    if matches!(kernel.get("name_as_built"), Some(J::Bool(true))) {
        found.push("name as built");
    }
    found
}

/// Whether the call went through the split, and so states its wide form.
fn has_split(call: &J) -> bool {
    call.get("split_in").is_some_and(|s| !matches!(s, J::Null))
}

/// The first line of `want` and `got` that differ, as a pair.
fn first_diff(want: &str, got: &str) -> String {
    let (mut a, mut b) = (want.lines(), got.lines());
    loop {
        match (a.next(), b.next()) {
            (Some(x), Some(y)) if x == y => continue,
            (None, None) => return "same lines, different ending".into(),
            (x, y) => {
                return format!(
                    "recorded {:?} / computed {:?}",
                    x.unwrap_or("<end>"),
                    y.unwrap_or("<end>")
                )
            }
        }
    }
}

/// What the recording holds for a unit's host render.
struct Expected {
    /// The text.
    text: String,
    /// The entry name.
    entry: String,
    /// The scalar mode.
    mode: String,
    /// Whether the lane loop was vectorised.
    vectorised: bool,
    /// The unit's thread box as (axis, lanes, stride), when it has one.
    boxed: Option<(i64, i64, i64)>,
    /// The test that made this render.
    test: String,
    /// The unit key of the per-unit corpus.
    ukey: String,
    /// The broadcast pattern of the kernel the unit was written from: each gathered plane and its flag for each lane axis.
    broadcast: Vec<(String, Vec<bool>)>,
}

/// The counts of the kernels written as a warp per lane.
#[derive(Default)]
struct WarpTally {
    /// Kernels asked for.
    asked: usize,
    /// Units whose walk digest equals the record's.
    digest_equal: usize,
    /// Kernels whose stage block equals the record's.
    block_equal: usize,
    /// Kernels whose digest, stage block and text all equal the record's.
    equal: usize,
    /// Kernels answered that differ from the record in one of the three.
    different: usize,
    /// Units whose text equals the record's.
    text_equal: usize,
    /// Refusals of the library, by reason.
    refused: BTreeMap<String, usize>,
    /// The first differences, one line each.
    diffs: Vec<String>,
}

/// The counts of one corpus.
#[derive(Default)]
struct Tally {
    /// The meta of the answer's units against the recorded sidecars.
    meta: oracle::Oracle,
    /// Build calls read.
    calls: usize,
    /// Calls that ended in an error.
    errors: usize,
    /// Calls served from the unit memo (no units, no texts).
    memo: usize,
    /// Calls whose kernels are all plain.
    plain_calls: usize,
    /// Plain kernels replayed.
    replayed: usize,
    /// Units equal: digest, text and meta.
    equal: usize,
    /// Units equal to an earlier recorded render of the same unit and not to the last one (digest
    /// and meta equal).
    other_render: usize,
    /// Units whose text differs.
    text_differs: usize,
    /// Units equal to the last render whose request the tool adapted (see the header), by
    /// adaptation.
    adapted_by: BTreeMap<&'static str, usize>,
    /// Units equal to the last render whose request was adapted, counted once however many
    /// adaptations they needed.
    adapted: usize,
    /// Units whose recorded same-name units cannot tell which one the answer is: the digest of the
    /// last differs and an
    /// earlier one equals the answer's. Listed with the test.
    ambiguous: Vec<String>,
    /// Every outcome counted once: a unit compared, a unit without text, a kernel refused, not
    /// built or set aside.
    items: usize,
    /// Units whose digest differs.
    digest_differs: usize,
    /// Units whose meta differs.
    meta_differs: usize,
    /// Replayed kernels the old lowering wrote with a thread box.
    boxed: usize,
    /// Of those, the units equal to the record.
    boxed_equal: usize,
    /// Units of segmented splits replayed.
    segment_units: usize,
    /// Of those, the units equal to the record.
    segment_equal: usize,
    /// Units of chains replayed.
    chain_units: usize,
    /// Units of chains equal to the record.
    chain_equal: usize,
    /// Chains a test built under its own fusion row (the row it set is not in the record), listed and not counted.
    fusion_overridden: Vec<String>,
    /// Calls whose record lacks a field the request needs, so the converter's stated default stands: the auto cap, the fusion rows,
    /// the plane declarations of a derivative, and the image format (no recording states it).
    without: [usize; 4],
    /// Chains whose planes between the stages differ from the record.
    chain_planes_differ: Vec<String>,
    /// Plain kernels with no recorded host text.
    no_text: usize,
    /// Units of segmented splits with no recorded host text.
    segment_no_text: usize,
    /// Units of segmented splits the record lists (the replayed ones and those without text).
    segment_recorded: usize,
    /// Replayed units equal in digest and meta whose text differs and whose key is on the accepted list.
    accepted: Vec<String>,
    /// The accepted list: unit keys whose text differs and is proven by value.
    accepted_keys: HashSet<String>,
    /// Kernels of a call whose test replaced the old renderer or the render seam: not compared.
    patched: usize,
    /// Units equal to an earlier render of the unit and not to the last: (unit, the matching render's test, the last render's test).
    earlier: Vec<String>,
    /// Units the old build wrote unboxed under its environment switch (the value is in the record).
    box_off: Vec<String>,
    /// Refusals of the library, by reason.
    refused: BTreeMap<String, usize>,
    /// Failures to build the request or the context from the record, by reason.
    unbuilt: BTreeMap<String, usize>,
    /// Shapes of the other kernels.
    shapes: BTreeMap<&'static str, usize>,
    /// Kernels built for a target the library does not serve, by the library's answer.
    unserved: BTreeMap<String, usize>,
    /// The kernels the library refused or that were not built, by call and name, with the reason.
    failed_ids: Vec<String>,
    /// Kernels with no recorded text, by call and name.
    no_text_ids: Vec<String>,
    /// The units of kernels written as a warp per lane, replayed.
    warp: WarpTally,
    /// Kernels built under a card of the test's own, listed and not compared.
    own_card: Vec<String>,
    /// Units of staged kernels replayed.
    staged_units: usize,
    /// Of those, the units equal to the record.
    staged_equal: usize,
    /// Staged kernels whose answer was compared with the recorded stage block and run facts.
    blocks: usize,
    /// Of those, the kernels whose stage block and run facts are equal.
    blocks_equal: usize,
    /// The first stage blocks or run facts that differ, as a pair.
    block_diffs: Vec<String>,
    /// The first differences shown.
    shown: usize,
}

/// The wanted units of one corpus process: the keys (call, unit) whose host text is needed.
fn needed_keys(path: &Path) -> HashSet<(String, String)> {
    let mut keys = HashSet::new();
    for line in BufReader::new(std::fs::File::open(path).unwrap()).lines() {
        let mut rec = json::parse(&line.unwrap()).unwrap();
        name_as_built(&mut rec);
        if (failed(&rec) && !after_render(&rec)) || rec.get("units").is_none() {
            continue;
        }
        if is_patched(&rec) {
            continue;
        }
        let units = rec.get("units").unwrap().arr();
        for kernel in rec.get("kernels").map_or(&[][..], J::arr) {
            let shape = host_shape(kernel, units, rec.get("kernels").unwrap().arr());
            let call = rec.get("call").and_then(J::str).unwrap_or("").to_string();
            if matches!(shape, Shape::Segmented) {
                let name = kernel.get("name").and_then(J::str).unwrap_or("");
                keys.extend(segment_units(name, units).map(|u| {
                    (
                        call.clone(),
                        u.get("unit").and_then(J::str).unwrap_or("").to_string(),
                    )
                }));
            }
            if matches!(shape, Shape::Staged) {
                let name = kernel.get("name").and_then(J::str).unwrap_or("");
                let kernels = rec.get("kernels").unwrap().arr();
                keys.extend(own_units(name, units, kernels).into_iter().map(|u| {
                    (
                        call.clone(),
                        u.get("unit").and_then(J::str).unwrap_or("").to_string(),
                    )
                }));
            }
            if matches!(shape, Shape::Chain) {
                let name = kernel.get("name").and_then(J::str).unwrap_or("");
                keys.extend(chain_units(name, units).map(|u| {
                    (
                        call.clone(),
                        u.get("unit").and_then(J::str).unwrap_or("").to_string(),
                    )
                }));
            }
            if matches!(shape, Shape::Plain | Shape::Warp) {
                keys.insert((
                    call,
                    kernel
                        .get("name")
                        .and_then(J::str)
                        .unwrap_or("")
                        .to_string(),
                ));
            }
        }
    }
    keys
}

/// Whether the call ended in an error.
fn failed(call: &J) -> bool {
    call.get("error").is_some_and(|e| !matches!(e, J::Null))
}

/// Whether the call's error came after its units were rendered: the device compiler was not there to build them.
fn after_render(call: &J) -> bool {
    call.get("error")
        .and_then(|e| e.get("cls"))
        .and_then(J::str)
        == Some("NvrtcUnavailable")
}

/// Whether the call's test had replaced the old renderer or the render seam.
fn is_patched(call: &J) -> bool {
    call.get("patched").is_some_and(|p| !p.arr().is_empty())
}

/// Whether the old build was told to build without a box (the environment value at the unit's lowering).
fn box_off(unit: Option<&J>) -> bool {
    unit.and_then(|u| u.get("box_env")).and_then(J::str) == Some("0")
}

/// The fields of a record that are the run's own bookkeeping, left out of a unit's key.
const KEY_LEFT_OUT: [&str; 3] = ["primal_unit", "_hawk_source", "host_entry"];

/// Writes `j` in the canonical form the unit keys are made from: sorted keys, no spaces, ASCII only.
fn canon(j: &J, out: &mut String) {
    match j {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Num(n) => out.push_str(&n.to_string()),
        J::Str(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    ' '..='\x7f' => out.push(c),
                    _ => {
                        let mut units = [0u16; 2];
                        for u in c.encode_utf16(&mut units) {
                            out.push_str(&format!("\\u{u:04x}"));
                        }
                    }
                }
            }
            out.push('"');
        }
        J::Arr(items) => {
            out.push('[');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canon(x, out);
            }
            out.push(']');
        }
        J::Obj(members) => {
            let mut sorted: Vec<_> = members
                .iter()
                .filter(|(k, _)| !KEY_LEFT_OUT.contains(&k.as_str()))
                .collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            out.push('{');
            for (i, (k, v)) in sorted.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canon(&J::Str(k.clone()), out);
                out.push(':');
                canon(v, out);
            }
            out.push('}');
        }
    }
}

/// `s` with each test temporary directory (`/...pytest-N/popen-gwN` or `/...basetemp/popen-gwN`, not crossing a quote) written as `<tmp>`.
fn mask_tmp(s: &str) -> String {
    let (mut out, mut done, mut from) = (String::new(), 0, 0);
    while let Some(rel) = s[from..].find("/popen-gw") {
        let gw = from + rel;
        let digits = s[gw + 9..].bytes().take_while(u8::is_ascii_digit).count();
        let head = &s[done..gw];
        let kept = head.trim_end_matches(|c: char| c.is_ascii_digit());
        let marker = if head.ends_with("basetemp") {
            Some(head.len() - 8)
        } else if digits_before(head) && kept.ends_with("pytest-") {
            Some(kept.len() - 7)
        } else {
            None
        };
        let quote = head.rfind('"').map_or(0, |q| q + 1);
        let slash = marker.and_then(|m| head[quote..m].find('/').map(|i| quote + i));
        match slash.filter(|_| digits > 0) {
            Some(i) => {
                out.push_str(&head[..i]);
                out.push_str("<tmp>");
                done = gw + 9 + digits;
                from = done;
            }
            None => from = gw + 1,
        }
    }
    out.push_str(&s[done..]);
    out
}

/// Whether `head` ends in at least one digit.
fn digits_before(head: &str) -> bool {
    head.ends_with(|c: char| c.is_ascii_digit())
}

/// `s` with the run's own folder name under `/snap/` written as `<run>`.
fn mask_run(s: &str) -> String {
    let (mut out, mut done, mut from) = (String::new(), 0, 0);
    while let Some(rel) = s[from..].find("/snap/") {
        let name = from + rel + 6;
        let len = s[name..].find(['/', '"']).unwrap_or(0);
        if len > 0 && s[name + len..].starts_with('/') {
            out.push_str(&s[done..name]);
            out.push_str("<run>");
            done = name + len;
        }
        from = name;
    }
    out.push_str(&s[done..]);
    out
}

/// The key of a unit on the accepted list: the digest of its target and inputs, as the per-unit tools compute it.
fn unit_key(target: &str, inputs: &J) -> String {
    let mut text = String::new();
    canon(
        &J::Arr(vec![J::Str(target.into()), inputs.clone()]),
        &mut text,
    );
    rcore::sha256::hex(mask_run(&mask_tmp(&text)).as_bytes())
}

/// The broadcast pattern a text record's kernel holds: the pairs of a plane name and its flags.
fn recorded_pattern(rec: &J) -> Vec<(String, Vec<bool>)> {
    let kernel = rec
        .get("inputs")
        .and_then(|i| i.get("lowered"))
        .map(self::rec)
        .and_then(|l| l.get("kernel"))
        .map(self::rec);
    let pairs = kernel
        .and_then(|k| k.get("broadcast"))
        .and_then(|b| b.get("d"))
        .map_or(&[][..], J::arr);
    pairs
        .iter()
        .filter_map(|pair| {
            let flags = pair.arr().get(1)?.get("t")?.arr();
            let flags = flags.iter().map(|f| matches!(f, J::Bool(true))).collect();
            Some((text(pair.arr().first())?.to_string(), flags))
        })
        .collect()
}

/// Gives each kernel of the call the broadcast pattern its recorded units were written from, as the build call would pass it.
fn recorded_broadcast(call: &mut J, expected: &[HashMap<(String, String), Vec<Expected>>; 2]) {
    let id = call.get("call").and_then(J::str).unwrap_or("").to_string();
    // the planes each written unit binds: the old build passed the call's whole pattern, the request names only planes its kernel has
    let bound: Vec<(String, Vec<String>)> = call
        .get("units")
        .map_or(&[][..], J::arr)
        .iter()
        .filter_map(|u| {
            let spec = u.get("sidecar")?.get("arg_spec")?.arr();
            let names = spec.iter().filter_map(|a| a.arr().get(1)?.str());
            Some((
                u.get("unit")?.str()?.to_string(),
                names.map(String::from).collect(),
            ))
        })
        .collect();
    let J::Obj(members) = call else { return };
    for (_, value) in members.iter_mut().filter(|m| m.0 == "kernels") {
        let J::Arr(kernels) = value else { continue };
        for kernel in kernels {
            let name = kernel
                .get("name")
                .and_then(J::str)
                .unwrap_or("")
                .to_string();
            let pattern = expected
                .iter()
                .filter_map(|side| side.get(&(id.clone(), name.clone())))
                .flatten()
                .map(|e| &e.broadcast)
                .find(|b| !b.is_empty());
            let (Some(pattern), J::Obj(fields)) = (pattern, kernel) else {
                continue;
            };
            let held = |plane: &str| {
                bound
                    .iter()
                    .find(|(unit, _)| *unit == name)
                    .is_none_or(|(_, names)| {
                        names
                            .iter()
                            .any(|n| n == plane || *n == format!("dot_{plane}"))
                    })
            };
            let pairs = pattern
                .iter()
                .filter(|(plane, _)| held(plane))
                .map(|(plane, flags)| {
                    let flags = flags.iter().map(|f| J::Bool(*f)).collect();
                    J::Arr(vec![J::Str(plane.clone()), J::Arr(flags)])
                });
            fields.push(("broadcast".into(), J::Arr(pairs.collect())));
        }
    }
}

/// The broadcast pattern a kernel record holds, as the request states it.
fn broadcast_of(kernel: &J) -> Vec<(String, Vec<bool>)> {
    kernel
        .get("broadcast")
        .map_or(&[][..], J::arr)
        .iter()
        .filter_map(|pair| {
            let flags = pair.arr().get(1)?.arr();
            let flags = flags.iter().map(|f| matches!(f, J::Bool(true))).collect();
            Some((pair.arr().first()?.str()?.to_string(), flags))
        })
        .collect()
}

/// The recorded texts for one target (`host` or `cuda`) of the wanted keys, read from a process's text records.
fn texts_of(
    path: &Path,
    wanted: &HashSet<(String, String)>,
    target: &str,
) -> HashMap<(String, String), Vec<Expected>> {
    let mut found = HashMap::new();
    let Ok(file) = std::fs::File::open(path) else {
        return found;
    };
    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        if !line.contains(&format!("\"target\":\"{target}\"")) {
            continue;
        }
        let rec = json::parse_without(&line, &["analyses"]).unwrap();
        let key = (
            rec.get("call").and_then(J::str).unwrap_or("").to_string(),
            rec.get("kunit").and_then(J::str).unwrap_or("").to_string(),
        );
        if !wanted.contains(&key) || rec.get("target").and_then(J::str) != Some(target) {
            continue;
        }
        let Some(value) = rec.get("output").and_then(|o| o.get("value")).map(rec_v) else {
            continue;
        };
        found.entry(key).or_insert_with(Vec::new).push(Expected {
            text: text(value.get("text")).unwrap_or("").to_string(),
            entry: text(value.get("entry")).unwrap_or("").to_string(),
            mode: text(value.get("mode")).unwrap_or("").to_string(),
            vectorised: matches!(value.get("vectorised"), Some(J::Bool(true))),
            boxed: value
                .get("box")
                .filter(|b| !matches!(b, J::Null))
                .and_then(|b| {
                    let b = b.get("v").unwrap_or(b);
                    Some((num(b.get("axis"))?, num(b.get("T"))?, num(b.get("stride"))?))
                }),
            test: rec.get("test").and_then(J::str).unwrap_or("").to_string(),
            broadcast: recorded_pattern(&rec),
            ukey: match rec.get("ukey").and_then(J::str) {
                Some(k) if !k.is_empty() => k.to_string(),
                _ => rec
                    .get("inputs")
                    .map(|i| unit_key(target, i))
                    .unwrap_or_default(),
            },
        });
    }
    found
}

/// The `v` of a recorded record.
fn rec_v(j: &J) -> &J {
    rec(j)
}

/// The body of a kernel (or of its single step) from the record: its sinks as handles and the inputs its walk was taken with.
fn member_of(name: &str, record: &J, handles: &[u32]) -> Result<Member, String> {
    let force = text(record.get("force")).map(|f| {
        *ACCESSES
            .iter()
            .find(|a| a.name() == f)
            .unwrap_or(&Access::SampleLocal)
    });
    Ok(Member {
        name: name.to_string(),
        sinks: record
            .get("sinks")
            .map_or(&[][..], J::arr)
            .iter()
            .filter_map(|n| handles.get(n.get("n")?.int() as usize).copied())
            .collect(),
        declared: items(record.get("declared"))
            .iter()
            .map(declared_of)
            .collect::<Result<_, _>>()?,
        force,
        exempt: items(record.get("exempt"))
            .iter()
            .filter_map(|e| text(Some(e)).map(String::from))
            .collect(),
    })
}

/// The five layout sizes a test put in the call's place of the derived ones, when it did.
fn exports_override(call: &J) -> Option<[u64; 5]> {
    let sizes: Vec<u64> = items(call.get("exports_override"))
        .iter()
        .filter_map(|size| num(Some(size)).and_then(|n| u64::try_from(n).ok()))
        .collect();
    sizes.try_into().ok()
}

/// Builds the request of a plain kernel from the record, for the targets of the facts answers it is given.
fn request_for(
    call: &J,
    kernel: &J,
    facts: Vec<(Target, String)>,
    handles: &[u32],
) -> Result<KernelRequest, String> {
    let spec = rec(call.get("spec").ok_or("a call without a spec")?);
    // The build's mode: the split's own when the call was split, else the call's argument.
    let smode = call
        .get("split_in")
        .and_then(|s| s.get("smode"))
        .and_then(J::str)
        .or_else(|| text(call.get("args").and_then(|a| a.get("mode"))))
        .unwrap_or("float64");
    let scalar: Scalar = *SCALARS
        .iter()
        .find(|s| format!("{s:?}").to_lowercase() == smode)
        .ok_or("a scalar mode")?;
    let name = kernel
        .get("name")
        .and_then(J::str)
        .ok_or("a kernel without a name")?
        .to_string();
    let member = member_of(&name, kernel, handles)?;
    Ok(KernelRequest {
        name,
        scalar,
        targets: facts.iter().map(|(target, _)| *target).collect(),
        kind: kind_of(kernel.get("kind").ok_or("a kernel without a kind")?)?,
        body: Body::Plain(member),
        one_step: kernel
            .get("one_step")
            .filter(|step| !matches!(step, J::Null))
            .map(|step| {
                let name = text(step.get("name")).unwrap_or("").to_string();
                member_of(&name, step, handles)
            })
            .transpose()?,
        group: kernel.get("group").map_or(1, J::int),
        planes: kernel
            .get("planes")
            .and_then(|p| p.get("d"))
            .map_or(&[][..], J::arr)
            .iter()
            .filter_map(|pair| text(pair.arr().first()).map(String::from))
            .collect(),
        settings: Settings {
            literal_table: build_option(call, "literal_table")
                .is_some_and(|flag| matches!(flag, J::Bool(true))),
            exports: exports_override(call),
            // a build that did not go through the split never chose a warp per lane
            wide_form: if has_split(call) {
                settings_of(spec)?.wide_form
            } else {
                WideForm::Thread
            },
            ..settings_of(spec)?
        },
        image_format: facts
            .iter()
            .any(|(target, _)| *target == Target::Device)
            .then_some(RECORDED_IMAGE_FORMAT),
        steps: steps_of(kernel, call),
        fusion: fusion_of(call),
        lane_space: lane_space_of(kernel),
        sink_axes: sink_axes_of(kernel),
        shared: shared_of(kernel),
        broadcast: broadcast_of(kernel),
        facts,
    })
}

/// The image format the converter states for a recording that names none: the recorder keeps no compile format, so every device
/// recording gets this one (the recorder owes the field).
const RECORDED_IMAGE_FORMAT: meta::ImageFormat = meta::ImageFormat::Cubin;

/// The cap of an auto step count for a record that holds none (an older recording): hawk's own constant (`AUTO_K_MAX` of its step
/// planner). A record made by the current recorder states the cap in force as `auto_k_max`.
const AUTO_STEPS_CAP: i64 = 64;

/// The steps the kernel record declares: a count, or `auto` with the call's recorded cap.
fn steps_of(kernel: &J, call: &J) -> Option<Steps> {
    let steps = kernel.get("steps")?;
    let cap = match call.get("auto_k_max") {
        Some(J::Num(n)) => *n as i64,
        _ => AUTO_STEPS_CAP,
    };
    match (num(Some(steps)), text(Some(steps))) {
        (Some(n), _) => Some(Steps::Fixed(n)),
        (_, Some("auto")) => Some(Steps::Auto { max: cap }),
        _ => None,
    }
}

/// A recorded number: a whole number or a float, `None` for null.
fn number_of(j: &J) -> Option<f64> {
    num(Some(j))
        .map(|n| n as f64)
        .or_else(|| j.get("f").and_then(J::str).and_then(|t| t.parse().ok()))
}

/// The fusion row the call's last chain cut used, when the recorder stored it.
fn fusion_of(call: &J) -> Option<FusionRow> {
    let row = call.get("fusion_rows")?.arr().last()?;
    let held = |key: &str| {
        row.get("d")
            .map_or(&[][..], J::arr)
            .iter()
            .find(|pair| text(pair.arr().first()) == Some(key))
            .and_then(|pair| pair.arr().get(1))
    };
    Some(FusionRow {
        regs_base: num(held("R0"))?,
        regs_file: num(held("R_file"))?,
        launch_us: held("L_us").and_then(number_of),
        plane_us_per_byte: held("c_plane_us_per_B").and_then(number_of),
        local_over_plane: held("c_local_over_plane").and_then(number_of),
    })
}

/// The kernel's lane axes; a record with none states a single axis with neither label nor extent (the old build's default for a
/// kernel that declares no lane space).
fn lane_space_of(kernel: &J) -> Vec<LaneAxis> {
    match kernel.get("lane_space").filter(|l| !matches!(l, J::Null)) {
        Some(space) => tuple_axes(Some(space)),
        None => vec![LaneAxis {
            label: None,
            extent: None,
        }],
    }
}

/// The axes of a recorded tuple of `(label, extent)` pairs; an entry may lack either.
fn tuple_axes(tuple: Option<&J>) -> Vec<LaneAxis> {
    tuple
        .and_then(|t| t.get("t"))
        .map_or(&[][..], J::arr)
        .iter()
        .map(|axis| {
            let pair = axis.get("t").map_or(&[][..], J::arr);
            LaneAxis {
                label: text(pair.first()).map(String::from),
                extent: num(pair.get(1)),
            }
        })
        .collect()
}

/// Whether the record lacks the fusion rows its chain cuts used (a recording made before the recorder stored them).
fn call_without_fusion_row(call: &J) -> bool {
    call.get("fusion_rows").is_none()
}

/// The planes the kernel record declares under `key` (`planes`, or `derived_planes` for a derivative), with their declaration record.
fn declared_planes<'a>(kernel: &'a J, key: &str) -> impl Iterator<Item = (String, &'a J)> {
    kernel
        .get(key)
        .and_then(|p| p.get("d"))
        .map_or(&[][..], J::arr)
        .iter()
        .filter_map(|pair| {
            let name = text(pair.arr().first())?.to_string();
            Some((name, pair.arr().get(1)?.get("v")?))
        })
}

/// The axes each declared sink plane (an accumulator) states in its `dims`.
fn sink_axes_of(kernel: &J) -> Vec<SinkAxes> {
    // a derivative's planes are all sinks; a declared plane is one when it is an accumulator
    let own = declared_planes(kernel, "planes")
        .filter(|(_, plane)| text(plane.get("form")) == Some("accum"));
    own.chain(declared_planes(kernel, "derived_planes"))
        .filter_map(|(name, plane)| {
            let axes = tuple_axes(plane.get("dims"));
            (!axes.is_empty()).then_some(SinkAxes { name, axes })
        })
        .collect()
}

/// The planes the kernel record declares shared: those with a non-empty `shared` shape.
fn shared_of(kernel: &J) -> Vec<String> {
    declared_planes(kernel, "planes")
        .filter(|(_, plane)| {
            plane
                .get("shared")
                .and_then(|s| s.get("t"))
                .is_some_and(|t| !t.arr().is_empty())
        })
        .map(|(name, _)| name)
        .collect()
}

/// Asks the library's table to lower the request in `ctx`.
fn ask(ctx: *mut raptor_ir::record::Ctx, request: &str) -> Result<KernelAnswer, String> {
    let table = unsafe { &*osprey_host::osprey_v1() };
    let mut cap = 1usize << 20;
    loop {
        let (mut buf, mut len) = (vec![0u8; cap], 0usize);
        let status = (table.lower)(
            ctx,
            request.as_ptr() as *const c_char,
            request.len(),
            buf.as_mut_ptr() as *mut c_char,
            cap,
            &mut len,
        );
        match status {
            RX_TRUNC => cap = len + 1,
            RX_OK => {
                buf.truncate(len);
                return KernelAnswer::from_json(&String::from_utf8_lossy(&buf))
                    .map_err(|r| r.reason);
            }
            _ => return Err(format!("{status}|{}", err(ctx))),
        }
    }
}

/// The facts answers for every target the kernel's units were built for (the host alone when the record names none), as one request
/// carries them: a kernel of a bundle is lowered once for all its targets.
fn facts_all(
    found: &Found,
    kernel: &J,
    machines: &mut HashMap<String, String>,
) -> Result<Vec<(Target, String)>, String> {
    let named = unit_targets(kernel, call_units(found.call));
    let mut facts = vec![];
    for (target, side) in SIDES {
        if target == found.target || named.contains(&side) {
            facts.push((target, facts_for(found.call, target, machines)?));
        }
    }
    Ok(facts)
}

/// Asks for the answer and keeps of each unit the text for one target.
fn ask_side(
    target: Target,
    ctx: *mut raptor_ir::record::Ctx,
    request: &str,
) -> Result<KernelAnswer, String> {
    let mut answer = ask(ctx, request)?;
    for unit in &mut answer.units {
        unit.texts.retain(|text| text.target == target);
    }
    Ok(answer)
}

/// Gives a kernel the name its units were built under: the old build left a kernel's leading underscores off the unit's name (it is
/// the caller's to send the name the units carry).
fn name_as_built(call: &mut J) {
    let built: Vec<String> = call
        .get("units")
        .map_or(&[][..], J::arr)
        .iter()
        .filter_map(|u| u.get("unit").and_then(J::str).map(String::from))
        .collect();
    let J::Obj(members) = call else { return };
    for (_, value) in members.iter_mut().filter(|m| m.0 == "kernels") {
        let J::Arr(kernels) = value else { continue };
        for kernel in kernels {
            let J::Obj(fields) = kernel else { continue };
            let mut renamed = false;
            for (_, name) in fields.iter_mut().filter(|f| f.0 == "name") {
                let J::Str(text) = name else { continue };
                let bare = text.trim_start_matches('_').to_string();
                if !built.contains(text) && built.contains(&bare) {
                    *text = bare;
                    renamed = true;
                }
            }
            if renamed {
                fields.push(("name_as_built".into(), J::Bool(true)));
            }
        }
    }
}

/// The targets a corpus is replayed for, with the name its records use for each.
const SIDES: [(Target, &str); 2] = [(Target::Host, "host"), (Target::Device, "cuda")];

/// Replays one corpus process: every kernel for each target its units were built for, the host counts in `tallies[0]` and the
/// device counts in `tallies[1]`.
fn replay(kpath: &Path, tpath: &Path, show: usize, tallies: &mut [Tally; 2]) {
    let wanted = needed_keys(kpath);
    let expected = [
        texts_of(tpath, &wanted, "host"),
        texts_of(tpath, &wanted, "cuda"),
    ];
    let mut machines = HashMap::new();
    for line in BufReader::new(std::fs::File::open(kpath).unwrap()).lines() {
        let mut call = json::parse(&line.unwrap()).unwrap();
        name_as_built(&mut call);
        recorded_broadcast(&mut call, &expected);
        tallies[0].calls += 1;
        let errored = failed(&call);
        if errored {
            tallies[0].errors += 1;
            if !after_render(&call) {
                continue;
            }
        }
        if call
            .get("memo_hit")
            .is_some_and(|m| matches!(m, J::Bool(true)))
            || call.get("units").is_none()
        {
            tallies[0].memo += 1;
            continue;
        }
        let (kernels, units) = (
            call.get("kernels").map_or(&[][..], J::arr),
            call.get("units").unwrap().arr(),
        );
        let derived = |k: &J| text(k.get("derived").and_then(|d| d.get("kind"))).is_some();
        tallies[0].without[0] += call.get("auto_k_max").is_none() as usize;
        tallies[0].without[1] += call.get("fusion_rows").is_none() as usize;
        tallies[0].without[2] += kernels
            .iter()
            .filter(|k| derived(k) && k.get("derived_planes").is_none())
            .count();
        tallies[0].without[3] += kernels
            .iter()
            .any(|k| unit_targets(k, units).contains(&"cuda"))
            as usize;
        let shapes: Vec<Shape> = kernels
            .iter()
            .map(|k| host_shape(k, units, kernels))
            .collect();
        tallies[0].plain_calls += (!errored
            && !kernels.is_empty()
            && shapes.iter().all(|s| matches!(s, Shape::Plain))
            && kernels.iter().all(|k| unit_targets(k, units) != ["cuda"]))
            as usize;
        let mut ctx = None;
        for (side, tally) in SIDES.iter().zip(tallies.iter_mut()) {
            let found = Found {
                call: &call,
                id: call.get("call").and_then(J::str).unwrap_or(""),
                show,
                target: side.0,
            };
            let texts = &expected[(side.0 == Target::Device) as usize];
            for (kernel, shape) in kernels.iter().zip(&shapes) {
                let targets = unit_targets(kernel, units);
                let own = targets.contains(&side.1)
                    || (side.0 == Target::Host
                        && (targets.is_empty() || matches!(shape, Shape::Unserved(_))));
                if !own {
                    continue;
                }
                replay_kernel(
                    &found,
                    (kernel, shape),
                    texts,
                    (&mut ctx, &mut machines),
                    tally,
                );
            }
        }
        if let Some((c, _)) = ctx {
            osprey_host_free(c);
        }
    }
}

/// Replays one kernel of a call for the call's target, by the shape the old lowering gave it.
fn replay_kernel(
    found: &Found,
    (kernel, shape): (&J, &Shape),
    expected: &HashMap<(String, String), Vec<Expected>>,
    (ctx, machines): (&mut Loaded, &mut HashMap<String, String>),
    tally: &mut Tally,
) {
    let call = found.call;
    let is_host = found.target == Target::Host;
    if is_patched(call)
        && matches!(
            shape,
            Shape::Plain | Shape::Segmented | Shape::Staged | Shape::Chain
        )
    {
        tally.patched += 1;
        tally.items += 1;
        return;
    }
    // a call that failed after rendering (the device compiler was missing) records its device texts only
    if failed(call) && (is_host || matches!(shape, Shape::Other(UNRECORDED))) {
        return;
    }
    match shape {
        Shape::Other(label) => *tally.shapes.entry(label).or_default() += 1,
        Shape::Unserved(target) if is_host => {
            let outcome = (|| -> Result<KernelAnswer, String> {
                if ctx.is_none() {
                    *ctx = Some(load(call)?);
                }
                let (c, handles) = ctx.as_ref().ok_or("no context")?;
                let facts = vec![(Target::Host, "{}".to_string())];
                let request = request_for(call, kernel, facts, handles)?.to_json();
                let named = format!("\"targets\":[\"{target}\"]");
                ask(*c, &request.replacen("\"targets\":[\"Host\"]", &named, 1))
            })();
            match outcome {
                Err(why) => {
                    let reason = why.split_once('|').map_or(why.clone(), |r| r.1.into());
                    *tally.unserved.entry(reason).or_default() += 1
                }
                Ok(_) => *tally.unserved.entry("answered".into()).or_default() += 1,
            }
        }
        Shape::Unserved(_) => {}
        Shape::Staged => replay_staged(found, kernel, expected, ctx, machines, tally),
        Shape::Chain => replay_chain(found, kernel, expected, (ctx, machines), tally),
        Shape::Segmented => replay_segmented(found, kernel, expected, ctx, machines, tally),
        Shape::Plain => replay_plain(found, kernel, expected, (ctx, machines), tally),
        Shape::Warp => {
            let name = kernel.get("name").and_then(J::str).unwrap_or("");
            let wants = expected.get(&(found.id.to_string(), name.to_string()));
            replay_warp(call, kernel, wants, ctx, tally)
        }
    }
}

/// The units a call recorded.
fn call_units(call: &J) -> &[J] {
    call.get("units").map_or(&[][..], J::arr)
}

/// Replays a kernel the old lowering left as one plain unit: one request, the answer's unit against the recorded unit.
fn replay_plain(
    found: &Found,
    kernel: &J,
    expected: &HashMap<(String, String), Vec<Expected>>,
    (ctx, machines): (&mut Loaded, &mut HashMap<String, String>),
    tally: &mut Tally,
) {
    let name = kernel.get("name").and_then(J::str).unwrap_or("");
    let units = call_units(found.call);
    let all: Vec<&J> = units.iter().collect();
    let same = built_as(&all, name, side_of(found.target));
    let Some(want) = expected.get(&(found.id.to_string(), name.to_string())) else {
        tally.no_text += 1;
        tally.items += 1;
        tally.no_text_ids.push(format!("{} {name}", found.id));
        return;
    };
    let spec = found.call.get("spec").unwrap_or(found.call);
    if own_card(spec, found.target == Target::Device) {
        tally.own_card.push(format!("{} {name}", found.id));
        tally.items += 1;
        return;
    }
    tally.replayed += 1;
    let outcome = (|| -> Result<KernelAnswer, String> {
        let facts = facts_all(found, kernel, machines)?;
        if ctx.is_none() {
            *ctx = Some(load(found.call)?);
        }
        let (c, handles) = ctx.as_ref().ok_or("no context")?;
        want.last().ok_or("no render")?;
        ask_side(
            found.target,
            *c,
            &request_for(found.call, kernel, facts, handles)?.to_json(),
        )
    })();
    let mut answer = match outcome {
        Ok(answer) => answer,
        Err(why) => return count_failure((found.id, name), why, tally),
    };
    let test = found.call.get("test").and_then(J::str).unwrap_or("");
    let own = same.last().copied();
    keep_recorded(
        &mut answer,
        &own.into_iter().collect::<Vec<_>>(),
        failed(found.call),
    );
    let written: Vec<&UnitAnswer> = answer.units.iter().filter(|u| !is_source(u)).collect();
    if let ([unit], Some(record)) = (written.as_slice(), own) {
        tally
            .meta
            .unit(found.id, unit, &answer, record, box_off(own));
    }
    match written.as_slice() {
        [unit] => compare(
            (found.id, name),
            digest_of(own),
            want,
            unit,
            Setting {
                show: found.show,
                test,
                unboxed: box_off(own),
                adapted: adaptations(found.call, kernel),
                ambiguous: ambiguous_record(&same, &unit.meta.digest),
            },
            tally,
        ),
        _ => {
            let names: Vec<&str> = written.iter().map(|u| u.meta.name.as_str()).collect();
            let why = format!("the answer does not hold one unit|the answer holds {names:?}");
            count_failure((found.id, name), why, tally)
        }
    }
}

/// A build option the call states, either as a keyword of the call or as a `cuda.<name>` entry of its options.
fn build_option<'a>(call: &'a J, name: &str) -> Option<&'a J> {
    let args = call.get("args")?;
    args.get(name)
        .or_else(|| entry(args.get("options")?, &format!("cuda.{name}")))
}

/// Whether the build ran with a card of the test's own: a thread box row on the device, or a host lazy-leaf width other than the
/// shipped one. The request carries no card, so such a kernel is listed and not compared.
fn own_card(spec: &J, device: bool) -> bool {
    let spec = rec(spec);
    if device {
        let cards =
            entry(spec.get("by_target").unwrap_or(spec), "cuda").and_then(|c| entry(c, "cards"));
        let rows = cards
            .and_then(|c| entry(c, "thread_box"))
            .and_then(|b| entry(b, "rows"));
        return !items(rows).is_empty();
    }
    let form = spec.get("host").and_then(|host| entry(host, "form"));
    let width = form.and_then(|f| num(entry(f, "lazy_leaf_width")));
    width.is_some_and(|w| w != 32)
}

/// The facts answer for the device the record's build was for: the architecture the call states, else none (the answer then names
/// the card's default architecture and keeps the default select weights, as a build with no stated architecture did).
fn device_facts(call: &J) -> Result<String, String> {
    let stated =
        text(build_option(call, "device_arch")).or_else(|| text(build_option(call, "arch")));
    let dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/facts"));
    let machine = facts::Stated {
        device_arch: stated.map(String::from),
        ..facts::Stated::default()
    };
    facts::answer(dir, "cuda", COMPILERS[0], &facts::probe(), &machine)
}

/// The facts answer the call's build used for the target: the machine its spec states for the host, the device architecture it
/// states for the device.
fn facts_for(
    call: &J,
    target: Target,
    machines: &mut HashMap<String, String>,
) -> Result<String, String> {
    match target {
        Target::Host => facts_answer(
            rec(call.get("spec").ok_or("a call without a spec")?),
            machines,
        ),
        Target::Device => device_facts(call),
    }
}

/// Replays one kernel built for the device alone as a warp per lane: the request names the device, and the answer is compared with
/// the record (walk digest, stage block, text, entry and scalar mode).
fn replay_warp(
    call: &J,
    kernel: &J,
    wants: Option<&Vec<Expected>>,
    ctx: &mut Loaded,
    t: &mut Tally,
) {
    let name = kernel.get("name").and_then(J::str).unwrap_or("");
    let id = call.get("call").and_then(J::str).unwrap_or("");
    t.warp.asked += 1;
    t.items += 1;
    let units = call.get("units").map_or(&[][..], J::arr);
    let all: Vec<&J> = units.iter().collect();
    let same = built_as(&all, name, "cuda");
    let outcome = (|| -> Result<KernelAnswer, String> {
        let facts = vec![(Target::Device, device_facts(call)?)];
        if ctx.is_none() {
            *ctx = Some(load(call)?);
        }
        let (c, handles) = ctx.as_ref().ok_or("no context")?;
        ask(*c, &request_for(call, kernel, facts, handles)?.to_json())
    })();
    let answer = match outcome {
        Err(why) => {
            let reason = why.split_once('|').map_or(why.clone(), |r| r.1.into());
            t.warp.diffs.push(format!("REFUSED {id} {name}: {reason}"));
            *t.warp.refused.entry(reason).or_default() += 1;
            return;
        }
        Ok(answer) => answer,
    };
    let [unit] = answer.units.as_slice() else {
        *t.warp
            .refused
            .entry("the answer does not hold one unit".into())
            .or_default() += 1;
        return;
    };
    let digest_ok = Some(unit.meta.digest.as_str()) == digest_of(same.last().copied());
    let recorded = call
        .get("split")
        .and_then(|s| entry(s, name))
        .map(recorded_block);
    let block_ok = recorded == answer_block(&answer);
    let text_ok = wants
        .and_then(|w| w.last())
        .is_some_and(|want| unit.texts[0].text == want.text);
    t.warp.digest_equal += digest_ok as usize;
    t.warp.block_equal += block_ok as usize;
    t.warp.text_equal += text_ok as usize;
    t.warp.equal += (digest_ok && block_ok && text_ok) as usize;
    t.warp.different += !(digest_ok && block_ok && text_ok) as usize;
    if !(digest_ok && block_ok && text_ok) {
        let how = match (digest_ok, block_ok) {
            (false, _) => "digest".to_string(),
            (_, false) => {
                let show = |v: Option<V>| v.map_or("none".to_string(), |v| v.to_string());
                format!(
                    "stage block: recorded {} / computed {}",
                    show(recorded),
                    show(answer_block(&answer))
                )
            }
            _ => wants
                .and_then(|w| w.last())
                .map_or("no recorded text".into(), |want| {
                    first_diff(&want.text, &unit.texts[0].text)
                }),
        };
        t.warp.diffs.push(format!("DIFF {id} {name}: {how}"));
    }
}

/// The units of a segmented kernel in the call's records: its branches and its classifying unit.
fn segment_units<'a>(name: &'a str, units: &'a [J]) -> impl Iterator<Item = &'a J> {
    units.iter().filter(move |u| {
        let unit = u.get("unit").and_then(J::str).unwrap_or("");
        unit.strip_prefix(name)
            .is_some_and(|rest| rest.starts_with("__k") || rest == "__cls")
    })
}

/// The call being replayed.
struct Found<'a> {
    /// The call's record.
    call: &'a J,
    /// The call's id.
    id: &'a str,
    /// How many differences to show.
    show: usize,
    /// The target the call is replayed for.
    target: Target,
}

/// A call that failed after its first units were rendered (the device compiler was not there) records only those: of the answer, the
/// units the record holds are compared.
fn keep_recorded(answer: &mut KernelAnswer, recorded: &[&J], failed: bool) {
    if failed {
        let named = |unit: &UnitAnswer| {
            is_source(unit)
                || recorded
                    .iter()
                    .any(|r| r.get("unit").and_then(J::str) == Some(unit.meta.name.as_str()))
        };
        answer.units.retain(named);
    }
}

/// Whether a unit is the entry-less source unit of a cut kernel: it names no entry on any side.
fn is_source(unit: &UnitAnswer) -> bool {
    unit.meta.host_entry.is_none() && unit.meta.device_function.is_none()
}

/// A refusal or a failure to build, counted by its reason and listed by kernel.
fn count_failure((id, name): (&str, &str), why: String, tally: &mut Tally) {
    tally.items += 1;
    tally.failed_ids.push(format!(
        "{id} {name}: {}",
        why.split_once('|').map_or(why.as_str(), |r| r.1)
    ));
    match why.split_once('|') {
        Some((_, reason)) => *tally.refused.entry(reason.to_string()).or_default() += 1,
        None => *tally.unbuilt.entry(why).or_default() += 1,
    }
}

/// Replays a segmented kernel: one request, and every unit of the answer against the recorded unit of its name.
fn replay_segmented(
    found: &Found,
    kernel: &J,
    expected: &HashMap<(String, String), Vec<Expected>>,
    ctx: &mut Loaded,
    machines: &mut HashMap<String, String>,
    tally: &mut Tally,
) {
    let name = kernel.get("name").and_then(J::str).unwrap_or("");
    let units = found.call.get("units").map_or(&[][..], J::arr);
    let recorded: Vec<&J> = segment_units(name, units).collect();
    tally.segment_recorded += recorded.len();
    let outcome = (|| -> Result<KernelAnswer, String> {
        let facts = facts_all(found, kernel, machines)?;
        if ctx.is_none() {
            *ctx = Some(load(found.call)?);
        }
        let (c, handles) = ctx.as_ref().ok_or("no context")?;
        ask_side(
            found.target,
            *c,
            &request_for(found.call, kernel, facts, handles)?.to_json(),
        )
    })();
    let answer = match outcome {
        Ok(answer) => answer,
        Err(why) => {
            return count_failure((found.id, name), why, tally);
        }
    };
    let test = found.call.get("test").and_then(J::str).unwrap_or("");
    let mut answer = answer;
    keep_recorded(&mut answer, &recorded, failed(found.call));
    if answer.units.len() != recorded.len() {
        let why = "the answer holds another number of units than the record";
        count_failure((found.id, name), why.into(), tally);
        return;
    }
    for (unit, record) in answer.units.iter().zip(&recorded) {
        let digest = record.get("digest").and_then(J::str);
        let Some(wants) = expected.get(&(found.id.to_string(), unit.meta.name.clone())) else {
            tally.segment_no_text += 1;
            tally.items += 1;
            continue;
        };
        let before = tally.equal;
        compare(
            (found.id, &unit.meta.name),
            digest,
            wants,
            unit,
            Setting {
                show: found.show,
                test,
                unboxed: box_off(Some(record)),
                adapted: adaptations(found.call, kernel),
                ambiguous: false,
            },
            tally,
        );
        tally.segment_units += 1;
        tally.segment_equal += tally.equal - before;
        tally
            .meta
            .unit(found.id, unit, &answer, record, box_off(Some(record)));
    }
}

/// A value in the one form the recorded blocks and the answer's blocks are both written in for comparison.
#[derive(PartialEq)]
enum V {
    /// Nothing.
    N,
    /// A flag.
    B(bool),
    /// An integer.
    I(i64),
    /// A text.
    S(String),
    /// A list.
    L(Vec<V>),
    /// A dictionary by key.
    D(BTreeMap<String, V>),
}

impl std::fmt::Display for V {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            V::N => write!(f, "None"),
            V::B(b) => write!(f, "{b}"),
            V::I(n) => write!(f, "{n}"),
            V::S(t) => write!(f, "{t:?}"),
            V::L(items) => {
                let parts: Vec<String> = items.iter().map(V::to_string).collect();
                write!(f, "[{}]", parts.join(", "))
            }
            V::D(map) => {
                let parts: Vec<String> = map.iter().map(|(k, v)| format!("{k}: {v}")).collect();
                write!(f, "{{{}}}", parts.join(", "))
            }
        }
    }
}

/// A recorded value (the recorder's tagged encoding) as a `V`.
fn from_record(j: &J) -> V {
    match j {
        J::Null => V::N,
        J::Bool(b) => V::B(*b),
        _ => {
            if let Some(inner) = j.get("of") {
                return from_record(inner);
            }
            if let Some(t) = j.get("s").and_then(J::str) {
                return V::S(t.to_string());
            }
            if let Some(n) = num(Some(j)) {
                return V::I(n);
            }
            if let Some(list) = j.get("l").or(j.get("t")) {
                return V::L(list.arr().iter().map(from_record).collect());
            }
            let pairs = j.get("d").map_or(&[][..], J::arr);
            V::D(
                pairs
                    .iter()
                    .filter_map(|p| match (p.arr().first(), p.arr().get(1)) {
                        (Some(k), Some(v)) => Some((from_record(k).to_string(), from_record(v))),
                        _ => None,
                    })
                    .map(|(k, v)| (k.trim_matches('"').to_string(), v))
                    .collect(),
            )
        }
    }
}

/// The stage facts of the answer's units in the form the record is compared in: the unit that lists the stages carries the kernel's
/// cells, keyed planes and scratch planes. The planes and keyed planes are left out when there are none, and a cell states
/// `internal` only when it is.
fn answer_block(answer: &KernelAnswer) -> Option<V> {
    let carrier = answer
        .units
        .iter()
        .map(|u| &u.meta)
        .find(|m| !m.stages.is_empty())?;
    let b = &carrier.body;
    let texts = |items: &[String]| V::L(items.iter().map(|t| V::S(t.clone())).collect());
    let one =
        |pairs: Vec<(&str, V)>| V::D(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect());
    // a slot of the lister's body, or of the stage unit the `*_stage` member names
    let slot = |at: Option<usize>, stage: Option<usize>| {
        let slots = match stage {
            None => Some(&b.slots),
            Some(n) => carrier
                .stages
                .get(n)
                .and_then(|s| answer.units.iter().find(|u| u.meta.name == s.unit.name))
                .map(|u| &u.meta.body.slots),
        };
        at.and_then(|n| slots?.get(n))
            .map_or(V::N, |s| V::S(s.name.clone()))
    };
    let mut top = vec![
        (
            "cells",
            V::D(
                b.cells
                    .iter()
                    .map(|c| {
                        let op =
                            c.op.c_name()
                                .trim_start_matches("EG_REDUCE_")
                                .to_lowercase();
                        let mut cell = vec![("columns", V::I(c.columns)), ("op", V::S(op))];
                        if c.internal {
                            cell.push(("internal", V::B(true)));
                        }
                        (c.name.clone(), one(cell))
                    })
                    .collect(),
            ),
        ),
        (
            "stages",
            V::L(
                carrier
                    .stages
                    .iter()
                    .map(|s| {
                        let mut stage = vec![
                            ("kernel", V::S(s.unit.name.clone())),
                            ("cells_in", texts(&s.cells_in)),
                            ("cells_out", texts(&s.cells_out)),
                        ];
                        if let Some(team) = &s.team {
                            let mut held = vec![("threads", V::I(team.threads))];
                            if team.lanes > 1 {
                                held.push(("lanes", V::I(team.lanes)));
                            }
                            if team.warps > 1 {
                                held.push(("warps", V::I(team.warps)));
                            }
                            let grid = match team.grid {
                                meta::TeamGrid::Lanes => None,
                                meta::TeamGrid::Keys => Some("keys"),
                                meta::TeamGrid::Chunks => Some("chunks"),
                                meta::TeamGrid::Targets => Some("targets"),
                            };
                            if let Some(grid) = grid {
                                held.push(("grid", V::S(grid.into())));
                            }
                            if team.grid == meta::TeamGrid::Keys {
                                if team.groups > 1 {
                                    held.push(("groups", V::I(team.groups)));
                                }
                                if let Some(k) = team.keyed.and_then(|n| b.keyed.get(n)) {
                                    held.push(("keyed", keyed_value(k)));
                                }
                            }
                            if matches!(team.grid, meta::TeamGrid::Chunks | meta::TeamGrid::Targets)
                            {
                                held.push(("of", slot(team.of_slot, team.of_stage)));
                                if let Some(plane) =
                                    team.status_plane.and_then(|n| b.team_planes.get(n))
                                {
                                    held.push(("status", V::S(plane.name.clone())));
                                }
                            }
                            stage.push(("team", one(held)));
                        }
                        one(stage)
                    })
                    .collect(),
            ),
        ),
    ];
    if !b.team_planes.is_empty() {
        let planes = b.team_planes.iter().map(|p| {
            let dtype = p
                .dtype
                .c_name()
                .trim_start_matches("EG_DTYPE_")
                .to_lowercase();
            let mut plane = vec![("dtype", V::S(dtype)), ("per_lane", V::I(p.per_lane))];
            if p.layout == meta::PlaneLayout::PlaneMajor {
                plane.push(("layout", V::S("plane_major".into())));
            }
            if let Some(k) = p.keyed.and_then(|n| b.keyed.get(n)) {
                plane.push(("keys", keyed_value(k)));
            }
            if p.unit == meta::PlaneUnit::Target {
                plane.push(("per_target", slot(p.target_slot, p.target_stage)));
            }
            if p.unit == meta::PlaneUnit::Once {
                plane.push(("once", V::B(true)));
            }
            if p.fill != 0 {
                plane.push(("fill", V::I(p.fill)));
            }
            (p.name.clone(), one(plane))
        });
        top.push(("team_planes", V::D(planes.collect())));
    }
    // the table holds a plane once however many stages and scratch planes use it: the block's own keyed planes are the ones no
    // team stage or scratch plane is keyed by
    let used: Vec<usize> = carrier
        .stages
        .iter()
        .filter_map(|s| s.team.as_ref()?.keyed)
        .chain(b.team_planes.iter().filter_map(|p| p.keyed))
        .collect();
    let own: Vec<_> = b
        .keyed
        .iter()
        .enumerate()
        .filter(|(at, _)| !used.contains(at))
        .collect();
    if !own.is_empty() {
        let keyed = own
            .into_iter()
            .map(|(_, k)| (k.name.clone(), keyed_value(k)));
        top.push(("keyed", V::D(keyed.collect())));
    }
    Some(one(top))
}

/// The recorded stage block as a value, without what the meta cannot state: the `recomputed` lists of its stages (no member) and a
/// fill of 0 (a plane is zero at the start of every run unless promised another value, so 0 promised and nothing promised are one
/// fact).
fn recorded_block(j: &J) -> V {
    let mut block = from_record(rec(j));
    if let V::D(top) = &mut block {
        if let Some(V::L(stages)) = top.get_mut("stages") {
            for stage in stages {
                if let V::D(held) = stage {
                    held.remove("recomputed");
                }
            }
        }
        if let Some(V::D(planes)) = top.get_mut("team_planes") {
            for plane in planes.values_mut() {
                if let V::D(held) = plane {
                    if held.get("fill") == Some(&V::I(0)) {
                        held.remove("fill");
                    }
                }
            }
        }
    }
    block
}

/// A keyed plane's record as a comparable value: its width, the kept lane axes and the payload shape.
fn keyed_value(keyed: &meta::Keyed) -> V {
    let axes = keyed
        .axis_labels
        .iter()
        .zip(&keyed.axis_extents)
        .map(|(label, extent)| {
            V::L(vec![
                label.clone().map_or(V::N, V::S),
                if *extent < 0 { V::N } else { V::I(*extent) },
            ])
        });
    let pairs = [
        ("width", V::I(keyed.width)),
        ("axes", V::L(axes.collect())),
        (
            "payload",
            V::L(keyed.payload.iter().map(|n| V::I(*n)).collect()),
        ),
    ];
    V::D(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// The cell row and the scatter chunks the answer's units state, when any does.
fn run_facts(answer: &KernelAnswer) -> (Option<i64>, Option<i64>) {
    let first = |value: fn(&meta::Body) -> i64| {
        answer
            .units
            .iter()
            .map(|u| value(&u.meta.body))
            .find(|n| *n != 0)
    };
    (first(|b| b.cell_row), first(|b| b.scatter_chunks))
}

/// Compares the answer's stage block and run facts with the first recorded unit's block and the recorded sidecars.
fn compare_block(
    found: &Found,
    name: &str,
    recorded: &[&J],
    answer: &KernelAnswer,
    tally: &mut Tally,
) {
    let want = recorded
        .first()
        .and_then(|u| u.get("stage_block"))
        .filter(|b| !matches!(b, J::Null))
        .map(recorded_block);
    // a call that failed after rendering never drafted its sidecar: the record has no block to compare
    if want.is_none() && failed(found.call) {
        return;
    }
    tally.blocks += 1;
    let got = answer_block(answer);
    let side = |key: &str| {
        recorded
            .iter()
            .filter_map(|u| u.get("sidecar"))
            .find_map(|c| match c.get(key) {
                Some(J::Num(n)) => Some(*n as i64),
                _ => None,
            })
    };
    let (row, chunks) = (side("cell_row"), side("scatter_chunks"));
    let mut why = String::new();
    if want != got {
        let show = |v: &Option<V>| v.as_ref().map_or("none".to_string(), V::to_string);
        why = format!(
            "stage block: recorded {} / computed {}",
            show(&want),
            show(&got)
        );
    } else if (row, chunks) != run_facts(answer) {
        why = format!(
            "run facts: recorded {row:?} {chunks:?} / computed {:?}",
            run_facts(answer)
        );
    }
    if why.is_empty() {
        tally.blocks_equal += 1;
    } else if tally.block_diffs.len() < found.show {
        tally
            .block_diffs
            .push(format!("{} {name}: {why}", found.id));
    }
}

/// Replays a kernel cut into stages: one request, every unit of the answer against the recorded unit of its name, and the stage
/// block and run facts against the record.
fn replay_staged(
    found: &Found,
    kernel: &J,
    expected: &HashMap<(String, String), Vec<Expected>>,
    ctx: &mut Loaded,
    machines: &mut HashMap<String, String>,
    tally: &mut Tally,
) {
    let name = kernel.get("name").and_then(J::str).unwrap_or("");
    let units = found.call.get("units").map_or(&[][..], J::arr);
    let kernels = found.call.get("kernels").map_or(&[][..], J::arr);
    // a call that builds one kernel twice (plain, then over an active set) records its units twice: the last build is the one compared
    let every = own_units(name, units, kernels);
    let mut recorded = every.clone();
    let names: Vec<&str> = recorded
        .iter()
        .filter_map(|u| u.get("unit").and_then(J::str))
        .collect();
    let last = |at: usize| !names[at + 1..].contains(&names[at]);
    let keep: Vec<bool> = (0..names.len()).map(last).collect();
    let mut keep = keep.into_iter();
    recorded.retain(|_| keep.next().unwrap_or(true));
    let outcome = (|| -> Result<KernelAnswer, String> {
        let facts = facts_all(found, kernel, machines)?;
        if ctx.is_none() {
            *ctx = Some(load(found.call)?);
        }
        let (c, handles) = ctx.as_ref().ok_or("no context")?;
        ask_side(
            found.target,
            *c,
            &request_for(found.call, kernel, facts, handles)?.to_json(),
        )
    })();
    let answer = match outcome {
        Ok(answer) => answer,
        Err(why) => {
            return count_failure((found.id, name), why, tally);
        }
    };
    let test = found.call.get("test").and_then(J::str).unwrap_or("");
    let mut answer = answer;
    keep_recorded(&mut answer, &recorded, failed(found.call));
    // the entry-less source unit of a cut kernel has no record: it is compared through the stage block
    let written: Vec<&UnitAnswer> = answer
        .units
        .iter()
        .filter(|u| !u.texts.is_empty())
        .collect();
    if written.len() != recorded.len() {
        let why = "the answer holds another number of units than the record";
        count_failure((found.id, name), why.into(), tally);
        return;
    }
    for (unit, record) in written.into_iter().zip(&recorded) {
        // of the units built under one name, the record ends with the last build: that is the one
        // compared
        let same = built_as(&every, &unit.meta.name, side_of(found.target));
        let digest = record.get("digest").and_then(J::str);
        let Some(wants) = expected.get(&(found.id.to_string(), unit.meta.name.clone())) else {
            tally.no_text += 1;
            tally.items += 1;
            continue;
        };
        let before = tally.equal;
        compare(
            (found.id, &unit.meta.name),
            digest,
            wants,
            unit,
            Setting {
                show: found.show,
                test,
                unboxed: box_off(Some(record)),
                adapted: adaptations(found.call, kernel),
                ambiguous: ambiguous_record(&same, &unit.meta.digest),
            },
            tally,
        );
        tally.staged_units += 1;
        tally.staged_equal += tally.equal - before;
        tally
            .meta
            .unit(found.id, unit, &answer, record, box_off(Some(record)));
    }
    compare_block(found, name, &recorded, &answer, tally);
}

/// A context holding the call's nodes, and the handles the record's node numbers stand for.
fn load(call: &J) -> Result<(*mut raptor_ir::record::Ctx, Vec<u32>), String> {
    let table = unsafe { &*osprey_host::osprey_v1() };
    let c = (table.ctx_new)();
    declare(c, &manifests(call)).map_err(|e| format!("declare: {e}"))?;
    let handles = make_nodes(c, call, &mut Vals::default()).map_err(|e| format!("make: {e}"))?;
    Ok((c, handles))
}

/// Frees a context of the library.
fn osprey_host_free(c: *mut raptor_ir::record::Ctx) {
    let table = unsafe { &*osprey_host::osprey_v1() };
    unsafe { (table.ctx_free)(c) };
}

/// What the comparison of one unit needs besides the record: how much to show, the test, and the
/// three things the request or the
/// record leaves open.
struct Setting<'a> {
    /// How many differences to show.
    show: usize,
    /// The test that made the call.
    test: &'a str,
    /// Whether the old build wrote the unit unboxed on purpose (its box switch was off).
    unboxed: bool,
    /// What the tool adapted in the request of this unit's kernel (see the header).
    adapted: Vec<&'static str>,
    /// Whether the record cannot say which of several same-name builds the answer is.
    ambiguous: bool,
}

/// Compares one unit with the record and counts it in exactly one category: the text must equal the
/// last recorded render to count
/// as `equal`; an earlier render, an adapted request, a unit unboxed on purpose, an ambiguous
/// record and a unit on the accepted list
/// each have a category of their own.
fn compare(
    (call, name): (&str, &str),
    digest: Option<&str>,
    wants: &[Expected],
    unit: &UnitAnswer,
    set: Setting,
    t: &mut Tally,
) {
    let test = set.test;
    t.items += 1;
    let got = &unit.texts[0];
    let digest_ok = Some(unit.meta.digest.as_str()) == digest;
    let last = wants.len().saturating_sub(1);
    // the last render is the one to match; an earlier one is told apart
    let found = wants
        .iter()
        .rposition(|w| w.text == got.text)
        .unwrap_or(last);
    let want = &wants[found];
    // The record states the old build's box switch at the unit's lowering: off means the unit was written unboxed on purpose, so
    // its unboxed text is listed, not compared against a library that boxes.
    if set.unboxed && want.boxed.is_none() && got.thread_box.is_some() {
        t.box_off.push(format!("{call} {name}"));
        return;
    }
    // A digest that differs from the last build's is a difference, unless an earlier build of the
    // name has it: then the record
    // does not say which build the answer is.
    if set.ambiguous && !digest_ok {
        t.ambiguous.push(format!("{call} {name} [{test}]"));
        return;
    }
    let meta_ok = got.entry == want.entry
        && got.mode == want.mode
        && got.vectorised == want.vectorised
        && got
            .thread_box
            .as_ref()
            .map(|b| (b.axis.unwrap_or(0), b.t, b.stride))
            == want.boxed;
    let mut text_ok = got.text == want.text;
    let mut accepted = false;
    if !text_ok
        && digest_ok
        && meta_ok
        && !want.ukey.is_empty()
        && t.accepted_keys.contains(&want.ukey)
    {
        t.accepted
            .push(format!("{call} {name} {} {test}", want.ukey));
        text_ok = true;
        accepted = true;
    }
    let all_ok = digest_ok && text_ok && meta_ok;
    t.digest_differs += !digest_ok as usize;
    t.text_differs += (digest_ok && !text_ok) as usize;
    t.meta_differs += (digest_ok && text_ok && !meta_ok) as usize;
    if all_ok && !accepted {
        if found != last {
            t.other_render += 1;
            t.earlier.push(format!(
                "{call} {name}: equal to the render of {} (of {} renders; the last is {})",
                want.test,
                wants.len(),
                wants[last].test
            ));
        } else if set.adapted.is_empty() {
            t.equal += 1;
        } else {
            t.adapted += 1;
            for reason in &set.adapted {
                *t.adapted_by.entry(reason).or_default() += 1;
            }
        }
    }
    t.boxed += want.boxed.is_some() as usize;
    t.boxed_equal += (want.boxed.is_some() && all_ok) as usize;
    if !all_ok && t.shown < set.show {
        t.shown += 1;
        let how = if !digest_ok {
            format!("digest; text {}", first_diff(&want.text, &got.text))
        } else if !text_ok {
            first_diff(&want.text, &got.text)
        } else {
            "meta".into()
        };
        println!(
            "DIFF {call} {name} key {} [{test}] ({} renders): {how}",
            want.ukey,
            wants.len()
        );
    }
}

/// Prints the one reconciliation line of a corpus: every outcome counted once equals the sum of the
/// categories printed below.
fn reconcile(t: &Tally) {
    let w = &t.warp;
    let refused = t.refused.values().sum::<usize>() + w.refused.values().sum::<usize>();
    let parts = [
        ("equal", t.equal + w.equal),
        ("equal with an adapted request", t.adapted),
        ("equal to an earlier render only", t.other_render),
        ("accepted", t.accepted.len()),
        ("box switch", t.box_off.len()),
        ("ambiguous record", t.ambiguous.len()),
        ("patched", t.patched),
        ("own card", t.own_card.len()),
        ("own fusion row", t.fusion_overridden.len()),
        ("no recorded text", t.no_text + t.segment_no_text),
        (
            "different",
            t.text_differs + t.digest_differs + t.meta_differs + w.different,
        ),
        ("refused", refused),
        ("request not built", t.unbuilt.values().sum::<usize>()),
    ];
    let sum: usize = parts.iter().map(|part| part.1).sum();
    let terms: Vec<String> = parts.iter().map(|(n, c)| format!("{n} {c}")).collect();
    println!(
        "  reconciliation: outcomes {} = {} : {}",
        t.items,
        terms.join(" + "),
        if sum == t.items {
            "adds up"
        } else {
            "DOES NOT ADD UP"
        }
    );
}

/// Prints the summary of one corpus.
fn print(label: &str, t: &Tally, side: &str) {
    if side == "host" {
        println!(
            "kreplay {label}: calls {} (error {}, memo {}), plain calls {}",
            t.calls, t.errors, t.memo, t.plain_calls
        );
    }
    println!(
        "{side} units of plain, boxed and split kernels {}: equal {}, text different {}, digest different {}, meta different {}, no recorded text {}",
        t.items, t.equal, t.text_differs, t.digest_differs, t.meta_differs, t.no_text
    );
    reconcile(t);
    println!(
        "  of them written with a thread box: {} (equal {})",
        t.boxed, t.boxed_equal
    );
    println!(
        "  of them units of a chain: {} (equal {}), chains with other planes between the stages: {}",
        t.chain_units,
        t.chain_equal,
        t.chain_planes_differ.len()
    );
    println!(
        "  records without auto_k_max: {}; without fusion_rows: {}; derivative kernels without derived_planes: {}; device records without image_format: {}",
        t.without[0], t.without[1], t.without[2], t.without[3]
    );
    println!(
        "  built under a test's own fusion row, which the record does not hold, listed: {}",
        t.fusion_overridden.len()
    );
    for r in t
        .fusion_overridden
        .iter()
        .take(10)
        .chain(t.chain_planes_differ.iter().take(10))
    {
        println!("  {}", &r[..r.len().min(1200)]);
    }
    println!(
        "  of them units of a segmented split: {} (equal {})",
        t.segment_units, t.segment_equal
    );
    println!(
        "  accepted, text differs (key on the accepted list): {}",
        t.accepted.len()
    );
    for a in &t.accepted {
        println!("    {a}");
    }
    println!(
        "  units of staged kernels: {} (equal {}); stage blocks and run facts: {} (equal {})",
        t.staged_units, t.staged_equal, t.blocks, t.blocks_equal
    );
    for d in &t.block_diffs {
        println!("    {d}");
    }
    println!(
        "  segmented splits: {} units recorded, {} compared, {} without recorded text; patched by its test (not compared): {}",
        t.segment_recorded, t.segment_units, t.segment_no_text, t.patched
    );
    println!(
        "  equal only to an earlier recorded render of the unit, not in equal: {}",
        t.other_render
    );
    for e in &t.earlier {
        println!("    {e}");
    }
    println!(
        "  equal to the last render but asked with an adapted request, not in equal: {} ({})",
        t.adapted,
        t.adapted_by
            .iter()
            .map(|(why, n)| format!("{why} {n}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "  ambiguous record (the last same-name build differs, an earlier one matches; \
         the recorder would need the index of the build on each unit and text): {}",
        t.ambiguous.len()
    );
    for a in &t.ambiguous {
        println!("    {a}");
    }
    let sum = |m: &BTreeMap<String, usize>| m.values().sum::<usize>();
    println!(
        "refused by the library: {}; request not built: {}",
        sum(&t.refused),
        sum(&t.unbuilt)
    );
    for (why, n) in t.refused.iter().chain(t.unbuilt.iter()) {
        println!("  {n:>6}  {why}");
    }
    for line in &t.failed_ids {
        println!("    {line}");
    }
    println!(
        "unboxed by the old environment switch (the record's box_env is 0), listed: {}",
        t.box_off.len()
    );
    for r in t.box_off.iter().take(20) {
        println!("  {r}");
    }
    println!(
        "built under a card of the test's own, which the request does not carry, listed: {}",
        t.own_card.len()
    );
    for r in &t.own_card {
        println!("  {r}");
    }
    println!("no recorded text: {}", t.no_text_ids.len());
    for id in &t.no_text_ids {
        println!("  {id}");
    }
    let w = &t.warp;
    println!(
        "warp per lane kernels {}: digest equal {}, stage block equal {}, text equal {}, refused {}",
        w.asked,
        w.digest_equal,
        w.block_equal,
        w.text_equal,
        w.refused.values().sum::<usize>()
    );
    for (why, n) in &w.refused {
        println!("  {n:>6}  {why}");
    }
    for line in w.diffs.iter().take(30) {
        println!("  {line}");
    }
    println!(
        "kernels built for a target the library does not serve: {}",
        t.unserved.values().sum::<usize>()
    );
    for (why, n) in &t.unserved {
        println!("  {n:>6}  {why}");
    }
    println!("kernels of other shapes, by shape (the work lists of the later slices):");
    for (shape, n) in &t.shapes {
        println!("  {n:>6}  {shape}");
    }
}

/// Whether the kernel is a chain: it lists member kernels that carry their own sinks (a fused group lists lanes that do not).
fn is_chain(kernel: &J) -> bool {
    let members = kernel.get("members").map_or(&[][..], J::arr);
    members.first().is_some_and(|m| m.get("sinks").is_some())
}

/// The chain's member kernels as the request states them.
fn chain_members(kernel: &J, handles: &[u32]) -> Result<Vec<Member>, String> {
    let list = kernel.get("members").map_or(&[][..], J::arr);
    list.iter()
        .map(|member| {
            Ok(Member {
                name: member
                    .get("name")
                    .and_then(J::str)
                    .unwrap_or("")
                    .to_string(),
                sinks: items_of_nodes(member.get("sinks"), handles),
                declared: items(member.get("declared"))
                    .iter()
                    .map(declared_of)
                    .collect::<Result<_, String>>()?,
                force: text(member.get("force"))
                    .and_then(|f| ACCESSES.iter().find(|a| a.name() == f).copied()),
                exempt: items(member.get("exempt"))
                    .iter()
                    .filter_map(|e| text(Some(e)).map(String::from))
                    .collect(),
            })
        })
        .collect()
}

/// The node handles a recorded list of node references stands for.
fn items_of_nodes(list: Option<&J>, handles: &[u32]) -> Vec<u32> {
    list.map_or(&[][..], J::arr)
        .iter()
        .filter_map(|n| handles.get(n.get("n")?.int() as usize).copied())
        .collect()
}

/// The units of a chain in the call's records: the chain's own name, then its name and the piece's place.
fn chain_units<'a>(name: &'a str, units: &'a [J]) -> impl Iterator<Item = &'a J> {
    units.iter().filter(move |u| {
        let unit = u.get("unit").and_then(J::str).unwrap_or("");
        unit == name
            || unit
                .strip_prefix(name)
                .and_then(|rest| rest.strip_prefix('_'))
                .is_some_and(|place| !place.is_empty() && place.bytes().all(|b| b.is_ascii_digit()))
    })
}

/// Replays a chain: one request with its members, and every unit of the answer against the recorded unit of its place, then the
/// planes between the stages against the recorded stage block.
fn replay_chain(
    found: &Found,
    kernel: &J,
    expected: &HashMap<(String, String), Vec<Expected>>,
    (ctx, machines): (&mut Loaded, &mut HashMap<String, String>),
    tally: &mut Tally,
) {
    let name = kernel.get("name").and_then(J::str).unwrap_or("");
    let units = found.call.get("units").map_or(&[][..], J::arr);
    let recorded: Vec<&J> = chain_units(name, units).collect();
    let outcome = (|| -> Result<KernelAnswer, String> {
        let facts = facts_all(found, kernel, machines)?;
        if ctx.is_none() {
            *ctx = Some(load(found.call)?);
        }
        let (c, handles) = ctx.as_ref().ok_or("no context")?;
        let mut request = request_for(found.call, kernel, facts, handles)?;
        let members = chain_members(kernel, handles)?;
        // a chain the old build composed through the fused-lanes hook is fused lanes
        request.body = match kernel.get("fused_body") {
            Some(J::Bool(true)) => Body::Fused(members),
            _ => Body::Chain(members),
        };
        ask_side(found.target, *c, &request.to_json())
    })();
    let answer = match outcome {
        Ok(answer) => answer,
        Err(why) => return count_failure((found.id, name), why, tally),
    };
    let mut answer = answer;
    keep_recorded(&mut answer, &recorded, failed(found.call));
    let written: Vec<&UnitAnswer> = answer.units.iter().filter(|u| !is_source(u)).collect();
    if written.len() != recorded.len() {
        let test = found.call.get("test").and_then(J::str).unwrap_or("");
        if test.contains("another_register_file") && call_without_fusion_row(found.call) {
            tally.fusion_overridden.push(format!("{} {name}", found.id));
            tally.items += 1;
            return;
        }
        let why = "the chain's answer holds another number of units than the record";
        count_failure((found.id, name), why.into(), tally);
        return;
    }
    let test = found.call.get("test").and_then(J::str).unwrap_or("");
    for (unit, record) in written.into_iter().zip(&recorded) {
        let digest = record.get("digest").and_then(J::str);
        let Some(wants) = expected.get(&(found.id.to_string(), unit.meta.name.clone())) else {
            tally.no_text += 1;
            tally.items += 1;
            continue;
        };
        let before = tally.equal;
        compare(
            (found.id, &unit.meta.name),
            digest,
            wants,
            unit,
            Setting {
                show: found.show,
                test,
                unboxed: box_off(Some(record)),
                adapted: adaptations(found.call, kernel),
                ambiguous: false,
            },
            tally,
        );
        tally.chain_units += 1;
        tally.chain_equal += tally.equal - before;
        tally
            .meta
            .unit(found.id, unit, &answer, record, box_off(Some(record)));
    }
    compare_block(found, name, &recorded, &answer, tally);
}

fn main() {
    if std::env::var_os("OSPREY_FACTS_DIR").is_none() {
        std::env::set_var(
            "OSPREY_FACTS_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/facts"),
        );
    }
    let args: Vec<String> = std::env::args().collect();
    if let Some(at) = args.iter().position(|a| a == "--emit-meta") {
        let (file, out) = (args.get(at + 1), args.get(at + 2));
        return oracle::emit(
            Path::new(file.expect("--emit-meta <calls> <folder>")),
            Path::new(out.expect("--emit-meta <calls> <folder>")),
            None,
        );
    }
    if let Some(at) = args.iter().position(|a| a == "--emit") {
        let (id, out) = (args.get(at + 1), args.get(at + 2));
        let (id, out) = (
            id.expect("--emit <call id> <folder>"),
            out.expect("--emit <call id> <folder>"),
        );
        let corpus = Path::new(
            args.get(1)
                .expect("kreplay <corpus dir> --emit <call id> <folder>"),
        );
        let mut files: Vec<_> = std::fs::read_dir(corpus)
            .expect("the corpus folder")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
            .filter(|p| {
                p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with('k'))
            })
            .collect();
        files.sort();
        for file in files {
            oracle::emit(&file, Path::new(out), Some(id));
        }
        return;
    }
    let dir = args
        .get(1)
        .expect("usage: kreplay <corpus dir> [--show k] [--accepted corpus_accepted.tsv]");
    let show = args
        .iter()
        .position(|a| a == "--show")
        .and_then(|i| args.get(i + 1))
        .and_then(|n| n.parse().ok())
        .unwrap_or(5);
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with('k') && n.ends_with(".jsonl"))
        })
        .collect();
    files.sort();
    let mut tallies = [Tally::default(), Tally::default()];
    if let Some(list) = args
        .iter()
        .position(|a| a == "--accepted")
        .and_then(|i| args.get(i + 1))
    {
        let keys: HashSet<String> = std::fs::read_to_string(list)
            .expect("the accepted list")
            .lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| l.split('\t').next().map(String::from))
            .collect();
        for t in &mut tallies {
            t.accepted_keys.clone_from(&keys);
        }
    }
    for kpath in &files {
        let tpath = kpath.with_file_name(
            kpath
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .replacen('k', "t", 1),
        );
        replay(kpath, &tpath, show, &mut tallies);
    }
    for (tally, (_, side)) in tallies.iter().zip(SIDES) {
        print(dir, tally, side);
        tally.meta.print(side);
    }
}
