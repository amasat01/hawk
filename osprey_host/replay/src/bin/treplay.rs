//! treplay <texts.jsonl> [--migrate] [--target host] [--limit n] [--only <substring of unit name or test id>] [--show k] [--ladder]
//! Replays the recorded render calls: each unit's nodes are made in a fresh context through the C door, the walk is rebuilt and
//! compared with the recorded one, and the unit's text is rendered by the writer crate and compared with the recorded text.
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
use aether_writers::contract::old::typed_of;
use aether_writers::contract::types::{Isa, Machine, Target, Unit};
use aether_writers::contract::{Contract, Json};
use aether_writers::refuse::Refusal;
use aether_writers::{cuda_render, entry_body, host_render, RxCtx};
use osprey_lowering::facts;
use raptor_ir::door::rx_ctx_new;
use raptor_ir::record::Ctx;
use raptor_support::corpus::replay_ctx;
use raptor_support::corpus::{
    declare, err, make_nodes, manifests, tagged_str, val, walk_options, with_analysis_nodes, Vals,
    WalkOpts,
};
use raptor_support::migrate::{self, Changed};
use raptor_support::safe::rx_ctx_free;
use raptor_support::safe::{rx_canonical, rx_digest, rx_walk_field};
extern crate raptor_wings; // links the wing analyses the writers call
use rcore::json::{self, J};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::io::BufRead;
use std::time::Instant;

/// The named kernels the `--ladder` mode replays, one unit each, in order of growing demands on the writers.
const LADDER: [&str; 7] = [
    "axpb",
    "vec3_scale",
    "newton",
    "share",
    "share_2",
    "gram",
    "grad",
];

/// The text of a walk field, as the door gives it.
fn walk_field(c: *mut Ctx, w: u32, f: &str) -> String {
    let f = CString::new(f).unwrap();
    let mut len = 0usize;
    rx_walk_field(c, w, f.as_ptr(), std::ptr::null_mut(), 0, &mut len);
    let mut buf = vec![0 as std::ffi::c_char; len + 1];
    rx_walk_field(c, w, f.as_ptr(), buf.as_mut_ptr(), len + 1, &mut len);
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}
/// The text quoted as a JSON string.
fn js(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// The contract: a tagged value as plain JSON (records are objects with "$type", node references {"$node": handle}, module-level
/// references {"$ref": "module:name"}, tuples and lists arrays, maps with text keys objects, other maps {"$map": [[k, v], ...]}).
fn contract(j: &J, h: &[u32], o: &mut String, types: &mut HashMap<String, usize>) {
    let items = |a: &[J], o: &mut String, types: &mut HashMap<String, usize>| {
        o.push('[');
        for (i, x) in a.iter().enumerate() {
            if i > 0 {
                o.push(',')
            }
            contract(x, h, o, types)
        }
        o.push(']');
    };
    match j {
        J::Null => o.push_str("null"),
        J::Bool(b) => o.push_str(&b.to_string()),
        J::Obj(_) => {
            if let Some(x) = j.get("i") {
                o.push_str(x.str().unwrap_or("0"))
            } else if let Some(x) = j.get("f") {
                let t = x.str().unwrap_or("0");
                if t.parse::<f64>().is_ok_and(f64::is_finite) {
                    o.push_str(t)
                } else {
                    o.push_str(&format!("{{\"$float\":{}}}", js(t)))
                }
            } else if let Some(x) = j.get("s") {
                o.push_str(&js(x.str().unwrap_or("")))
            } else if let Some(x) = j.get("t").or(j.get("l")).or(j.get("fs")) {
                items(x.arr(), o, types)
            } else if let Some(x) = j.get("mp") {
                contract(x, h, o, types)
            } else if let Some(x) = j.get("d") {
                if x.arr()
                    .iter()
                    .all(|p| p.arr().first().is_some_and(|k| k.get("s").is_some()))
                {
                    o.push('{');
                    for (i, p) in x.arr().iter().enumerate() {
                        if i > 0 {
                            o.push(',')
                        }
                        o.push_str(&format!(
                            "{}:",
                            js(p.arr()[0].get("s").and_then(J::str).unwrap_or(""))
                        ));
                        contract(&p.arr()[1], h, o, types);
                    }
                    o.push('}');
                } else {
                    let pairs: Vec<J> = x
                        .arr()
                        .iter()
                        .map(|p| J::Obj(vec![("t".into(), J::Arr(p.arr().to_vec()))]))
                        .collect();
                    o.push_str("{\"$map\":");
                    items(&pairs, o, types);
                    o.push('}')
                }
            } else if j.obj().is_empty() {
                o.push_str("{}")
            } else if let Some(x) = j.get("n") {
                o.push_str(&format!(
                    "{{\"$node\":{}}}",
                    h.get(x.int() as usize).map_or(-1, |n| *n as i64)
                ))
            } else if let Some(x) = j.get("fn") {
                o.push_str(&format!("{{\"$ref\":{}}}", js(x.str().unwrap_or(""))))
            } else if let Some(r) = j.get("r") {
                let name = r.str().unwrap_or("?");
                *types.entry(name.to_string()).or_default() += 1;
                o.push_str(&format!("{{\"$type\":{}", js(name)));
                for (k, x) in j.get("v").map_or(&[][..], |v| v.obj()) {
                    o.push_str(&format!(",{}:", js(k)));
                    contract(x, h, o, types)
                }
                o.push('}');
            } else {
                // a tag outside the grammar's list (a lock, a subclass of dict): its fields as they are
                *types
                    .entry(format!(
                        "other({})",
                        j.obj()
                            .iter()
                            .map(|(k, _)| k.as_str())
                            .collect::<Vec<_>>()
                            .join(",")
                    ))
                    .or_default() += 1;
                o.push('{');
                for (i, (k, x)) in j.obj().iter().enumerate() {
                    if i > 0 {
                        o.push(',')
                    }
                    o.push_str(&format!("{}:", js(k)));
                    contract(x, h, o, types)
                }
                o.push('}');
            }
        }
        _ => o.push_str("null"),
    }
}

/// Writes what a unit's render gave to `<folder>/<target>/<key>`: `.txt` and `.body.txt` for a text (kept when the file
/// is already there, so the first path that wrote it stays), `.refused.txt` for a refusal. No folder: nothing is written.
fn dump_unit(
    folder: &Option<String>,
    target: &str,
    key: &str,
    text: Option<(&str, &str)>,
    refused: Option<&str>,
) {
    let Some(folder) = folder else { return };
    let dir = std::path::Path::new(folder).join(target);
    std::fs::create_dir_all(&dir).expect("create the dump folder");
    if let Some((text, body)) = text {
        for (suffix, content) in [(".txt", text), (".body.txt", body)] {
            let file = dir.join(format!("{key}{suffix}"));
            if !file.exists() {
                std::fs::write(file, content).expect("write the dump");
            }
        }
    }
    if let Some(message) = refused {
        std::fs::write(dir.join(format!("{key}.refused.txt")), message).expect("write the dump");
    }
}

/// The reason text of a refusal, for the counts that group refusals by their text.
fn reason(refusal: Refusal) -> String {
    refusal.reason
}

/// The compiler identity of a clang the facts oracle asks with, and of a GCC, as the recording's build had.
const COMPILER_IDENTITIES: [&str; 2] = [
    "c++ || Apple clang version 21.0.0 (clang-2100.0.123.102) || 0",
    "g++ || g++ (GCC) 13.2.0 || 0",
];

/// The first value under `key` anywhere in a recorded value.
fn find<'a>(json: &'a Json, key: &str) -> Option<&'a Json> {
    match json {
        Json::Obj(members) => {
            members
                .iter()
                .find_map(|(k, v)| if k == key { Some(v) } else { find(v, key) })
        }
        Json::Arr(items) => items.iter().find_map(|v| find(v, key)),
        _ => None,
    }
}

/// The machine the recording states for its build (its instruction set, cache sizes and host profile in effect) and the
/// compiler identity that fits it (GCC when the recording says so, or when it builds the vector math profile on x86): what a
/// caller building for that machine would pass to the facts entry, so that the rules are what the oracle tests.
fn recorded_machine(unit: &Unit, contract: &Contract) -> (facts::Stated, &'static str) {
    let isa = unit.machine.isa.as_ref().map(|isa| match isa {
        Isa::Arm64 => "arm64".to_string(),
        Isa::X86_64 => "x86_64".to_string(),
    });
    let host_profile = find(&contract.0, "host_profile").and_then(Json::str);
    let gcc = matches!(find(&contract.0, "arm64_gcc"), Some(Json::Bool(true)))
        || (isa.as_deref() == Some("x86_64") && host_profile.is_some_and(|p| p.contains("vector")));
    let stated = facts::Stated {
        isa,
        vector_isa: find(&contract.0, "vector_isa")
            .and_then(Json::str)
            .map(String::from),
        caches: unit.machine.caches.clone(),
        host_profile: host_profile.map(String::from),
        headers: Some("roots".into()),
        device_arch: None,
        ..Default::default()
    };
    (stated, COMPILER_IDENTITIES[gcc as usize])
}

/// The machine record of the facts answer for the machine the recording states, as the typed wire reads it: the answer replaces
/// the `machine` of the unit's own JSON, which is read back.
fn facts_machine(unit: &Unit, contract: &Contract) -> Result<Machine, String> {
    let (stated, compiler) = recorded_machine(unit, contract);
    let dir = std::path::Path::new(osprey_replay::CPU_FACTS);
    let answer = facts::answer(
        dir,
        &osprey_cpu_lib::Cpu,
        compiler,
        &facts::probe(),
        &stated,
    )
    .map_err(String::from)?;
    let Contract(mut top) = Contract::parse(&unit.to_json()).map_err(|refusal| refusal.reason)?;
    let Json::Obj(members) = &mut top else {
        return Err("the unit is not an object".into());
    };
    let slot = members
        .iter_mut()
        .find(|(key, _)| key == "machine")
        .ok_or("the unit has no machine")?;
    slot.1 = Contract::parse(&answer)
        .map_err(|refusal| refusal.reason)?
        .0;
    // the build commands are the caller's, not a member of the typed machine record
    if let Json::Obj(stated) = &mut slot.1 {
        stated.retain(|(key, _)| key != "host_build");
    }
    Unit::from_value(&top)
        .map(|read| read.machine)
        .map_err(|refusal| refusal.reason)
}

/// What differs between the recorded machine record and the facts answer, one clause per field; empty when they agree. A select
/// weight the converter has no recorded answer for (the unit never asked) is not compared.
fn machine_differences(recorded: &Machine, ours: &Machine) -> String {
    let mut clauses = vec![];
    let mut clause = |name: &str, differs: bool, want: String, got: String| {
        if differs {
            clauses.push(format!("{name}: recorded {want}, answer {got}"))
        }
    };
    clause(
        "isa",
        recorded.isa != ours.isa,
        format!("{:?}", recorded.isa),
        format!("{:?}", ours.isa),
    );
    clause(
        "exact_libm",
        recorded.exact_libm != ours.exact_libm,
        recorded.exact_libm.to_string(),
        ours.exact_libm.to_string(),
    );
    clause(
        "routed_pow",
        recorded.routed_pow != ours.routed_pow,
        recorded.routed_pow.to_string(),
        ours.routed_pow.to_string(),
    );
    clause(
        "caches",
        recorded.caches != ours.caches,
        format!("{:?}", recorded.caches),
        format!("{:?}", ours.caches),
    );
    let (want, got) = (&recorded.select_weight, &ours.select_weight);
    let weight = |recorded: Option<f64>, ours: Option<f64>| recorded.is_some() && recorded != ours;
    let differs = weight(want.float64, got.float64) || weight(want.float32, got.float32);
    clause(
        "select_weight",
        differs,
        format!("{want:?}"),
        format!("{got:?}"),
    );
    clauses.join("; ")
}

/// How many rendered units the typed contract converts, and the refusal names of the others.
#[derive(Default)]
struct Typed {
    /// Units counted.
    total: usize,
    /// Refusal names of the typed contract's conversion, with counts.
    refused: HashMap<String, usize>,
    /// Refusal names of the typed wire's round trip, with counts.
    wire_refused: HashMap<String, usize>,
    /// Units whose machine record equals the facts answer for this machine.
    facts_equal: usize,
    /// Each way the facts answer differs from the converter's machine record, with the number of units it holds for.
    facts_diffs: HashMap<String, usize>,
    /// Bodies the writers rendered themselves whose recorded needs were compared with their text.
    #[cfg(feature = "shadow")]
    needs_bodies: usize,
    /// Each disagreement: the unit, the target and what differs.
    #[cfg(feature = "shadow")]
    needs_diffs: Vec<String>,
}

impl Typed {
    /// Counts one unit: its contract text and the target it renders for.
    fn note(&mut self, json: &str, target: &str) {
        self.total += 1;
        if target == "host" {
            self.machine(json)
        }
        match aether_writers::unit::entry::wire_refusal(json, target) {
            Ok(None) => {}
            Ok(Some(why)) => *self.wire_refused.entry(why).or_default() += 1,
            Err(why) => {
                *self
                    .wire_refused
                    .entry(format!("unreadable: {why}"))
                    .or_default() += 1
            }
        }
        match aether_writers::unit::entry::typed_refusal(json, target) {
            Ok(None) => {}
            Ok(Some(why)) => *self.refused.entry(why).or_default() += 1,
            Err(why) => {
                *self
                    .refused
                    .entry(format!("unreadable: {why}"))
                    .or_default() += 1
            }
        }
    }

    /// Compares the machine record the converter derives from the recording with the facts answer for the machine the recording
    /// states: the answer is read back through the typed wire, as a caller's request would carry it. A unit the converter
    /// refuses has no record to compare.
    fn machine(&mut self, json: &str) {
        let Ok(contract) = aether_writers::contract::Contract::parse(json) else {
            return;
        };
        let Ok(unit) = typed_of(&contract, Target::Host) else {
            return;
        };
        match facts_machine(&unit, &contract) {
            Ok(ours) => match machine_differences(&unit.machine, &ours) {
                none if none.is_empty() => self.facts_equal += 1,
                some => *self.facts_diffs.entry(some).or_default() += 1,
            },
            Err(why) => {
                *self
                    .facts_diffs
                    .entry(format!("answer refused: {why}"))
                    .or_default() += 1
            }
        }
    }

    /// Compares the needs recorded while the body of one unit was written with the needs its text shows.
    #[cfg(feature = "shadow")]
    fn needs(&mut self, c: *mut RxCtx, w: u32, json: &str, target: &str, unit: &str) {
        if let Ok(Some(diffs)) = aether_writers::unit::entry::needs_check(c, w, json, target) {
            self.needs_bodies += 1;
            for d in diffs {
                self.needs_diffs.push(format!("{unit} ({target}): {d}"))
            }
        }
    }

    /// Without the feature nothing is compared.
    #[cfg(not(feature = "shadow"))]
    fn needs(&mut self, _c: *mut RxCtx, _w: u32, _json: &str, _target: &str, _unit: &str) {}

    /// Prints the totality line, the wire line and the refusal names with counts; with the `shadow` feature, the needs line too.
    fn print(&self) {
        let refused: usize = self.refused.values().sum();
        println!(
            "typed contract: {} of {} units converted, {refused} refused",
            self.total - refused,
            self.total
        );
        let wire: usize = self.wire_refused.values().sum();
        println!(
            "typed wire: {} of {} units round-trip",
            self.total - wire,
            self.total
        );
        for (why, n) in &self.wire_refused {
            println!("  typed wire refusal {n}: {why}")
        }
        let different: usize = self.facts_diffs.values().sum();
        println!(
            "facts oracle: machine record equals the facts answer on {} of {} converted units, {different} differ",
            self.facts_equal,
            self.facts_equal + different
        );
        for (why, n) in &self.facts_diffs {
            println!("  facts differ on {n}: {why}")
        }
        let mut names: Vec<_> = self.refused.iter().collect();
        names.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        for (why, n) in names {
            println!("  typed refusal {n}: {why}")
        }
        #[cfg(feature = "shadow")]
        self.print_needs();
    }

    /// Prints the needs comparison: the bodies compared and each disagreement.
    #[cfg(feature = "shadow")]
    fn print_needs(&self) {
        println!(
            "needs: {} bodies, {} disagreements",
            self.needs_bodies,
            self.needs_diffs.len()
        );
        for d in self.needs_diffs.iter().take(20) {
            println!("  needs differ: {d}")
        }
    }
}

/// Everything one replay counts, printed at its end.
#[derive(Default)]
struct Tally {
    /// What the typed contract made of the units this run rendered.
    typed: Typed,
    /// The `--dump` folder.
    dump: Option<String>,
    /// Units read.
    units: usize,
    /// Units the recording excludes.
    excluded: usize,
    /// Units the recording marks as raised by the oracle.
    raised: usize,
    /// Units rendered.
    ran: usize,
    /// Walks loaded into the context.
    loaded: usize,
    /// Walks whose digest equals the recorded one, which the context computes with the plane-shape fields removed from every key.
    digest_eq: usize,
    /// Walks whose argument spec equals the recorded one.
    spec_eq: usize,
    /// Walks whose access equals the recorded one.
    access_eq: usize,
    /// Team-stage bodies equal to the recorded ones.
    team_eq: usize,
    /// Team-stage units.
    team_n: usize,
    /// Bodies equal to the recorded ones.
    body_eq: usize,
    /// Lines the recording reader refused.
    unparsed: usize,
    /// Texts that differ from the recorded ones.
    text_diff: usize,
    /// Bodies that differ from the recorded ones.
    body_diff: usize,
    /// The first construct not written yet, with counts.
    features: HashMap<String, usize>,
    /// Walks with an access op set.
    access_op: usize,
    /// Texts equal to the recorded ones.
    text_eq: usize,
    /// Units this build does not write.
    not_built: usize,
    /// Units whose render failed.
    render_err: usize,
    /// Units the canonical form refused.
    canon_refused: usize,
    /// Unit names by refusal.
    names: HashMap<String, Vec<String>>,
    /// Unit names by reason of exclusion.
    excluded_by: HashMap<String, Vec<String>>,
    /// Refusals of making a walk, with counts.
    make_refused: HashMap<String, usize>,
    /// Render failures, with counts.
    render_errs: HashMap<String, usize>,
    /// Canonical-form refusals, with counts.
    canon_errs: HashMap<String, usize>,
    /// Contract key sets seen, with counts.
    keysets: HashMap<String, usize>,
    /// Digests that more than one option set reproduces.
    ambiguous: usize,
    /// Units without a kernel walk.
    no_walk: usize,
    /// Contract types seen, with counts.
    types: HashMap<String, usize>,
    /// Differing lines shown so far.
    shown: usize,
    /// The `--accepted` list: (unit key, unit name, reason).
    accepted: Vec<(String, String, String)>,
    /// Keys of listed units whose text differed, and how many such units there were.
    accepted_hit: Vec<String>,
    /// Texts that differ but are on the accepted list.
    accepted_diff: usize,
    /// Keys of listed units this run reached (a listed key of another oracle is not stale).
    accepted_seen: Vec<String>,
}

/// The units of an `--accepted` file: one `key<TAB>name<TAB>reason` per line, `#` lines skipped.
fn read_accepted(path: &str) -> Vec<(String, String, String)> {
    let text = std::fs::read_to_string(path).expect("open accepted list");
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut f = l.splitn(3, '\t');
            let mut next = || f.next().unwrap_or("").to_string();
            (next(), next(), next())
        })
        .collect()
}

/// The reason a record is left out of the replay: its `exclude` text, or none (a record may also be unmirrorable).
fn excluded_reason(rec: &J) -> Option<String> {
    match rec.get("exclude") {
        Some(J::Str(r)) if !r.is_empty() => Some(r.clone()),
        Some(J::Bool(true)) => Some("excluded".to_string()),
        _ => {
            if matches!(rec.get("unmirrorable"), Some(J::Bool(true))) {
                Some("unmirrorable".to_string())
            } else {
                None
            }
        }
    }
}

/// A recorded value as JSON text, unchanged.
fn raw(j: &J, o: &mut String) {
    match j {
        J::Null => o.push_str("null"),
        J::Bool(b) => o.push_str(if *b { "true" } else { "false" }),
        J::Num(n) => o.push_str(&if n.fract() == 0.0 && n.abs() < 1e15 {
            format!("{}", *n as i64)
        } else {
            format!("{n}")
        }),
        J::Str(x) => o.push_str(&js(x)),
        J::Arr(v) => {
            o.push('[');
            for (i, x) in v.iter().enumerate() {
                if i > 0 {
                    o.push(',')
                }
                raw(x, o)
            }
            o.push(']')
        }
        J::Obj(v) => {
            o.push('{');
            for (i, (k, x)) in v.iter().enumerate() {
                if i > 0 {
                    o.push(',')
                }
                o.push_str(&js(k));
                o.push(':');
                raw(x, o)
            }
            o.push('}')
        }
    }
}

/// The record's `facts` (facts of the machine the old writer read from inside itself) as a top-level contract key.
fn facts_key(rec: &J, json: &mut String) {
    if let Some(f) = rec.get("facts").filter(|f| !matches!(f, J::Null)) {
        json.push_str(",\"facts\":");
        raw(f, json)
    }
}

/// A unit's name with its first test id, as listed under a refusal by `--names`.
fn label(rec: &J) -> String {
    let test = rec
        .get("tests")
        .and_then(|t| t.arr().first())
        .and_then(J::str)
        .unwrap_or("");
    format!(
        "{} ({test})",
        rec.get("unit").and_then(J::str).unwrap_or("?")
    )
}

/// Prints the unit names listed under every refusal whose text contains `wanted`.
fn print_names(names: &HashMap<String, Vec<String>>, wanted: Option<&String>) {
    let Some(wanted) = wanted else { return };
    let mut keys: Vec<_> = names
        .keys()
        .filter(|k| k.contains(wanted.as_str()))
        .collect();
    keys.sort();
    for k in keys {
        println!("names under `{k}`:");
        for n in &names[k] {
            println!("  {n}")
        }
    }
}

/// The first line that differs between two texts, as a message.
fn first_diff(want: &str, got: &str) -> String {
    let (w, g): (Vec<&str>, Vec<&str>) = (want.lines().collect(), got.lines().collect());
    let i = (0..w.len().max(g.len()))
        .find(|&i| w.get(i) != g.get(i))
        .unwrap_or(0);
    format!(
        "line {}: recorded {:?} computed {:?}",
        i + 1,
        w.get(i),
        g.get(i)
    )
}

/// The `n` most frequent entries of a tally (`--lines n` sets n).
fn top(m: &HashMap<String, usize>, n: usize) {
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (k, c) in v.into_iter().take(n) {
        println!("  {c} x {k}")
    }
}

/// The elements of a sequence value; a tuple subclass is recorded as {"base", "of"}.
/// A kernel the recorder could not encode is an object of the record: its attributes are the kernel's fields; a wrapper object
/// (an active-set unit) holds the kernel under `_kernel` and its own walk.
fn kernel_object(low: Option<&J>, inp: &J) -> Option<J> {
    let a = low
        .and_then(|l| l.get("kernel"))
        .and_then(|k| k.get("o"))
        .and_then(|o| inp.get("objs")?.arr().get(o.int() as usize))
        .and_then(|ob| ob.get("a"))?;
    let fields = match a.get("_kernel").and_then(|k| k.get("v")) {
        Some(k) => k
            .obj()
            .iter()
            .filter(|(n, _)| n != "walk")
            .cloned()
            .chain(a.get("walk").map(|w| ("walk".to_string(), w.clone())))
            .collect(),
        None => a.obj().to_vec(),
    };
    Some(J::Obj(vec![
        ("r".into(), J::Str("Kernel".into())),
        ("v".into(), J::Obj(fields)),
    ]))
}
/// The elements of an optional JSON list, empty when absent.
fn elems(j: Option<&J>) -> &[J] {
    match j {
        Some(j) => j
            .get("t")
            .or(j.get("l"))
            .or(j.get("of"))
            .map_or(&[][..], |x| {
                if x.get("t").is_some() || x.get("l").is_some() {
                    elems(Some(x))
                } else {
                    x.arr()
                }
            }),
        None => &[],
    }
}

/// One unit: returns its ladder line.
fn unit(rec: &J, render: bool, show: usize, t: &mut Tally) -> String {
    let inp_all = with_analysis_nodes(rec.get("inputs").unwrap(), rec);
    let inp = &inp_all;
    let name = rec.get("unit").and_then(J::str).unwrap_or("?").to_string();
    // the recorded digests are of keys that had no plane-shape fields
    let c = replay_ctx(true);
    unsafe { raptor_wings::declare::rw_declare_kinds(c) };
    let mut line = format!(
        "{name:<14} nodes {:>4}",
        inp.get("nodes").map_or(0, |n| n.arr().len())
    );
    let done = (|| {
        if let Err(e) = declare(c, &manifests(inp)) {
            *t.make_refused.entry(e.clone()).or_default() += 1;
            return format!("{line}  make refused");
        }
        let h = match make_nodes(c, inp, &mut Vals::default()) {
            Ok(h) => h,
            Err(e) => {
                *t.make_refused.entry(e).or_default() += 1;
                return format!("{line}  make refused");
            }
        };
        // two recorded call forms: the backend's render(lowered, options), and the common render_source(args, kwargs)
        let low = inp.get("lowered").and_then(|l| l.get("v"));
        let args = inp.get("args").and_then(|a| a.get("l")).map(|a| a.arr());
        if low.is_none() && args.is_none() {
            t.canon_refused += 1;
            *t.canon_errs
                .entry("no lowered record and no call arguments in the inputs".into())
                .or_default() += 1;
            return format!("{line}  no call record");
        }
        // a kernel the recorder could not encode is an object of the record: its attributes are the kernel's fields
        let kern_obj: Option<J> = kernel_object(low, inp);
        let kernel = kern_obj
            .as_ref()
            .and_then(|k| k.get("v"))
            .or(low.and_then(|l| l.get("kernel")).and_then(|k| k.get("v")));
        let (sinks_j, rw) = match args {
            Some(a) => (a.get(1), a.get(2).and_then(|w| w.get("v"))),
            None => (
                kernel.and_then(|k| k.get("sinks")),
                kernel.and_then(|k| k.get("walk")).and_then(|w| w.get("v")),
            ),
        };
        let mut w = u32::MAX;
        if let Some(rw) = rw {
            let sinks: Vec<u32> = elems(sinks_j)
                .iter()
                .map(|s| h[s.get("n").unwrap().int() as usize])
                .collect();
            // the options the walk was recorded with; a recorded walk without them is asked for with its own slots declared
            let rec_digest = rw
                .get("digest")
                .and_then(tagged_str)
                .unwrap_or("")
                .to_string();
            let mut cands = walk_options(rec, &rec_digest, &h);
            if cands.is_empty() {
                let leaves: Vec<String> = rw
                    .get("leaves")
                    .map_or(&[][..], |l| {
                        l.get("t").or(l.get("l")).map_or(&[][..], |x| x.arr())
                    })
                    .iter()
                    .map(|x| val(x, &h))
                    .collect();
                // the record lacks the options of this walk (no call for it was seen): its slots stand for the declaration, and the
                // exempt plane every recorded call with options names is the second try
                cands.push(WalkOpts {
                    declared: format!("[{}]", leaves.join(", ")),
                    force: None,
                    exempt: "[]".into(),
                });
                cands.push(WalkOpts {
                    declared: format!("[{}]", leaves.join(", ")),
                    force: None,
                    exempt: "['fused_steps']".into(),
                });
            }
            // the first set that reproduces the recorded digest; when none does, the first set that walks
            let (mut matched, mut last_err) = (0usize, String::new());
            for o in &cands {
                let (declared, exempt, force) = (
                    CString::new(o.declared.as_str()).unwrap(),
                    CString::new(o.exempt.as_str()).unwrap(),
                    o.force.as_ref().map(|f| CString::new(f.as_str()).unwrap()),
                );
                let mut w1 = u32::MAX;
                if rx_canonical(
                    c,
                    sinks.as_ptr(),
                    sinks.len(),
                    declared.as_ptr(),
                    force.as_ref().map_or(std::ptr::null(), |f| f.as_ptr()),
                    exempt.as_ptr(),
                    &mut w1,
                ) != 0
                {
                    last_err = err(c);
                    continue;
                }
                let mut dg = [0 as std::ffi::c_char; 65];
                rx_digest(c, w1, dg.as_mut_ptr());
                let same =
                    unsafe { CStr::from_ptr(dg.as_ptr()) }.to_string_lossy() == rec_digest.as_str();
                if same {
                    if matched == 0 {
                        w = w1
                    }
                    matched += 1
                } else if w == u32::MAX {
                    w = w1
                }
            }
            t.ambiguous += (matched > 1) as usize;
            if w == u32::MAX {
                t.canon_refused += 1;
                *t.canon_errs.entry(last_err).or_default() += 1;
                return format!("{line}  walk refused");
            }
            t.loaded += 1;
            let mut dg = [0 as std::ffi::c_char; 65];
            rx_digest(c, w, dg.as_mut_ptr());
            let dg = unsafe { CStr::from_ptr(dg.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            let want = |f: &str| rw.get(f);
            let recorded_dg = want("digest").and_then(tagged_str);
            let same_walk = recorded_dg == Some(dg.as_str());
            let dg_ok = same_walk;
            let spec_ok =
                want("arg_spec").is_some_and(|a| val(a, &h) == walk_field(c, w, "arg_spec"));
            // the door's access is the class text; the recorded Access also holds `op`
            let acc = want("access").and_then(|a| a.get("v"));
            let acc_ok = acc
                .and_then(|a| a.get("cls"))
                .or(want("access"))
                .is_some_and(|a| val(a, &h) == walk_field(c, w, "access"));
            t.access_op += acc
                .and_then(|a| a.get("op"))
                .is_some_and(|o| !matches!(o, J::Null)) as usize;
            (t.digest_eq, t.spec_eq, t.access_eq) = (
                t.digest_eq + dg_ok as usize,
                t.spec_eq + spec_ok as usize,
                t.access_eq + acc_ok as usize,
            );
            if !(dg_ok && spec_ok && acc_ok) && t.shown < show {
                t.shown += 1;
                println!("WALK {name}: access recorded {} computed {}; digest recorded {} computed {dg}{}", want("access").map_or(String::new(), |a| val(a, &h)), walk_field(c, w, "access"), want("digest").and_then(tagged_str).unwrap_or(""), if spec_ok { String::new() } else { format!("\n  arg_spec recorded {}\n  arg_spec computed {}", want("arg_spec").map_or(String::new(), |a| val(a, &h)), walk_field(c, w, "arg_spec")) });
            }
            line = format!(
                "{line}  loaded  digest {}  arg_spec {}  access {}",
                if dg_ok { "=" } else { "!=" },
                if spec_ok { "=" } else { "!=" },
                if acc_ok { "=" } else { "!=" }
            );
        } else {
            t.no_walk += 1;
            line = format!("{line}  no walk in the record")
        }
        if !render {
            return line;
        }
        let mut json = String::new();
        let keys: Vec<String> = match low {
            Some(low) => {
                json.push_str("{\"$type\":\"BuiltLowered\"");
                for (k, x) in low.obj() {
                    json.push_str(&format!(",{}:", js(k)));
                    contract(
                        if k == "kernel" {
                            kern_obj.as_ref().unwrap_or(x)
                        } else {
                            x
                        },
                        &h,
                        &mut json,
                        &mut t.types,
                    )
                }
                json.push_str(",\"options\":");
                contract(
                    inp.get("options").unwrap_or(&J::Null),
                    &h,
                    &mut json,
                    &mut t.types,
                );
                low.obj()
                    .iter()
                    .map(|(k, _)| k.clone())
                    .chain(["options".to_string()])
                    .collect()
            }
            None => {
                json.push_str("{\"$type\":\"RenderSource\",\"args\":");
                contract(inp.get("args").unwrap(), &h, &mut json, &mut t.types);
                json.push_str(",\"kwargs\":");
                contract(
                    inp.get("kwargs").unwrap_or(&J::Null),
                    &h,
                    &mut json,
                    &mut t.types,
                );
                vec!["args".into(), "kwargs".into()]
            }
        };
        facts_key(rec, &mut json);
        json.push('}');
        t.typed.note(&json, "host");
        *t.keysets.entry(keys.join(", ")).or_default() += 1;
        let src = rec
            .get("output")
            .and_then(|o| o.get("value"))
            .and_then(|v| v.get("v"))
            .unwrap();
        let rec_text = |k: &str| src.get(k).and_then(tagged_str).unwrap_or("").to_string();
        // the body: written by the renderer core, compared with the recorded one
        let key = rec.get("key").and_then(J::str).unwrap_or("").to_string();
        let listed = t.accepted.iter().any(|a| a.0 == key);
        if listed {
            t.accepted_seen.push(key.clone())
        }
        let team_stage = aether_writers::contract::Contract::parse(&json)
            .is_ok_and(|k| aether_writers::is_team_stage(&k));
        let body_state = match entry_body(c as *mut RxCtx, w, &json, "host").map_err(reason) {
            Ok(b) if b == rec_text("body") => {
                t.body_eq += 1;
                t.team_eq += team_stage as usize;
                "=".to_string()
            }
            Ok(_) if listed => "!= (accepted)".to_string(),
            Ok(b) => {
                t.body_diff += 1;
                if t.shown < show {
                    t.shown += 1;
                    println!("BODY {name}: {}", first_diff(&rec_text("body"), &b))
                }
                "!=".to_string()
            }
            Err(e) => {
                let f = e
                    .strip_prefix("not in this build: ")
                    .unwrap_or(&e)
                    .to_string();
                *t.features.entry(f.clone()).or_default() += 1;
                t.names.entry(f.clone()).or_default().push(label(rec));
                format!("nib ({f})")
            }
        };
        t.team_n += team_stage as usize;
        line = format!("{line}  body {body_state}");
        t.typed.needs(c as *mut RxCtx, w, &json, "host", &name);
        t.typed.needs(c as *mut RxCtx, w, &json, "cuda", &name);
        match host_render(c as *mut RxCtx, w, &json).map_err(reason) {
            Ok(s) => {
                #[cfg(feature = "shadow")]
                if let Some(diffs) = &s.needs_diffs {
                    t.typed.needs_bodies += 1;
                    for d in diffs {
                        t.typed.needs_diffs.push(format!("{name} (host): {d}"))
                    }
                }
                dump_unit(&t.dump, "host", &key, Some((&s.text, &s.body)), None);
                let (bo, to) = (s.body == rec_text("body"), s.text == rec_text("text"));
                t.text_eq += to as usize;
                if !to && listed {
                    t.accepted_diff += 1;
                    t.accepted_hit.push(key.clone())
                } else {
                    t.text_diff += (!to) as usize
                }
                if !(bo && to) && (t.shown < show || (listed && !to)) {
                    t.shown += 1;
                    let (b_d, t_d) = (
                        if bo {
                            "equal".into()
                        } else {
                            first_diff(&rec_text("body"), &s.body)
                        },
                        if to {
                            "equal".into()
                        } else {
                            first_diff(&rec_text("text"), &s.text)
                        },
                    );
                    println!(
                        "MISMATCH {name} [{key}]{} {}: body {b_d}; text {t_d}",
                        if listed { " (accepted)" } else { "" },
                        label(rec)
                    );
                }
                format!("{line}  text {}", if to { "=" } else { "!=" })
            }
            Err(e) if e.starts_with("not in this build") => {
                dump_unit(&t.dump, "host", &key, None, Some(&e));
                t.not_built += 1;
                t.names.entry(e.clone()).or_default().push(label(rec));
                format!("{line}  {e}")
            }
            Err(e) => {
                dump_unit(&t.dump, "host", &key, None, Some(&e));
                t.render_err += 1;
                *t.render_errs.entry(e.clone()).or_default() += 1;
                format!("{line}  render error: {e}")
            }
        }
    })();
    rx_ctx_free(c);
    done
}

/// What bringing the read records to the current IR form did: the nodes changed, the records they were in, and the records the
/// migration could not express exactly (these are not replayed), by reason.
#[derive(Default)]
struct Migrations {
    /// Nodes changed, by rule.
    nodes: Changed,
    /// Records that held at least one old-form node.
    records: usize,
    /// Records left out, with the reason.
    refused: HashMap<String, usize>,
}

/// What reading one recorded line gave.
enum Read {
    /// The record, in the current IR form.
    Record(J),
    /// The json reader refused the line.
    Unparsed,
    /// The migration could not express the record exactly (counted by reason in the migrations).
    NotExpressible,
}

impl Migrations {
    /// Reads one recorded line into a record in the current form.
    fn read(&mut self, line: &str) -> Read {
        let Ok(mut rec) = json::parse_without(line, &["analyses"]) else {
            return Read::Unparsed;
        };
        match migrate::migrate(&mut rec) {
            Ok(changed) => {
                self.records += usize::from(!changed.is_none());
                self.nodes.add(changed);
                Read::Record(rec)
            }
            Err(reason) => {
                *self.refused.entry(reason).or_default() += 1;
                Read::NotExpressible
            }
        }
    }

    /// The report lines of the migration.
    fn print(&self) {
        println!(
            "migrated to the current IR form: {} records ({} extents, {} reads, {} stores); not expressible {}",
            self.records,
            self.nodes.extents,
            self.nodes.reads,
            self.nodes.stores,
            self.refused.values().sum::<usize>()
        );
        for (reason, n) in &self.refused {
            println!("not expressible: {reason}: {n}");
        }
    }
}

/// Brings the records of the small in-tree data file at `path` to the current IR form, in place, and says how many changed.
fn migrate_in_place(path: &str) {
    match migrate::migrate_file(std::path::Path::new(path)) {
        Ok(report) => {
            println!("records {}, migrated {}", report.records, report.changed);
            for (line, reason) in &report.refused {
                println!("not expressible: line {line}: {reason}");
            }
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).expect("usage: treplay <texts.jsonl> [--target host] [--limit n] [--only s] [--show k] [--ladder] [--text-from-recorded-body]");
    if a.iter().any(|x| x == "--migrate") {
        return migrate_in_place(path);
    }
    if a.iter().any(|x| x == "--text-from-recorded-body") {
        return text_from_recorded_body(path, &a);
    }
    let arg = |f: &str| a.iter().position(|x| x == f).and_then(|i| a.get(i + 1));
    let target = arg("--target").map_or("host", |s| s.as_str());
    let limit: usize = arg("--limit")
        .and_then(|x| x.parse().ok())
        .unwrap_or(usize::MAX);
    let show: usize = arg("--show").and_then(|x| x.parse().ok()).unwrap_or(0);
    let lines: usize = arg("--lines").and_then(|x| x.parse().ok()).unwrap_or(15);
    let (only, ladder) = (arg("--only"), a.iter().any(|x| x == "--ladder"));
    let mut t = Tally::default();
    let mut migrations = Migrations::default();
    if let Some(f) = arg("--accepted") {
        t.accepted = read_accepted(f)
    }
    t.dump = arg("--dump").cloned();
    let mut laddered: Vec<&str> = vec![];
    let start = Instant::now();
    let needle = format!("\"target\":\"{target}\"");
    for l in std::io::BufReader::new(std::fs::File::open(path).expect("open")).lines() {
        if t.ran >= limit {
            break;
        }
        let l = l.unwrap();
        if !l.contains(&needle) {
            continue;
        }
        let rec = match migrations.read(&l) {
            Read::Record(rec) => rec,
            Read::Unparsed => {
                t.unparsed += 1;
                continue;
            }
            Read::NotExpressible => continue,
        };
        if rec.get("target").and_then(J::str) != Some(target) {
            continue;
        }
        let name = rec.get("unit").and_then(J::str).unwrap_or("");
        let tests: Vec<&str> = rec
            .get("tests")
            .map_or(&[][..], |x| x.arr())
            .iter()
            .filter_map(J::str)
            .collect();
        if let Some(o) = only {
            if !name.contains(o.as_str()) && !tests.iter().any(|x| x.contains(o.as_str())) {
                continue;
            }
        }
        if ladder {
            match LADDER.iter().find(|n| **n == name) {
                Some(n)
                    if !laddered.contains(n)
                        && matches!(rec.get("error"), None | Some(J::Null)) =>
                {
                    laddered.push(*n)
                }
                _ => continue,
            }
        }
        t.units += 1;
        if let Some(why) = excluded_reason(&rec) {
            t.excluded += 1;
            t.excluded_by.entry(why).or_default().push(label(&rec));
            continue;
        }
        if !matches!(rec.get("error"), None | Some(J::Null)) {
            t.raised += 1;
            continue;
        }
        t.ran += 1;
        let line = unit(&rec, target == "host", show, &mut t);
        if ladder {
            println!("{line}")
        }
    }
    let u = t.ran;
    println!("lines the json reader refused {}", t.unparsed);
    migrations.print();
    println!(
        "units {} (excluded {}, oracle-raised {})",
        t.units, t.excluded, t.raised
    );
    for (why, units) in &t.excluded_by {
        println!("excluded: {why}: {}", units.join(", "))
    }
    println!("walk-loaded {} / {u}", t.loaded);
    println!(
        "walk-digest-equal {} / {u}   (arg_spec {} / {u}, access {} / {u})",
        t.digest_eq, t.spec_eq, t.access_eq
    );
    println!("access with op set {}", t.access_op);
    println!(
        "units whose recorded digest more than one option set reproduces {}",
        t.ambiguous
    );
    println!("make-refused {}", t.make_refused.values().sum::<usize>());
    top(&t.make_refused, 5);
    println!("walk none (no kernel in the record) {}", t.no_walk);
    println!("walk-refused {}", t.canon_refused);
    top(&t.canon_errs, 5);
    if target == "host" {
        println!("body identical {} / {u}", t.body_eq);
        println!("team stage body identical {} / {}", t.team_eq, t.team_n);
        println!(
            "text identical (whole path) {} / {u}   (written but different {})",
            t.text_eq, t.text_diff
        );
        println!(
            "accepted, text differs (value-proven list) {}",
            t.accepted_diff
        );
        for a in &t.accepted {
            if t.accepted_seen.contains(&a.0) && !t.accepted_hit.contains(&a.0) {
                println!(
                    "stale accepted entry (identical or refused): {} {}",
                    a.0, a.1
                )
            }
        }
        println!("body written but different {}", t.body_diff);
        println!("body by feature (first construct not written yet):");
        top(&t.features, lines);
        print_names(&t.names, arg("--names").as_ref().copied());
        println!("not in this build {}", t.not_built);
        println!("render errors {}", t.render_err);
        top(&t.render_errs, 5);
        for (k, n) in &t.keysets {
            println!("contract keys ({n} units): {k}")
        }
        let mut ty: Vec<_> = t.types.iter().collect();
        ty.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        println!(
            "contract $types: {}",
            ty.iter()
                .map(|(k, n)| format!("{k} {n}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if target == "host" {
        t.typed.print()
    }
    let s = start.elapsed().as_secs_f64();
    println!(
        "timing: total {s:.2} s, {:.2} ms per unit",
        s * 1e3 / u.max(1) as f64
    );
    if (target == "host" || target == "cuda") && !ladder {
        text_from_recorded_body(path, &a)
    }
}

// ---- --text-from-recorded-body: the host arm's text from the recorded body, the analyses the real wings library answers ----

/// Replays the recorded bodies as texts from the file at `path`.
fn text_from_recorded_body(path: &str, a: &[String]) {
    let arg = |f: &str| a.iter().position(|x| x == f).and_then(|i| a.get(i + 1));
    let dump = arg("--dump").cloned();
    let target = arg("--target").map_or("host", |s| s.as_str());
    let needle = format!("\"target\":\"{target}\"");
    let limit: usize = arg("--limit")
        .and_then(|x| x.parse().ok())
        .unwrap_or(usize::MAX);
    let show: usize = arg("--show").and_then(|x| x.parse().ok()).unwrap_or(0);
    let lines: usize = arg("--lines").and_then(|x| x.parse().ok()).unwrap_or(15);
    let (only, ladder) = (arg("--only"), a.iter().any(|x| x == "--ladder"));
    let (mut seen, mut laddered): (Vec<String>, Vec<&str>) = (vec![], vec![]);
    let (mut units, mut identical, mut shown, mut unparsed) = (0usize, 0usize, 0usize, 0usize);
    let mut feats: HashMap<String, usize> = HashMap::new();
    let mut typed = Typed::default();
    let mut migrations = Migrations::default();
    let mut names: HashMap<String, Vec<String>> = HashMap::new();
    let start = Instant::now();
    for l in std::io::BufReader::new(std::fs::File::open(path).expect("open")).lines() {
        let l = l.unwrap();
        if units >= limit {
            break;
        }
        if !l.contains(&needle) {
            continue;
        }
        let rec = match migrations.read(&l) {
            Read::Record(rec) => rec,
            Read::Unparsed => {
                unparsed += 1;
                continue;
            }
            Read::NotExpressible => continue,
        };
        if rec.get("target").and_then(J::str) != Some(target)
            || !matches!(rec.get("error"), None | Some(J::Null))
            || excluded_reason(&rec).is_some()
        {
            continue;
        }
        let Some(src) = rec
            .get("output")
            .and_then(|o| o.get("value"))
            .and_then(|v| v.get("v"))
        else {
            continue;
        };
        let name = rec.get("unit").and_then(J::str).unwrap_or("").to_string();
        if only.is_some_and(|o| {
            !name.contains(o.as_str())
                && !rec
                    .get("tests")
                    .map_or(&[][..], |x| x.arr())
                    .iter()
                    .filter_map(J::str)
                    .any(|x| x.contains(o.as_str()))
        }) {
            continue;
        }
        let rt = |k: &str| src.get(k).and_then(tagged_str).unwrap_or("").to_string();
        let ident = format!(
            "{name}|{}|{}",
            rec.get("texts").map_or(String::new(), |t| t
                .arr()
                .iter()
                .filter(|x| matches!(x.arr().first().and_then(J::str), Some("text" | "body")))
                .map(|x| x.arr().get(1).and_then(J::str).unwrap_or("").to_string())
                .collect::<Vec<_>>()
                .join(",")),
            rec.get("key").and_then(J::str).unwrap_or("")
        );
        if seen.contains(&ident) {
            continue;
        }
        seen.push(ident);
        if ladder {
            match LADDER.iter().find(|n| **n == name) {
                Some(n) if !laddered.contains(n) => laddered.push(*n),
                _ => continue,
            }
        }
        units += 1;
        let inp_all = with_analysis_nodes(rec.get("inputs").unwrap(), &rec);
        let inp = &inp_all;
        // a literal-table record: the input is a finished source text, the output the text with its literals read from a table
        let source_text = |j: Option<&J>| {
            j.and_then(|v| v.get("v"))
                .and_then(|v| v.get("text"))
                .and_then(tagged_str)
                .map(String::from)
        };
        if let Some(input) = inp
            .get("args")
            .and_then(|a| a.get("l"))
            .and_then(|l| l.arr().first())
            .or(inp.get("source"))
            .and_then(|x| source_text(Some(x)))
        {
            let want = source_text(rec.get("output").and_then(|o| o.get("value")));
            // the table is read only in float64 mode; other modes keep the text
            let float64 = inp
                .get("args")
                .and_then(|a| a.get("l"))
                .and_then(|l| l.arr().first())
                .or(inp.get("source"))
                .and_then(|x| x.get("v"))
                .and_then(|v| v.get("mode"))
                .and_then(tagged_str)
                == Some("float64");
            let made = if float64 {
                aether_writers::cuda::literals::literal_table(&input)
            } else {
                input.clone()
            };
            if want.as_deref() == Some(made.as_str()) {
                identical += 1
            } else {
                *feats.entry("differs".to_string()).or_default() += 1;
                if shown < show {
                    shown += 1;
                    println!("MISMATCH {name}: literal table")
                }
            }
            continue;
        }
        let c = rx_ctx_new();
        unsafe { raptor_wings::declare::rw_declare_kinds(c) };
        let verdict: Result<(), String> = (|| {
            declare(c, &manifests(inp))?;
            let h = make_nodes(c, inp, &mut Vals::default())?;
            let low = inp.get("lowered").and_then(|l| l.get("v"));
            // a kernel the recorder could not encode is an object of the record: its attributes are the kernel's fields
            let kern_obj: Option<J> = kernel_object(low, inp);
            let args = inp.get("args").and_then(|x| x.get("l")).map(|x| x.arr());
            let kernel = kern_obj
                .as_ref()
                .and_then(|k| k.get("v"))
                .or(low.and_then(|l| l.get("kernel")).and_then(|k| k.get("v")));
            let (sinks_j, rw) = match args {
                Some(x) => (x.get(1), x.get(2).and_then(|w| w.get("v"))),
                None => (
                    kernel.and_then(|k| k.get("sinks")),
                    kernel.and_then(|k| k.get("walk")).and_then(|w| w.get("v")),
                ),
            };
            let mut w = u32::MAX;
            if let Some(rw) = rw {
                let sinks: Vec<u32> = elems(sinks_j)
                    .iter()
                    .map(|s| h[s.get("n").unwrap().int() as usize])
                    .collect();
                let leaves: Vec<String> = rw
                    .get("leaves")
                    .map_or(&[][..], |l| {
                        l.get("t").or(l.get("l")).map_or(&[][..], |x| x.arr())
                    })
                    .iter()
                    .map(|x| val(x, &h))
                    .collect();
                let (declared, exempt) = (
                    CString::new(format!("[{}]", leaves.join(", "))).unwrap(),
                    CString::new("[]").unwrap(),
                );
                if rx_canonical(
                    c,
                    sinks.as_ptr(),
                    sinks.len(),
                    declared.as_ptr(),
                    std::ptr::null(),
                    exempt.as_ptr(),
                    &mut w,
                ) != 0
                {
                    return Err(format!("walk refused: {}", err(c)));
                }
            }
            let mut json = String::new();
            match low {
                Some(low) => {
                    json.push_str("{\"$type\":\"BuiltLowered\"");
                    for (k, x) in low.obj() {
                        json.push_str(&format!(",{}:", js(k)));
                        contract(
                            if k == "kernel" {
                                kern_obj.as_ref().unwrap_or(x)
                            } else {
                                x
                            },
                            &h,
                            &mut json,
                            &mut HashMap::new(),
                        )
                    }
                    json.push_str(",\"options\":");
                    contract(
                        inp.get("options").unwrap_or(&J::Null),
                        &h,
                        &mut json,
                        &mut HashMap::new(),
                    );
                }
                None => {
                    json.push_str("{\"$type\":\"RenderSource\",\"args\":");
                    contract(
                        inp.get("args").ok_or("no call record")?,
                        &h,
                        &mut json,
                        &mut HashMap::new(),
                    );
                    json.push_str(",\"kwargs\":");
                    contract(
                        inp.get("kwargs").unwrap_or(&J::Null),
                        &h,
                        &mut json,
                        &mut HashMap::new(),
                    );
                }
            }
            let body = rt("body");
            facts_key(&rec, &mut json);
            json.push_str(&format!(
                ",\"$body\":{},\"$needs_random\":{}}}",
                js(&body),
                body.contains("aether::random::")
            ));
            typed.note(&json, target);
            let s = if target == "cuda" {
                cuda_render(c as *mut RxCtx, w, &json).map_err(reason)?
            } else {
                host_render(c as *mut RxCtx, w, &json).map_err(reason)?
            };
            let unit_key = rec.get("key").and_then(J::str).unwrap_or("");
            dump_unit(&dump, target, unit_key, Some((&s.text, &s.body)), None);
            let meta_text = s.meta.json();
            let meta_ok = s.entry == rt("entry")
                && meta_text.contains(&format!("\"mode\":\"{}\"", rt("mode")))
                && meta_text.contains(&format!(
                    "\"vectorised\":{}",
                    matches!(src.get("vectorised"), Some(J::Bool(true)))
                ));
            if s.text != rt("text") {
                return Err(format!("differs: {}", first_diff(&rt("text"), &s.text)));
            }
            if !meta_ok {
                return Err(format!(
                    "differs: entry, mode or vectorised (computed {}; recorded {:?}, {:?}, {:?})",
                    meta_text,
                    rt("entry"),
                    rt("mode"),
                    src.get("vectorised")
                ));
            }
            Ok(())
        })();
        rx_ctx_free(c);
        if let Err(e) = &verdict {
            if !e.starts_with("differs") {
                let unit_key = rec.get("key").and_then(J::str).unwrap_or("");
                dump_unit(&dump, target, unit_key, None, Some(e));
            }
        }
        match verdict {
            Ok(()) => {
                identical += 1;
                if ladder {
                    println!("{name:<14} text identical")
                }
            }
            Err(e) => {
                let f = e
                    .strip_prefix("not in this build: ")
                    .map(|x| format!("not in this build: {x}"))
                    .unwrap_or_else(|| {
                        if e.starts_with("differs") {
                            "differs".into()
                        } else {
                            format!("error: {}", e.chars().take(80).collect::<String>())
                        }
                    });
                if ladder {
                    println!("{name:<14} {f}")
                }
                if e.starts_with("differs") && shown < show {
                    shown += 1;
                    println!("MISMATCH {name}: {e}")
                }
                if f.starts_with("error:") && shown < show {
                    shown += 1;
                    println!("UNREADABLE {name}: {e}")
                }
                names.entry(f.clone()).or_default().push(label(&rec));
                *feats.entry(f).or_default() += 1;
            }
        }
    }
    println!("text identical (from the recorded body) {identical} / {units}");
    if target == "cuda" {
        typed.print()
    }
    println!("text by feature (lines the json reader refused: {unparsed})");
    migrations.print();
    top(&feats, lines);
    print_names(&names, arg("--names").as_ref().copied());
    println!("timing: total {:.2} s", start.elapsed().as_secs_f64());
}
