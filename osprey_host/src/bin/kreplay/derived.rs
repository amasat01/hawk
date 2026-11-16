//! The derived sets of a recording: a call whose kernels hold a primal and its taping twin and reverse unit is replayed as ONE request
//! with the list of derived kernels. The stated facts are converted from the reserved names the old build used (`tape`,
//! `applied_steps`, `tape_on`, `<time>_0`, `bar_*`, `*_next`): a guess of this proof tool, not a rule of the library.
use super::*;
use meta::{DerivedRole, LinkForm, LinkWhat, SlotRole, WireDtype};
use osprey_host::lower::derived::{DerivedFacts, DerivedKernel, TapedPlane};
use osprey_host::lower::links::LinkFact;

/// What one set of a call came to.
enum Built {
    /// The set was answered: (texts equal modulo the block, texts different, blocks omitted).
    Answered(usize, usize, usize),
    /// The library or the converter refused it, with the reason.
    Refused(String),
}

/// The slot of a name in a unit.
fn slot<'a>(unit: &'a meta::UnitMeta, name: &str) -> Option<(usize, &'a meta::Slot)> {
    unit.body
        .slots
        .iter()
        .enumerate()
        .find(|(_, s)| s.name == name)
}

/// The text with its self-check block cut out and its runs of blank lines folded.
fn without_block(text: &str) -> String {
    let cut = text.find("\n// The self-check exports").map(|from| {
        let tail = &text[from..];
        let end = tail.find("};\n").map_or(tail.len(), |e| e + 3);
        format!("{}{}", &text[..from], &tail[end..])
    });
    let mut out = String::new();
    for line in cut.as_deref().unwrap_or(text).lines() {
        if !(line.is_empty() && out.ends_with("\n\n")) {
            out.push_str(line);
            out.push('\n')
        }
    }
    out
}

/// The facts of a taping twin, from the names of its slots and the primal's.
fn twin_facts(
    primal: &meta::UnitMeta,
    own: &meta::UnitMeta,
    time: Option<&str>,
) -> Result<DerivedFacts, String> {
    let mut facts = DerivedFacts::default();
    for (plane, rows) in taped_of(primal) {
        facts.tape_rows += rows;
        facts.taped.push(TapedPlane { plane, rows });
    }
    let has = |n: &str| slot(own, n).map(|_| n.to_string());
    facts.tape = has("tape");
    facts.steps = has("applied_steps");
    facts.tape_on = has("tape_on");
    facts.fused_steps = has("fused_steps");
    let time0 = time.map(|t| format!("{t}_0")).and_then(|n| has(&n));
    facts.time0 = time0.clone();
    let runtime: Vec<usize> = own
        .finish
        .iter()
        .map(|f| f.counter_slot)
        .chain(own.launch.active_map_slot)
        .chain(own.launch.active_count_slot)
        .collect();
    for (at, s) in own.body.slots.iter().enumerate() {
        if runtime.contains(&at) {
            continue;
        }
        let reserved = [
            &facts.tape,
            &facts.steps,
            &facts.tape_on,
            &facts.fused_steps,
        ]
        .iter()
        .any(|r| r.as_deref() == Some(s.name.as_str()));
        if reserved {
            continue;
        }
        let link = if Some(&s.name) == time0.as_ref() {
            (LinkWhat::Time0, time.map(String::from))
        } else if slot(primal, &s.name).is_some() {
            (LinkWhat::PrimalSlot, Some(s.name.clone()))
        } else {
            return Err(format!("the twin's slot {} has no primal slot", s.name));
        };
        facts.links.push(LinkFact {
            slot: s.name.clone(),
            what: link.0,
            form: LinkForm::Value,
            of: link.1,
            reduced: false,
        });
    }
    Ok(facts)
}

/// The planes a twin tapes: the real mutable planes the primal reads before it stores, with their rows.
fn taped_of(primal: &meta::UnitMeta) -> Vec<(String, i64)> {
    primal
        .body
        .slots
        .iter()
        .filter(|s| {
            s.role == SlotRole::Mutable
                && s.flags & meta::SLOT_PRIOR != 0
                && s.dtype == WireDtype::Real
        })
        .map(|s| (s.name.clone(), s.shape.iter().product::<i64>().max(1)))
        .collect()
}

/// The facts of a reverse unit, from the names of its slots and the primal's.
fn vjp_facts(
    primal: &meta::UnitMeta,
    own: &meta::UnitMeta,
    twin: Option<&DerivedFacts>,
    time: Option<&str>,
) -> Result<DerivedFacts, String> {
    let mut facts = DerivedFacts::default();
    let taped: Vec<String> = taped_of(primal).into_iter().map(|t| t.0).collect();
    for s in &own.body.slots {
        let n = s.name.as_str();
        let state = |x: &str| slot(primal, x).filter(|p| p.1.role == SlotRole::Mutable);
        let (what, form, of) =
            if let Some(x) = n.strip_prefix("bar_").and_then(|r| r.strip_suffix("_next")) {
                (
                    LinkWhat::PrimalSlot,
                    LinkForm::AdjointIn,
                    Some(x.to_string()),
                )
            } else if time.is_some_and(|t| n == format!("bar_{t}_0")) {
                (LinkWhat::Time0, LinkForm::Grad, time.map(String::from))
            } else if let Some(x) = n.strip_prefix("bar_") {
                if state(x).is_some() {
                    (
                        LinkWhat::PrimalSlot,
                        LinkForm::AdjointOut,
                        Some(x.to_string()),
                    )
                } else if slot(primal, x).is_some() {
                    (LinkWhat::PrimalSlot, LinkForm::Grad, Some(x.to_string()))
                } else {
                    return Err(format!("the reverse unit's slot {n} has no primal"));
                }
            } else if n == "applied_steps" {
                (
                    LinkWhat::Steps,
                    LinkForm::Value,
                    twin.and_then(|t| t.steps.clone()),
                )
            } else if taped.iter().any(|t| t == n) {
                (LinkWhat::TapeRow, LinkForm::Value, Some(n.to_string()))
            } else if slot(primal, n).is_some() {
                (LinkWhat::PrimalSlot, LinkForm::Value, Some(n.to_string()))
            } else {
                return Err(format!("the reverse unit's slot {n} has no primal"));
            };
        facts.links.push(LinkFact {
            slot: s.name.clone(),
            what,
            form,
            of,
            reduced: form == LinkForm::Grad && what == LinkWhat::PrimalSlot,
        });
    }
    Ok(facts)
}

/// The names a refine unit's slots are made from: the primal's time plane and time step, and the taping twin's applied-steps slot.
pub struct RefineNames<'a> {
    /// The primal's time plane.
    pub time: &'a str,
    /// The primal's time step.
    pub step: &'a str,
    /// The twin's applied-steps slot, when the set has a twin.
    pub twin_steps: Option<&'a str>,
}

/// The unit-name suffixes of the three refine units.
pub const REFINE_SUFFIXES: [(&str, DerivedRole); 3] = [
    ("_newton", DerivedRole::RefineNewton),
    ("_state_guard_vjp", DerivedRole::StateGuardVjp),
    ("_refine_vjp", DerivedRole::RefineVjp),
];

/// The event index of a guard plane name (`event_g<i>`).
fn event_of(name: &str) -> Option<usize> {
    name.strip_prefix("event_g")?.parse().ok()
}

/// A link as (what, form, of) with the gradient flag the old build summed over samples.
type Meaning = (LinkWhat, LinkForm, Option<String>);

/// What a slot of a refine unit refers to, from its name by the reserved names (`event_g<i>`, `dot_event_g<i>`, `bar_event_g<i>`,
/// `<state>_refined` and its `dot_` and `bar_` forms, `<step>_tau` and its forms, `<time>_0`, `bar_<time>_0`, `applied_steps`) and
/// the primal's slots. A guess of this proof tool, as the other converters are.
fn refine_meaning(
    role: DerivedRole,
    primal: &meta::UnitMeta,
    taped: &[String],
    names: &RefineNames,
    n: &str,
) -> Option<Meaning> {
    let of = |s: &str| Some(s.to_string());
    let state = |x: &str| slot(primal, x).is_some_and(|p| p.1.role == SlotRole::Mutable);
    let known = |x: &str| slot(primal, x).is_some();
    let reverse = role != DerivedRole::RefineNewton;
    let step = names.step;
    if let Some(rest) = n.strip_prefix("bar_").filter(|_| reverse) {
        if let Some(i) = event_of(rest) {
            return Some((LinkWhat::Guard, LinkForm::AdjointIn, of(&i.to_string())));
        }
        if rest == format!("{step}_tau") {
            return Some((LinkWhat::Tau, LinkForm::Discard, None));
        }
        if rest == format!("{}_0", names.time) {
            return Some((LinkWhat::Time0, LinkForm::Grad, of(names.time)));
        }
        if let Some(x) = rest.strip_suffix("_refined").filter(|x| state(x)) {
            return Some((LinkWhat::Refined, LinkForm::AdjointIn, of(x)));
        }
        if rest == step && role == DerivedRole::RefineVjp {
            return Some((LinkWhat::TimeStep, LinkForm::Grad, of(step)));
        }
        if state(rest) {
            let what = if role == DerivedRole::StateGuardVjp {
                LinkWhat::Refined
            } else {
                LinkWhat::PrimalSlot
            };
            return Some((what, LinkForm::AdjointOut, of(rest)));
        }
        return known(rest).then(|| (LinkWhat::PrimalSlot, LinkForm::Grad, of(rest)));
    }
    if role == DerivedRole::RefineNewton {
        if let Some(i) = n.strip_prefix("dot_").and_then(event_of) {
            return Some((LinkWhat::Guard, LinkForm::Dot, of(&i.to_string())));
        }
        if let Some(i) = event_of(n) {
            return Some((LinkWhat::Guard, LinkForm::Value, of(&i.to_string())));
        }
        if let Some(x) = n
            .strip_prefix("dot_")
            .and_then(|r| r.strip_suffix("_refined"))
        {
            return Some((LinkWhat::Refined, LinkForm::Dot, of(x)));
        }
        if n == format!("dot_{step}_tau") {
            return Some((LinkWhat::Tau, LinkForm::Seed, None));
        }
    }
    if let Some(x) = n.strip_suffix("_refined") {
        return Some((LinkWhat::Refined, LinkForm::Value, of(x)));
    }
    if n == format!("{step}_tau") {
        return Some((LinkWhat::Tau, LinkForm::Value, None));
    }
    if n == "applied_steps" {
        return Some((
            LinkWhat::Steps,
            LinkForm::Value,
            names.twin_steps.map(String::from),
        ));
    }
    if n == step {
        return Some((LinkWhat::TimeStep, LinkForm::Value, of(step)));
    }
    if n == format!("{}_0", names.time) {
        return Some((LinkWhat::Time0, LinkForm::Value, of(names.time)));
    }
    if role == DerivedRole::StateGuardVjp && state(n) {
        return Some((LinkWhat::Refined, LinkForm::Value, of(n)));
    }
    if taped.iter().any(|t| t == n) && role != DerivedRole::StateGuardVjp {
        return Some((LinkWhat::TapeRow, LinkForm::Value, of(n)));
    }
    known(n).then(|| (LinkWhat::PrimalSlot, LinkForm::Value, of(n)))
}

/// The facts of a refine unit (the Newton unit, the state-guard reverse or the refine reverse), from the names of its slots.
pub fn refine_facts(
    role: DerivedRole,
    primal: &meta::UnitMeta,
    own: &meta::UnitMeta,
    names: &RefineNames,
) -> Result<DerivedFacts, String> {
    let taped: Vec<String> = taped_of(primal).into_iter().map(|t| t.0).collect();
    let mut facts = DerivedFacts::default();
    for s in &own.body.slots {
        let Some((what, form, of)) = refine_meaning(role, primal, &taped, names, &s.name) else {
            return Err(format!("the refine unit's slot {} has no primal", s.name));
        };
        let gradient =
            form == LinkForm::Grad && matches!(what, LinkWhat::PrimalSlot | LinkWhat::TimeStep);
        facts.links.push(LinkFact {
            slot: s.name.clone(),
            what,
            form,
            of,
            reduced: gradient,
        });
    }
    Ok(facts)
}

/// Asks the library for one kernel alone.
fn alone(
    call: &J,
    kernel: &J,
    facts: &[(Target, String)],
    ctx: &(*mut raptor_ir::record::Ctx, Vec<u32>),
) -> Result<(KernelRequest, KernelAnswer), String> {
    let request = request_for(call, kernel, facts.to_vec(), &ctx.1)?;
    let answer = ask(ctx.0, &request.to_json())?;
    Ok((request, answer))
}

/// One set of a call: the primal kernel and the twin and reverse kernels that name it.
fn one_set(
    call: &J,
    kernels: &[&J],
    primal: &J,
    id: &str,
    dir: Option<&Path>,
    machines: &mut HashMap<String, String>,
    ctx: &mut Option<(*mut raptor_ir::record::Ctx, Vec<u32>)>,
) -> Built {
    let name = primal.get("name").and_then(J::str).unwrap_or("");
    let found = Found {
        call,
        id,
        show: 0,
        target: Target::Host,
    };
    let mut go = || -> Result<Built, String> {
        let mut facts = facts_all(&found, primal, machines)?;
        facts.retain(|(t, _)| *t == Target::Host);
        if ctx.is_none() {
            *ctx = Some(load(call)?);
        }
        let held = ctx.as_ref().ok_or("no context")?;
        let (base, own) = alone(call, primal, &facts, held)?;
        let time = primal
            .get("time")
            .and_then(J::str)
            .or_else(|| text(primal.get("time")));
        let mut list = vec![];
        let mut singles = vec![own.units[0].texts[0].text.clone()];
        let mut twin: Option<DerivedFacts> = None;
        for kernel in kernels {
            let kname = kernel.get("name").and_then(J::str).unwrap_or("");
            let (request, answer) = alone(call, kernel, &facts, held)?;
            let (role, facts_of) = if kname == format!("{name}_tape") {
                let f = twin_facts(&own.units[0].meta, &answer.units[0].meta, time)?;
                twin = Some(f.clone());
                let role = if f.fused_steps.is_some() {
                    DerivedRole::TapeFused
                } else {
                    DerivedRole::Tape
                };
                (role, f)
            } else if let Some(&(_, role)) = REFINE_SUFFIXES
                .iter()
                .find(|(suffix, _)| kname == format!("{name}{suffix}"))
            {
                let step = text(primal.get("time_step"))
                    .ok_or("a refine unit of a kernel without a time step")?;
                let names = RefineNames {
                    time: time.ok_or("a refine unit of a kernel without a time plane")?,
                    step,
                    twin_steps: twin.as_ref().and_then(|t| t.steps.as_deref()),
                };
                let meta = &answer.units[0].meta;
                (role, refine_facts(role, &own.units[0].meta, meta, &names)?)
            } else {
                (
                    DerivedRole::StepVjp,
                    vjp_facts(
                        &own.units[0].meta,
                        &answer.units[0].meta,
                        twin.as_ref(),
                        time,
                    )?,
                )
            };
            singles.push(answer.units[0].texts[0].text.clone());
            list.push(DerivedKernel {
                request,
                role,
                facts: facts_of,
            });
        }
        let mut set = base;
        set.derived = list;
        let answer = ask(held.0, &set.to_json())?;
        if let Some(dir) = dir {
            std::fs::write(dir.join(format!("{id}_{name}_set.c")), &answer.meta_source)
                .map_err(|e| e.to_string())?;
            for unit in &answer.units {
                std::fs::write(
                    dir.join(format!("{id}_{}_set.cpp", unit.meta.name)),
                    &unit.texts[0].text,
                )
                .map_err(|e| e.to_string())?;
            }
        }
        let (mut equal, mut different, mut omitted) = (0, 0, 0);
        for (unit, single) in answer.units.iter().zip(&singles) {
            let set_text = &unit.texts[0].text;
            if single.contains("eagle_abi_tag") && !set_text.contains("eagle_abi_tag") {
                omitted += 1;
            }
            if without_block(set_text) == without_block(single) {
                equal += 1
            } else {
                different += 1
            }
        }
        Ok(Built::Answered(equal, different, omitted))
    };
    go().unwrap_or_else(|why| Built::Refused(format!("{id} {name}: {why}")))
}

/// Whether a refusal says the twin's slots do not match the primal's: the twin was recorded without its own primal (another kernel
/// of the same name stands in the call), which is no set to build.
fn is_lone_twin(why: &str) -> bool {
    why.contains("the twin's slot") && why.contains("has no primal slot")
}

/// The counts of a replay of derived sets: found, built, refused, texts equal modulo the block, texts different, blocks omitted,
/// twins recorded without their primal, and the first refusal.
#[derive(Default)]
pub struct Totals {
    /// The seven counts.
    pub counts: [usize; 7],
    /// The first refusal.
    pub first: Option<String>,
}

impl Totals {
    /// Adds the counts of another replay.
    pub fn add(&mut self, other: Totals) {
        for (a, b) in self.counts.iter_mut().zip(other.counts) {
            *a += b;
        }
        if self.first.is_none() {
            self.first = other.first;
        }
    }

    /// The two count lines.
    pub fn print(&self) {
        let c = self.counts;
        println!(
            "derived sets: found {} / built {} / refused {}; first refusal: {}",
            c[0],
            c[1],
            c[2],
            self.first.as_deref().unwrap_or("none")
        );
        println!(
            "derived texts: equal modulo the block {} / different {}; blocks omitted {}",
            c[3], c[4], c[5]
        );
        println!("twins recorded without their primal: {}", c[6]);
    }
}

/// The recordings a corpus argument names: the `k*.jsonl` files of a folder, in name order, or the one file.
pub fn files_of(path: &Path) -> Vec<std::path::PathBuf> {
    if !path.is_dir() {
        return vec![path.to_path_buf()];
    }
    let mut files: Vec<_> = std::fs::read_dir(path)
        .map(|d| d.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    files.retain(|f| {
        f.file_name().is_some_and(|n| {
            n.to_string_lossy().starts_with('k') && n.to_string_lossy().ends_with(".jsonl")
        })
    });
    files.sort();
    files
}

/// Replays every derived set of the calls in `file` as one request each and prints the counts; the meta source and the texts of the set
/// of the call `only` are written into `dir`.
pub fn census(file: &Path, only: Option<&str>, dir: Option<&Path>) -> Totals {
    let text_of_file = std::fs::read_to_string(file).expect("the calls");
    let texts = file.with_file_name(
        file.file_name()
            .map_or(String::new(), |n| n.to_string_lossy().replacen('k', "t", 1)),
    );
    let mut machines = HashMap::new();
    let mut expected = None;
    let (mut found, mut built, mut refused, mut first) = (0, 0, 0, None);
    let mut lone = 0;
    let (mut equal, mut different, mut omitted) = (0, 0, 0);
    for line in text_of_file.lines() {
        let mut call = json::parse(line).expect("a call");
        if only.is_some_and(|wanted| call.get("call").and_then(J::str) != Some(wanted)) {
            continue;
        }
        if texts.is_file() {
            let both = expected.get_or_insert_with(|| {
                let wanted = needed_keys(file);
                [
                    texts_of(&texts, &wanted, "host"),
                    texts_of(&texts, &wanted, "cuda"),
                ]
            });
            name_as_built(&mut call);
            recorded_broadcast(&mut call, both);
        }
        let id = call.get("call").and_then(J::str).unwrap_or("").to_string();
        let kernels: Vec<&J> = call.get("kernels").map_or(&[][..], J::arr).iter().collect();
        let mut ctx = None;
        for primal in &kernels {
            let name = primal.get("name").and_then(J::str).unwrap_or("");
            let mine: Vec<&J> = kernels
                .iter()
                .copied()
                .filter(|k| {
                    let kname = k.get("name").and_then(J::str).unwrap_or("");
                    let reverse = text(k.get("derived").and_then(|d| d.get("primal")))
                        == Some(name)
                        && (kname == format!("{name}_step_vjp")
                            || REFINE_SUFFIXES
                                .iter()
                                .any(|(suffix, _)| kname == format!("{name}{suffix}")));
                    kname == format!("{name}_tape") || reverse
                })
                .collect();
            if mine.is_empty() {
                continue;
            }
            let answered = one_set(&call, &mine, primal, &id, dir, &mut machines, &mut ctx);
            if matches!(&answered, Built::Refused(why) if is_lone_twin(why)) {
                lone += 1;
                continue;
            }
            found += 1;
            match answered {
                Built::Answered(e, d, o) => {
                    built += 1;
                    (equal, different, omitted) = (equal + e, different + d, omitted + o);
                }
                Built::Refused(why) => {
                    if only.is_some() {
                        println!("refused: {why}");
                    }
                    refused += 1;
                    first.get_or_insert(why);
                }
            }
        }
        if let Some((c, _)) = ctx {
            osprey_host_free(c);
        }
    }
    Totals {
        counts: [found, built, refused, equal, different, omitted, lone],
        first,
    }
}

/// The comparisons of a recorded stop rule, read from the recorded nodes alone: the leaves of the `or` chain under the finish value
/// when every one is `<`, `<=`, `>` or `>=`, else none.
fn recorded_events(nodes: &[J], value: usize) -> usize {
    let kind = |i: usize| nodes.get(i).and_then(|n| text(rec(n).get("kind")));
    let operands = |i: usize| -> Vec<usize> {
        let held = nodes.get(i).map(rec).and_then(|n| n.get("operands"));
        items(held)
            .iter()
            .map(|r| r.get("n").map_or(0, J::int) as usize)
            .collect()
    };
    let mut leaves = vec![];
    let mut todo = vec![value];
    while let Some(at) = todo.pop() {
        if kind(at) == Some("lor") {
            todo.extend(operands(at).into_iter().rev());
        } else {
            leaves.push(at);
        }
    }
    let comparison = |at: &usize| matches!(kind(*at), Some("lt" | "le" | "gt" | "ge"));
    if leaves.iter().all(comparison) {
        leaves.len()
    } else {
        0
    }
}

/// The tally of the event counts of a recording: kernels compared, equal, different, not built, the distribution of the counts, and
/// the library's count for the kernel of the call named by the caller.
#[derive(Default)]
pub struct Events {
    /// Kernels with a finish that were built.
    pub compared: usize,
    /// Of them, those whose library count equals the recorded reading.
    pub equal: usize,
    /// The kernels with a finish the library refused.
    pub not_built: usize,
    /// The differing kernels, by call and name.
    pub different: Vec<String>,
    /// How many kernels have each count (index = count).
    pub counts: [usize; 10],
    /// The library's counts of the kernels named `ev_osc`.
    pub ev_osc: Vec<usize>,
}

impl Events {
    /// Adds the tally of another recording.
    pub fn add(&mut self, other: Events) {
        self.compared += other.compared;
        self.equal += other.equal;
        self.not_built += other.not_built;
        self.different.extend(other.different);
        for (a, b) in self.counts.iter_mut().zip(other.counts) {
            *a += b;
        }
        self.ev_osc.extend(other.ev_osc);
    }

    /// The summary lines.
    pub fn print(&self) {
        println!(
            "events: kernels with a finish compared {} / equal {} / different {}; not built {}",
            self.compared,
            self.equal,
            self.different.len(),
            self.not_built
        );
        let spread: Vec<String> = self
            .counts
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 0)
            .map(|(count, n)| format!("{count} events: {n}"))
            .collect();
        println!("events: distribution of the counts: {}", spread.join("; "));
        println!("events: ev_osc library counts: {:?}", self.ev_osc);
        for d in self.different.iter().take(10) {
            println!("events: DIFFERENT {d}");
        }
    }
}

/// Compares, for every kernel of `file` that has a finish, the library's `n_events` with the recorded reading of its stop rule.
pub fn event_census(file: &Path) -> Events {
    let text_of_file = std::fs::read_to_string(file).expect("the calls");
    let mut machines = HashMap::new();
    let mut tally = Events::default();
    for line in text_of_file.lines() {
        let call = json::parse(line).expect("a call");
        let id = call.get("call").and_then(J::str).unwrap_or("").to_string();
        let nodes = call.get("nodes").map_or(&[][..], J::arr);
        let mut ctx = None;
        for kernel in call.get("kernels").map_or(&[][..], J::arr) {
            let finish = kernel
                .get("sinks")
                .map_or(&[][..], J::arr)
                .iter()
                .filter_map(|s| s.get("n").map(|n| n.int() as usize))
                .find(|&i| {
                    nodes
                        .get(i)
                        .is_some_and(|n| n.get("c").and_then(J::str) == Some("Finish"))
                });
            let Some(finish) = finish else { continue };
            let held = rec(&nodes[finish]).get("operands");
            let value = items(held)
                .first()
                .and_then(|r| r.get("n"))
                .map_or(0, J::int) as usize;
            let want = recorded_events(nodes, value);
            let name = kernel.get("name").and_then(J::str).unwrap_or("");
            let units = call.get("units").map_or(&[][..], J::arr);
            let side = [(Target::Host, "host"), (Target::Device, "cuda")]
                .into_iter()
                .find(|(_, s)| unit_targets(kernel, units).contains(s));
            let Some((target, _)) = side else { continue };
            let found = Found {
                call: &call,
                id: &id,
                show: 0,
                target,
            };
            let got = (|| {
                let mut facts = facts_all(&found, kernel, &mut machines)?;
                facts.retain(|(t, _)| *t == target);
                if ctx.is_none() {
                    ctx = Some(load(&call)?);
                }
                let (c, handles) = ctx.as_ref().ok_or("no context")?;
                let answer = ask(*c, &request_for(&call, kernel, facts, handles)?.to_json())?;
                let n = answer
                    .units
                    .iter()
                    .find_map(|u| u.meta.finish.as_ref().map(|f| f.n_events));
                Ok::<_, String>(n.unwrap_or(-1))
            })();
            match got {
                Ok(n) if n >= 0 => {
                    tally.compared += 1;
                    tally.counts[(n as usize).min(9)] += 1;
                    if n as usize == want {
                        tally.equal += 1;
                    } else {
                        tally
                            .different
                            .push(format!("{id} {name}: library {n}, recorded {want}"));
                    }
                    if name == "ev_osc" {
                        tally.ev_osc.push(n as usize);
                    }
                }
                _ => tally.not_built += 1,
            }
        }
        if let Some((c, _)) = ctx {
            osprey_host_free(c);
        }
    }
    tally
}

/// Counts, over the answers of single kernels in `file`, the texts after the first unit of an answer that carry the self-check block
/// (the ones a "one block per answer" rule would change), per target.
pub fn split_blocks(file: &Path) -> [usize; 3] {
    let text_of_file = std::fs::read_to_string(file).expect("the calls");
    let texts = file.with_file_name(
        file.file_name()
            .map_or(String::new(), |n| n.to_string_lossy().replacen('k', "t", 1)),
    );
    let mut machines = HashMap::new();
    let mut expected = None;
    let (mut host, mut device, mut answers) = (0, 0, 0);
    for line in text_of_file.lines() {
        let mut call = json::parse(line).expect("a call");
        if texts.is_file() {
            let both = expected.get_or_insert_with(|| {
                let wanted = needed_keys(file);
                [
                    texts_of(&texts, &wanted, "host"),
                    texts_of(&texts, &wanted, "cuda"),
                ]
            });
            name_as_built(&mut call);
            recorded_broadcast(&mut call, both);
        }
        let id = call.get("call").and_then(J::str).unwrap_or("").to_string();
        let units = call.get("units").map_or(&[][..], J::arr);
        let mut ctx = None;
        for kernel in call.get("kernels").map_or(&[][..], J::arr) {
            for (target, side) in SIDES {
                if !unit_targets(kernel, units).contains(&side) {
                    continue;
                }
                let found = Found {
                    call: &call,
                    id: &id,
                    show: 0,
                    target,
                };
                let answer = (|| {
                    let mut facts = facts_all(&found, kernel, &mut machines)?;
                    facts.retain(|(asked, _)| *asked == target);
                    if ctx.is_none() {
                        ctx = Some(load(&call)?);
                    }
                    let (c, handles) = ctx.as_ref().ok_or("no context")?;
                    ask(*c, &request_for(&call, kernel, facts, handles)?.to_json())
                })();
                let Ok(answer) = answer else { continue };
                answers += 1;
                let later = answer
                    .units
                    .iter()
                    .skip(1)
                    .flat_map(|u| &u.texts)
                    .filter(|t| t.target == target && t.text.contains("eagle_abi_tag"));
                let n = later.count();
                if side == "host" {
                    host += n
                } else {
                    device += n
                }
            }
        }
        if let Some((c, _)) = ctx {
            osprey_host_free(c);
        }
    }
    [answers, host, device]
}

#[cfg(test)]
mod tests {
    use super::*;
    use meta::{Slot, UnitMeta};

    /// A unit with the slots given as (name, role, element type).
    fn unit(slots: &[(String, SlotRole, WireDtype)]) -> UnitMeta {
        let mut out = UnitMeta::default();
        for (name, role, dtype) in slots {
            out.body.slots.push(Slot {
                name: name.clone(),
                role: *role,
                dtype: *dtype,
                ..Slot::default()
            });
        }
        out
    }

    /// The links a refine unit's slot names give, as (slot, what, form, of, reduced) words.
    fn links(role: DerivedRole, own: &[(String, SlotRole, WireDtype)]) -> Vec<String> {
        use SlotRole::{Mutable, PerSample, Uniform};
        let real = WireDtype::Real;
        let named = |n: &str, role| (n.to_string(), role, real);
        let mut primal = unit(&[
            named("t", Mutable),
            named("v", Mutable),
            named("x", Mutable),
            named("a", Uniform),
            named("dt", Uniform),
            named("w", Uniform),
        ]);
        for s in primal.body.slots.iter_mut().take(3) {
            s.flags |= meta::SLOT_PRIOR;
        }
        let _ = PerSample;
        let names = RefineNames {
            time: "t",
            step: "dt",
            twin_steps: Some("applied_steps"),
        };
        refine_facts(role, &primal, &unit(own), &names)
            .unwrap()
            .links
            .iter()
            .map(|l| {
                format!(
                    "{} {:?} {:?} {} {}",
                    l.slot,
                    l.what,
                    l.form,
                    l.of.as_deref().unwrap_or("-"),
                    l.reduced
                )
            })
            .collect()
    }

    #[test]
    fn the_refine_units_of_the_event_oscillator_get_the_loaded_records_links() {
        use SlotRole::{Mutable, PerSample, Uniform};
        let (real, int) = (WireDtype::Real, WireDtype::Int64);
        let slots = |names: &[&str], role, dtype| -> Vec<(String, SlotRole, WireDtype)> {
            names.iter().map(|n| (n.to_string(), role, dtype)).collect()
        };
        let own =
            |parts: Vec<Vec<(String, SlotRole, WireDtype)>>| -> Vec<(String, SlotRole, WireDtype)> {
                parts.concat()
            };
        // the Newton unit, slot by slot as the loaded record lists them
        let newton = own(vec![
            slots(
                &[
                    "dot_event_g0",
                    "dot_t_refined",
                    "dot_v_refined",
                    "dot_x_refined",
                    "event_g0",
                    "t_refined",
                    "v_refined",
                    "x_refined",
                ],
                Mutable,
                real,
            ),
            slots(&["applied_steps"], PerSample, int),
            slots(
                &["dot_dt_tau", "dt", "dt_tau", "t_0", "v", "x"],
                PerSample,
                real,
            ),
            slots(&["a", "w"], Uniform, real),
        ]);
        assert_eq!(
            links(DerivedRole::RefineNewton, &newton),
            [
                "dot_event_g0 Guard Dot 0 false",
                "dot_t_refined Refined Dot t false",
                "dot_v_refined Refined Dot v false",
                "dot_x_refined Refined Dot x false",
                "event_g0 Guard Value 0 false",
                "t_refined Refined Value t false",
                "v_refined Refined Value v false",
                "x_refined Refined Value x false",
                "applied_steps Steps Value applied_steps false",
                "dot_dt_tau Tau Seed - false",
                "dt TimeStep Value dt false",
                "dt_tau Tau Value - false",
                "t_0 Time0 Value t false",
                "v TapeRow Value v false",
                "x TapeRow Value x false",
                "a PrimalSlot Value a false",
                "w PrimalSlot Value w false",
            ]
        );
        let guard = own(vec![
            slots(&["bar_a", "bar_t", "bar_x"], Mutable, real),
            slots(&["bar_event_g0", "t"], PerSample, real),
            slots(&["a"], Uniform, real),
        ]);
        assert_eq!(
            links(DerivedRole::StateGuardVjp, &guard),
            [
                "bar_a PrimalSlot Grad a true",
                "bar_t Refined AdjointOut t false",
                "bar_x Refined AdjointOut x false",
                "bar_event_g0 Guard AdjointIn 0 false",
                "t Refined Value t false",
                "a PrimalSlot Value a false",
            ]
        );
        let refine = own(vec![
            slots(
                &[
                    "bar_a",
                    "bar_dt",
                    "bar_dt_tau",
                    "bar_t_0",
                    "bar_v",
                    "bar_w",
                    "bar_x",
                ],
                Mutable,
                real,
            ),
            slots(&["applied_steps"], PerSample, int),
            slots(
                &[
                    "bar_event_g0",
                    "bar_t_refined",
                    "bar_v_refined",
                    "bar_x_refined",
                    "dt",
                    "dt_tau",
                    "t_0",
                    "v",
                    "x",
                ],
                PerSample,
                real,
            ),
            slots(&["a", "w"], Uniform, real),
        ]);
        assert_eq!(
            links(DerivedRole::RefineVjp, &refine),
            [
                "bar_a PrimalSlot Grad a true",
                "bar_dt TimeStep Grad dt true",
                "bar_dt_tau Tau Discard - false",
                "bar_t_0 Time0 Grad t false",
                "bar_v PrimalSlot AdjointOut v false",
                "bar_w PrimalSlot Grad w true",
                "bar_x PrimalSlot AdjointOut x false",
                "applied_steps Steps Value applied_steps false",
                "bar_event_g0 Guard AdjointIn 0 false",
                "bar_t_refined Refined AdjointIn t false",
                "bar_v_refined Refined AdjointIn v false",
                "bar_x_refined Refined AdjointIn x false",
                "dt TimeStep Value dt false",
                "dt_tau Tau Value - false",
                "t_0 Time0 Value t false",
                "v TapeRow Value v false",
                "x TapeRow Value x false",
                "a PrimalSlot Value a false",
                "w PrimalSlot Value w false",
            ]
        );
    }
}
