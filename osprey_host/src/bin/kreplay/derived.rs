//! The derived sets of a recording: a call whose kernels hold a primal and its taping twin and reverse unit is replayed as ONE request
//! with the list of derived kernels. The stated facts are converted from the reserved names the old build used (`tape`,
//! `applied_steps`, `tape_on`, `<time>_0`, `bar_*`, `*_next`): a guess of this proof tool, not a rule of the library.
use super::*;
use meta::{DerivedRole, DerivedSource, SlotRole, WireDtype};
use osprey_host::lower::derived::{DerivedFacts, DerivedKernel, LinkFact, TapedPlane};

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
            (DerivedSource::Time0, time.map(String::from))
        } else if slot(primal, &s.name).is_some() {
            (DerivedSource::Primal, Some(s.name.clone()))
        } else {
            return Err(format!("the twin's slot {} has no primal slot", s.name));
        };
        facts.links.push(LinkFact {
            slot: s.name.clone(),
            source: link.0,
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
        let (source, of, reduced) =
            if let Some(x) = n.strip_prefix("bar_").and_then(|r| r.strip_suffix("_next")) {
                (DerivedSource::StateAdjointIn, Some(x.to_string()), false)
            } else if time.is_some_and(|t| n == format!("bar_{t}_0")) {
                (DerivedSource::Time0, time.map(String::from), false)
            } else if let Some(x) = n.strip_prefix("bar_") {
                if state(x).is_some() {
                    (DerivedSource::StateAdjointOut, Some(x.to_string()), false)
                } else if slot(primal, x).is_some() {
                    (DerivedSource::ParamGrad, Some(x.to_string()), true)
                } else {
                    return Err(format!("the reverse unit's slot {n} has no primal"));
                }
            } else if n == "applied_steps" {
                (
                    DerivedSource::Steps,
                    twin.and_then(|t| t.steps.clone()),
                    false,
                )
            } else if taped.iter().any(|t| t == n) {
                (DerivedSource::Taped, Some(n.to_string()), false)
            } else if slot(primal, n).is_some() {
                (DerivedSource::Primal, Some(n.to_string()), false)
            } else {
                return Err(format!("the reverse unit's slot {n} has no primal"));
            };
        facts.links.push(LinkFact {
            slot: s.name.clone(),
            source,
            of,
            reduced,
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

/// The counts of a replay of derived sets: found, built, refused, texts equal modulo the block, texts different, blocks omitted, and
/// the first refusal.
#[derive(Default)]
pub struct Totals {
    /// The six counts.
    pub counts: [usize; 6],
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
                        && kname == format!("{name}_step_vjp");
                    kname == format!("{name}_tape") || reverse
                })
                .collect();
            if mine.is_empty() {
                continue;
            }
            found += 1;
            match one_set(&call, &mine, primal, &id, dir, &mut machines, &mut ctx) {
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
        counts: [found, built, refused, equal, different, omitted],
        first,
    }
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
