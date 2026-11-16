//! The derived units of a kernel: a taping twin or a reverse unit of a primal kernel. The caller states, by slot name, what each
//! derived unit is to the primal (its tape, its reserved slots, where each slot comes from); the library resolves the names to the
//! slot indices of the units it lowers, range-checks everything stated, and writes the record into the unit's meta.
use super::request::{
    flag_of, int_of, list_of, member_of, name_of, read_part, text_of, Body, KernelRequest, Keys,
    Member, Settings,
};
use super::unit_meta::{
    Derived, DerivedRole, DerivedSource, Link, SlotRole, Taped, UnitMeta, WireDtype,
};
use aether_writers::contract::Json;
use aether_writers::refuse::{bad, not_written, Refusal};

/// One taped plane the caller states: a plane of the primal and its rows in the tape.
#[derive(Debug, Clone, PartialEq)]
pub struct TapedPlane {
    /// The plane's name in the primal.
    pub plane: String,
    /// The rows of one recorded step it takes.
    pub rows: i64,
}

/// One slot of the derived unit and where it comes from, by name.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkFact {
    /// The slot's name in the derived unit.
    pub slot: String,
    /// Where it comes from.
    pub source: DerivedSource,
    /// The primal's slot it is of (a step plane names a slot of the taping twin); none where the source has no slot.
    pub of: Option<String>,
    /// Whether a parameter's gradient is summed over samples.
    pub reduced: bool,
}

/// What the caller states of a derived unit; every slot is named, a slot left out is absent.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DerivedFacts {
    /// The tape slot of a taping twin.
    pub tape: Option<String>,
    /// The rows of one recorded step.
    pub tape_rows: i64,
    /// The taped planes, in tape order.
    pub taped: Vec<TapedPlane>,
    /// The applied steps slot.
    pub steps: Option<String>,
    /// The slot that switches taping.
    pub tape_on: Option<String>,
    /// The start copy of the time plane.
    pub time0: Option<String>,
    /// The per-sample step of a refine run.
    pub tau: Option<String>,
    /// The word of steps a fused unit runs.
    pub fused_steps: Option<String>,
    /// The slots' sources.
    pub links: Vec<LinkFact>,
}

/// A derived kernel: the request of its own lowering, its role and what the caller states of it.
#[derive(Debug, Clone, PartialEq)]
pub struct DerivedKernel {
    /// The kernel's own request: the primal's targets, scalar and facts, its own body and per-kernel members.
    pub request: KernelRequest,
    /// What the unit is to the primal.
    pub role: DerivedRole,
    /// The stated facts.
    pub facts: DerivedFacts,
}

/// A name or none.
fn name_opt(keys: &mut Keys, key: &'static str) -> Result<Option<String>, Refusal> {
    keys.opt(key).map(|j| text_of(j, key)).transpose()
}

/// What is stated of a derived kernel besides its request: the name of its unit, its role and its facts.
pub(super) struct Stated {
    /// The unit's name.
    pub name: String,
    /// What the unit is to the primal.
    pub role: DerivedRole,
    /// The stated facts.
    pub facts: DerivedFacts,
}

impl DerivedKernel {
    /// What is stated of the kernel, to be resolved.
    pub(super) fn stated(&self) -> Stated {
        Stated {
            name: self.name().to_string(),
            role: self.role,
            facts: self.facts.clone(),
        }
    }

    /// The kernel's name, which is its unit's name.
    pub fn name(&self) -> &str {
        match &self.request.body {
            Body::Plain(member) => &member.name,
            _ => "",
        }
    }

    /// Reads a derived kernel over the kernel `base`: a request of its own that takes the base's targets, scalar, facts and kind, and the
    /// per-kernel members the entry states, read by the function that reads the base's.
    pub(super) fn read(json: &Json, base: &KernelRequest) -> Result<DerivedKernel, Refusal> {
        let mut keys = Keys::of(json, "derived")?;
        let member = Member::read(keys.req("member")?)?;
        let role = member_of(DerivedRole::ALL, keys.req("role")?, "role")?;
        let mut request = KernelRequest {
            name: member.name.clone(),
            body: Body::Plain(member),
            one_step: None,
            derived: vec![],
            group: aether_writers::contract::types::DEFAULT_GROUP,
            planes: vec![],
            settings: Settings::read(&Json::Obj(vec![]))?,
            steps: None,
            lane_space: vec![],
            sink_axes: vec![],
            shared: vec![],
            broadcast: vec![],
            ..base.clone()
        };
        read_part(&mut keys, &mut request)?;
        let links = match keys.opt("links") {
            Some(l) => list_of(l, "links", |one| {
                let mut keys = Keys::of(one, "links")?;
                let link = LinkFact {
                    slot: text_of(keys.req("slot")?, "slot")?,
                    source: member_of(DerivedSource::ALL, keys.req("source")?, "source")?,
                    of: name_opt(&mut keys, "of")?,
                    reduced: keys
                        .opt("reduced")
                        .map_or(Ok(false), |f| flag_of(f, "reduced"))?,
                };
                keys.done()?;
                Ok(link)
            })?,
            None => vec![],
        };
        let taped = match keys.opt("taped") {
            Some(l) => list_of(l, "taped", |one| {
                let mut keys = Keys::of(one, "taped")?;
                let plane = TapedPlane {
                    plane: text_of(keys.req("plane")?, "plane")?,
                    rows: int_of(keys.req("rows")?, "rows")?,
                };
                keys.done()?;
                Ok(plane)
            })?,
            None => vec![],
        };
        let facts = DerivedFacts {
            tape: name_opt(&mut keys, "tape")?,
            tape_rows: keys
                .opt("tape_rows")
                .map_or(Ok(0), |n| int_of(n, "tape_rows"))?,
            taped,
            steps: name_opt(&mut keys, "steps_slot")?,
            tape_on: name_opt(&mut keys, "tape_on")?,
            time0: name_opt(&mut keys, "time0")?,
            tau: name_opt(&mut keys, "tau")?,
            fused_steps: name_opt(&mut keys, "fused_steps")?,
            links,
        };
        keys.done()?;
        Ok(DerivedKernel {
            request,
            role,
            facts,
        })
    }

    /// Writes the derived kernel.
    pub(super) fn write(&self) -> Json {
        let text = |s: &str| Json::Str(s.to_string());
        let member = match &self.request.body {
            Body::Plain(member) => member.write(),
            _ => Json::Null,
        };
        let mut o = vec![
            ("member".to_string(), member),
            ("role".into(), name_of(&self.role)),
            ("kind".into(), self.request.kind.write()),
        ];
        o.extend(self.request.part_members());
        let f = &self.facts;
        for (key, name) in [
            ("tape", &f.tape),
            ("steps_slot", &f.steps),
            ("tape_on", &f.tape_on),
            ("time0", &f.time0),
            ("tau", &f.tau),
            ("fused_steps", &f.fused_steps),
        ] {
            if let Some(name) = name {
                o.push((key.into(), text(name)))
            }
        }
        o.push(("tape_rows".into(), Json::Num(f.tape_rows.to_string())));
        o.push((
            "taped".into(),
            Json::Arr(
                f.taped
                    .iter()
                    .map(|t| {
                        Json::Obj(vec![
                            ("plane".into(), text(&t.plane)),
                            ("rows".into(), Json::Num(t.rows.to_string())),
                        ])
                    })
                    .collect(),
            ),
        ));
        o.push((
            "links".into(),
            Json::Arr(
                f.links
                    .iter()
                    .map(|l| {
                        let mut one = vec![
                            ("slot".to_string(), text(&l.slot)),
                            ("source".into(), name_of(&l.source)),
                        ];
                        if let Some(of) = &l.of {
                            one.push(("of".into(), text(of)))
                        }
                        one.push(("reduced".into(), Json::Bool(l.reduced)));
                        Json::Obj(one)
                    })
                    .collect(),
            ),
        ));
        Json::Obj(o)
    }
}

/// The units a derived kernel is resolved against.
pub(super) struct Context<'a> {
    /// The primal's unit.
    pub primal: &'a UnitMeta,
    /// The primal's index in the table.
    pub of_unit: usize,
    /// The set's taping twin, whose step slot a step plane names.
    pub twin: Option<&'a UnitMeta>,
}

/// The slot of a name in a unit.
fn slot_in(unit: &UnitMeta, name: &str) -> Option<usize> {
    unit.body.slots.iter().position(|s| s.name == name)
}

/// The refusal of a stated fact: the kernel and the member are named.
fn refuse(kernel: &str, member: &str, why: impl std::fmt::Display) -> Refusal {
    bad(format!("derived kernel {kernel}: {member}: {why}"))
}

/// Resolves what the caller states of the derived unit `own` (the unit of the kernel `kernel`) to its record, checking every
/// stated fact; a refusal is a bad request naming the kernel and the member.
pub(super) fn resolve(kernel: &Stated, own: &UnitMeta, at: &Context) -> Result<Derived, Refusal> {
    let (name, f) = (kernel.name.as_str(), &kernel.facts);
    let twin = matches!(kernel.role, DerivedRole::Tape | DerivedRole::TapeFused);
    let mine = |member: &str, slot: &Option<String>| -> Result<Option<usize>, Refusal> {
        slot.as_deref()
            .map(|s| {
                slot_in(own, s).ok_or_else(|| refuse(name, member, format!("no slot named {s}")))
            })
            .transpose()
    };
    let (tape_slot, steps_slot, tape_on_slot) = (
        mine("tape", &f.tape)?,
        mine("steps", &f.steps)?,
        mine("tape_on", &f.tape_on)?,
    );
    let (time0_slot, tau_slot, fused_steps_slot) = (
        mine("time0", &f.time0)?,
        mine("tau", &f.tau)?,
        mine("fused_steps", &f.fused_steps)?,
    );
    let mut named: Vec<(&str, usize)> = [
        ("tape", tape_slot),
        ("steps", steps_slot),
        ("tape_on", tape_on_slot),
        ("tau", tau_slot),
        ("fused_steps", fused_steps_slot),
    ]
    .into_iter()
    .filter_map(|(member, slot)| slot.map(|s| (member, s)))
    .collect();
    let mut linked: Vec<usize> = vec![];
    for link in &f.links {
        let slot = slot_in(own, &link.slot)
            .ok_or_else(|| refuse(name, "links", format!("no slot named {}", link.slot)))?;
        if linked.contains(&slot) || named.iter().any(|n| n.1 == slot) {
            return Err(refuse(
                name,
                "links",
                format!("the slot {} is named twice", link.slot),
            ));
        }
        linked.push(slot);
    }
    named.dedup();
    if let Some(at) = (0..named.len()).find(|&i| named[..i].iter().any(|n| n.1 == named[i].1)) {
        return Err(refuse(name, named[at].0, "the slot is named twice"));
    }
    if twin != tape_slot.is_some() {
        return Err(refuse(
            name,
            "tape",
            if twin {
                "a taping twin has exactly one tape slot"
            } else {
                "only a taping twin has a tape slot"
            },
        ));
    }
    let taped = taped_rows(kernel, own, at, tape_slot)?;
    let links = links_of(kernel, own, at, time0_slot, &f.links)?;
    Ok(Derived {
        role: kernel.role,
        of_unit: at.of_unit,
        tape_slot,
        tape_rows: f.tape_rows,
        steps_slot,
        tape_on_slot,
        time0_slot,
        tau_slot,
        fused_steps_slot,
        taped,
        links,
    })
}

/// The taped planes of the unit with their rows laid out, checked against the tape.
fn taped_rows(
    kernel: &Stated,
    own: &UnitMeta,
    at: &Context,
    tape_slot: Option<usize>,
) -> Result<Vec<Taped>, Refusal> {
    let (name, f) = (kernel.name.as_str(), &kernel.facts);
    if tape_slot.is_none() {
        let none = f.taped.is_empty() && f.tape_rows == 0;
        return if none {
            Ok(vec![])
        } else {
            Err(refuse(name, "taped", "only a taping twin has taped planes"))
        };
    }
    let mut row0 = 0;
    let mut out = vec![];
    for plane in &f.taped {
        let slot = slot_in(at.primal, &plane.plane).ok_or_else(|| {
            refuse(
                name,
                "taped",
                format!("the primal has no slot named {}", plane.plane),
            )
        })?;
        if plane.rows < 1 {
            return Err(refuse(
                name,
                "taped",
                format!("{} takes {} rows", plane.plane, plane.rows),
            ));
        }
        out.push(Taped {
            slot,
            row0,
            rows: plane.rows,
        });
        row0 += plane.rows;
    }
    if row0 != f.tape_rows {
        return Err(refuse(
            name,
            "tape_rows",
            format!("the taped rows sum to {row0}, not {}", f.tape_rows),
        ));
    }
    let declared: i64 = tape_slot.map_or(0, |s| own.body.slots[s].shape.iter().product());
    if f.tape_rows < 1 || declared < 1 || declared % f.tape_rows != 0 {
        return Err(refuse(
            name,
            "tape",
            format!(
                "its declared rows {declared} are not a whole multiple of tape_rows {}",
                f.tape_rows
            ),
        ));
    }
    Ok(out)
}

/// The slots the unit's own record names as the runtime's: the finish counter and the launch's active map and count.
fn runtime_slots(unit: &UnitMeta) -> Vec<usize> {
    let finish = unit.finish.as_ref().map(|f| f.counter_slot);
    [
        finish,
        unit.launch.active_map_slot,
        unit.launch.active_count_slot,
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// Checks a time-step link: it is of a uniform or per-sample slot of the primal with a real element type.
fn time_step_of(name: &str, link: &LinkFact, primal: &UnitMeta) -> Result<(), Refusal> {
    let found = link
        .of
        .as_deref()
        .and_then(|of| slot_in(primal, of).map(|at| &primal.body.slots[at]));
    match found {
        Some(s)
            if matches!(s.role, SlotRole::Uniform | SlotRole::PerSample)
                && s.dtype == WireDtype::Real =>
        {
            Ok(())
        }
        _ => Err(refuse(
            name,
            "links",
            format!(
                "the time step {} is of a uniform or per-sample real slot of the primal",
                link.slot
            ),
        )),
    }
}

/// The slot links of the unit, checked against the role.
fn links_of(
    kernel: &Stated,
    own: &UnitMeta,
    at: &Context,
    time0_slot: Option<usize>,
    stated: &[LinkFact],
) -> Result<Vec<Link>, Refusal> {
    let name = kernel.name.as_str();
    let twin = matches!(kernel.role, DerivedRole::Tape | DerivedRole::TapeFused);
    let mut out = vec![];
    let mut time0s = 0;
    let mut steps_of_time = 0;
    let runtime_owned = runtime_slots(own);
    for link in stated {
        let slot = slot_in(own, &link.slot).unwrap_or(0);
        if runtime_owned.contains(&slot) {
            return Err(refuse(
                name,
                "links",
                format!("{} is a slot the runtime owns (the finish counter or the active set) and takes no link", link.slot),
            ));
        }
        if matches!(
            link.source,
            DerivedSource::Event | DerivedSource::Guard | DerivedSource::GuardDot
        ) {
            return Err(not_written(format!("derived kernel {name}: links: the slot {} is an event value ({:?}); the library has no event list", link.slot, link.source)));
        }
        if link.reduced && link.source != DerivedSource::ParamGrad {
            return Err(refuse(
                name,
                "links",
                format!("{} is reduced but is not a parameter gradient", link.slot),
            ));
        }
        if twin && !matches!(link.source, DerivedSource::Primal | DerivedSource::Time0) {
            return Err(refuse(
                name,
                "links",
                format!(
                    "a taping twin's slot {} comes from {:?}",
                    link.slot, link.source
                ),
            ));
        }
        if link.source == DerivedSource::Steps
            && !matches!(
                own.body.slots[slot].dtype,
                WireDtype::Int64 | WireDtype::Int32 | WireDtype::Uint32
            )
        {
            return Err(refuse(
                name,
                "links",
                format!("the step plane {} is not an integer", link.slot),
            ));
        }
        if link.source == DerivedSource::Time0 {
            time0s += 1;
            if twin && Some(slot) != time0_slot {
                return Err(refuse(
                    name,
                    "links",
                    format!("{} is a start copy but is not the time0 slot", link.slot),
                ));
            }
        }
        if link.source == DerivedSource::TimeStep {
            steps_of_time += 1;
            if kernel.role != DerivedRole::RefineNewton {
                return Err(refuse(
                    name,
                    "links",
                    format!(
                        "{} is a time step, which only a refine-newton unit has",
                        link.slot
                    ),
                ));
            }
            time_step_of(name, link, at.primal)?;
        }
        let of_slot = match (&link.of, link.source) {
            (Some(_), DerivedSource::TauSeed) => {
                return Err(refuse(
                    name,
                    "links",
                    format!("{} is a tau seed and names no slot it is of", link.slot),
                ))
            }
            (None, DerivedSource::TauSeed | DerivedSource::Steps) => None,
            (Some(of), DerivedSource::Steps) => {
                Some(at.twin.and_then(|t| slot_in(t, of)).ok_or_else(|| {
                    refuse(name, "links", format!("no step slot {of} in a taping twin"))
                })?)
            }
            (Some(of), _) => {
                let found = slot_in(at.primal, of).ok_or_else(|| {
                    refuse(name, "links", format!("the primal has no slot named {of}"))
                })?;
                let source_dtype = at.primal.body.slots[found].dtype;
                let free = !matches!(
                    link.source,
                    DerivedSource::Primal
                        | DerivedSource::StateAdjointIn
                        | DerivedSource::StateAdjointOut
                        | DerivedSource::Taped
                        | DerivedSource::Time0
                );
                if link.source == DerivedSource::StateGrad
                    && at.primal.body.slots[found].role != SlotRole::Mutable
                {
                    return Err(refuse(
                        name,
                        "links",
                        format!("{of} is not a state of the primal"),
                    ));
                }
                if !free && own.body.slots[slot].dtype != source_dtype {
                    return Err(refuse(
                        name,
                        "links",
                        format!("{} and {of} differ in element type", link.slot),
                    ));
                }
                Some(found)
            }
            (None, _) => {
                return Err(refuse(
                    name,
                    "links",
                    format!("{} names no slot it is of", link.slot),
                ))
            }
        };
        out.push(Link {
            slot,
            source: link.source,
            of_slot,
            reduced: link.reduced,
        });
    }
    if kernel.role == DerivedRole::RefineNewton && steps_of_time != 1 {
        return Err(refuse(
            name,
            "links",
            if steps_of_time == 0 {
                "a refine kernel needs a declared step".to_string()
            } else {
                format!("a refine kernel has {steps_of_time} step slots, not one")
            },
        ));
    }
    if twin && time0s != usize::from(time0_slot.is_some()) {
        return Err(refuse(
            name,
            "time0",
            "a start copy is stated exactly when the twin has one TIME0 link",
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::unit_meta::{Body, Slot, SlotRole};
    use aether_writers::refuse::RefusalKind;

    /// A unit with the slots given as (name, element type, extents).
    fn unit(slots: &[(&str, WireDtype, &[i64])]) -> UnitMeta {
        let slot = |(name, dtype, shape): &(&str, WireDtype, &[i64])| Slot {
            name: name.to_string(),
            dtype: *dtype,
            shape: shape.to_vec(),
            role: SlotRole::Mutable,
            ..Slot::default()
        };
        UnitMeta {
            body: Body {
                slots: slots.iter().map(slot).collect(),
                ..Body::default()
            },
            ..UnitMeta::default()
        }
    }

    /// A primal with a state `x`, a time plane `t` and a parameter `c`, and a taping twin of it that states everything right.
    fn set() -> (UnitMeta, UnitMeta, Stated) {
        let real = WireDtype::Real;
        let primal = unit(&[("x", real, &[]), ("t", real, &[]), ("c", real, &[])]);
        let twin = unit(&[
            ("applied_steps", WireDtype::Int64, &[]),
            ("x", real, &[]),
            ("t", real, &[]),
            ("tape", real, &[40]),
            ("c", real, &[]),
            ("t_0", real, &[]),
        ]);
        let link = |slot: &str, source, of: &str| LinkFact {
            slot: slot.into(),
            source,
            of: Some(of.into()),
            reduced: false,
        };
        let facts = DerivedFacts {
            tape: Some("tape".into()),
            tape_rows: 2,
            taped: vec![
                TapedPlane {
                    plane: "x".into(),
                    rows: 1,
                },
                TapedPlane {
                    plane: "t".into(),
                    rows: 1,
                },
            ],
            steps: Some("applied_steps".into()),
            time0: Some("t_0".into()),
            links: vec![
                link("x", DerivedSource::Primal, "x"),
                link("t", DerivedSource::Primal, "t"),
                link("c", DerivedSource::Primal, "c"),
                link("t_0", DerivedSource::Time0, "t"),
            ],
            ..DerivedFacts::default()
        };
        let kernel = Stated {
            name: "k_tape".into(),
            role: DerivedRole::Tape,
            facts,
        };
        (primal, twin, kernel)
    }

    /// The refusal of the set after `change`, which must name the kernel and `member`.
    fn refused(member: &str, change: impl Fn(&mut Stated, &mut UnitMeta)) -> Refusal {
        let (primal, mut twin, mut kernel) = set();
        change(&mut kernel, &mut twin);
        let at = Context {
            primal: &primal,
            of_unit: 0,
            twin: None,
        };
        let why = resolve(&kernel, &twin, &at).unwrap_err();
        assert!(
            why.reason
                .contains(&format!("derived kernel k_tape: {member}")),
            "{}",
            why.reason
        );
        why
    }

    #[test]
    fn a_right_set_resolves_and_each_wrong_one_is_refused_by_member() {
        let (primal, twin, kernel) = set();
        let at = Context {
            primal: &primal,
            of_unit: 0,
            twin: None,
        };
        let done = resolve(&kernel, &twin, &at).unwrap();
        assert_eq!(
            (done.tape_slot, done.steps_slot, done.time0_slot),
            (Some(3), Some(0), Some(5))
        );
        assert_eq!(
            done.taped.iter().map(|t| t.row0).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            refused("links", |k, _| k.facts.links[0].slot = "nope".into()).kind,
            RefusalKind::BadRequest
        );
        refused("links", |k, _| k.facts.links[1].slot = "x".into());
        refused("tape", |k, _| k.facts.tape = None);
        refused("tape_rows", |k, _| k.facts.tape_rows = 3);
        refused("tape", |_, u| u.body.slots[3].shape = vec![41]);
        refused("links", |k, _| k.facts.links[0].reduced = true);
        refused("links", |k, _| {
            k.facts.links[0].source = DerivedSource::StateAdjointIn
        });
        refused("links", |k, _| k.facts.links[3].slot = "c".into());
        refused("links", |k, _| k.facts.time0 = None);
        refused("time0", |k, _| {
            k.facts.links.pop();
        });
        refused("links", |_, u| u.body.slots[2].dtype = WireDtype::Int32);
        refused("links", |k, u| {
            u.body.slots[0].dtype = WireDtype::Real;
            k.facts.links[0] = LinkFact {
                slot: "applied_steps".into(),
                source: DerivedSource::Steps,
                of: None,
                reduced: false,
            }
        });
        let event = refused("links", |k, _| {
            k.facts.links[0].source = DerivedSource::Event
        });
        assert_eq!(event.kind, RefusalKind::NotWritten);
    }

    #[test]
    fn a_reverse_unit_has_no_tape_and_a_per_sample_slot_may_link_a_uniform() {
        let (primal, _, mut kernel) = set();
        let real = WireDtype::Real;
        let vjp = unit(&[("bar_c", real, &[]), ("c", real, &[])]);
        kernel.role = DerivedRole::StepVjp;
        kernel.facts = DerivedFacts {
            links: vec![
                LinkFact {
                    slot: "bar_c".into(),
                    source: DerivedSource::ParamGrad,
                    of: Some("c".into()),
                    reduced: true,
                },
                LinkFact {
                    slot: "c".into(),
                    source: DerivedSource::Primal,
                    of: Some("c".into()),
                    reduced: false,
                },
            ],
            ..DerivedFacts::default()
        };
        let at = Context {
            primal: &primal,
            of_unit: 0,
            twin: None,
        };
        assert!(resolve(&kernel, &vjp, &at).is_ok());
        kernel.facts.tape = Some("c".into());
        assert!(resolve(&kernel, &vjp, &at).is_err());
    }

    /// A refine-newton unit over a primal with a state `x`, a uniform `dt` and an integer `n`, with the links given as (slot, source, of).
    fn newton(
        links: &[(&str, DerivedSource, Option<&str>)],
        role: DerivedRole,
    ) -> Result<Derived, Refusal> {
        let real = WireDtype::Real;
        let mut primal = unit(&[
            ("x", real, &[]),
            ("dt", real, &[]),
            ("n", WireDtype::Int32, &[]),
        ]);
        primal.body.slots[1].role = SlotRole::Uniform;
        let own = unit(&[("h", real, &[]), ("tau", real, &[]), ("w", real, &[])]);
        let facts = DerivedFacts {
            links: links
                .iter()
                .map(|(slot, source, of)| LinkFact {
                    slot: slot.to_string(),
                    source: *source,
                    of: of.map(String::from),
                    reduced: false,
                })
                .collect(),
            ..DerivedFacts::default()
        };
        let kernel = Stated {
            name: "k_newton".into(),
            role,
            facts,
        };
        resolve(
            &kernel,
            &own,
            &Context {
                primal: &primal,
                of_unit: 0,
                twin: None,
            },
        )
    }

    #[test]
    fn the_refine_sources_and_the_runtime_slots_are_checked_by_name() {
        use DerivedSource::{Guard, GuardDot, Primal, StateGrad, TauSeed, TimeStep};
        let ok = [("h", TimeStep, Some("dt")), ("tau", TauSeed, None)];
        let done = newton(&ok, DerivedRole::RefineNewton).unwrap();
        assert_eq!(done.links[1].of_slot, None);
        let why =
            |links: &[(&str, DerivedSource, Option<&str>)], role| newton(links, role).unwrap_err();
        let needs = why(&[("tau", TauSeed, None)], DerivedRole::RefineNewton);
        assert!(
            needs
                .reason
                .contains("a refine kernel needs a declared step"),
            "{}",
            needs.reason
        );
        assert!(why(
            &[("h", TimeStep, Some("dt")), ("w", TimeStep, Some("dt"))],
            DerivedRole::RefineNewton
        )
        .reason
        .contains("2 step slots"));
        assert!(why(&ok, DerivedRole::StepVjp)
            .reason
            .contains("only a refine-newton unit"));
        assert!(
            why(&[("h", TimeStep, Some("x"))], DerivedRole::RefineNewton)
                .reason
                .contains("uniform or per-sample real")
        );
        assert!(
            why(&[("h", TimeStep, Some("n"))], DerivedRole::RefineNewton)
                .reason
                .contains("uniform or per-sample real")
        );
        assert!(why(
            &[("h", TimeStep, Some("dt")), ("tau", TauSeed, Some("x"))],
            DerivedRole::RefineNewton
        )
        .reason
        .contains("tau seed"));
        for source in [Guard, GuardDot] {
            let refused = why(
                &[("h", TimeStep, Some("dt")), ("w", source, Some("x"))],
                DerivedRole::RefineNewton,
            );
            assert_eq!(refused.kind, RefusalKind::NotWritten);
        }
        let grad = why(
            &[("h", TimeStep, Some("dt")), ("w", StateGrad, Some("dt"))],
            DerivedRole::RefineNewton,
        );
        assert!(grad.reason.contains("not a state"), "{}", grad.reason);
        assert!(newton(
            &[("h", TimeStep, Some("dt")), ("w", StateGrad, Some("x"))],
            DerivedRole::RefineNewton
        )
        .is_ok());
        // a slot the unit's own record names as the runtime's takes no link
        let (primal, mut own, _) = set();
        own.finish = Some(crate::lower::unit_meta::Finish {
            counter_slot: 4,
            ..Default::default()
        });
        let kernel = Stated {
            name: "k".into(),
            role: DerivedRole::StepVjp,
            facts: DerivedFacts {
                links: vec![LinkFact {
                    slot: "c".into(),
                    source: Primal,
                    of: Some("c".into()),
                    reduced: false,
                }],
                ..DerivedFacts::default()
            },
        };
        let refused = resolve(
            &kernel,
            &own,
            &Context {
                primal: &primal,
                of_unit: 0,
                twin: None,
            },
        )
        .unwrap_err();
        assert!(
            refused.reason.contains("the runtime owns"),
            "{}",
            refused.reason
        );
        assert_eq!(refused.kind, RefusalKind::BadRequest);
    }
}
