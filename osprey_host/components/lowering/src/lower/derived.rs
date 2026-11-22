//! The derived units of a kernel: a taping twin or a reverse unit of a primal kernel. The caller states, by slot name, what each
//! derived unit is to the primal (its tape, its reserved slots, where each slot comes from); the library resolves the names to the
//! slot indices of the units it lowers, range-checks everything stated, and writes the record into the unit's meta.
use super::links::{links_of, role_has_rows, LinkFact};
use super::request::{
    flag_of, int_of, list_of, member_of, name_of, read_part, text_of, Body, KernelRequest, Keys,
    Member, Settings,
};
use super::unit_meta::{Derived, DerivedRole, LinkForm, LinkWhat, Taped, UnitMeta};
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
            shared: vec![],
            broadcast: vec![],
            table_payloads: vec![],
            ..base.clone()
        };
        read_part(&mut keys, &mut request)?;
        let links = match keys.opt("links") {
            Some(l) => list_of(l, "links", |one| {
                let mut keys = Keys::of(one, "links")?;
                let link = LinkFact {
                    slot: text_of(keys.req("slot")?, "slot")?,
                    what: member_of(LinkWhat::ALL, keys.req("what")?, "what")?,
                    form: member_of(LinkForm::ALL, keys.req("form")?, "form")?,
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
                            ("what".into(), name_of(&l.what)),
                            ("form".into(), name_of(&l.form)),
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
    if !role_has_rows(kernel.role) {
        return Err(not_written(format!(
            "derived kernel {name}: the role {:?} has no link rows in the library",
            kernel.role
        )));
    }
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
    let (time0_slot, fused_steps_slot) = (
        mine("time0", &f.time0)?,
        mine("fused_steps", &f.fused_steps)?,
    );
    let mut named: Vec<(&str, usize)> = [
        ("tape", tape_slot),
        ("steps", steps_slot),
        ("tape_on", tape_on_slot),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::unit_meta::{Body, LinkForm, LinkWhat, Slot, SlotRole, WireDtype};
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
        let link = |slot: &str, what, of: &str| LinkFact {
            slot: slot.into(),
            what,
            form: LinkForm::Value,
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
                link("x", LinkWhat::PrimalSlot, "x"),
                link("t", LinkWhat::PrimalSlot, "t"),
                link("c", LinkWhat::PrimalSlot, "c"),
                link("t_0", LinkWhat::Time0, "t"),
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
        // each check of the links is refused by name: reduced without a gradient, a triple that is no row of the role
        refused("links", |k, _| k.facts.links[0].reduced = true);
        refused("links", |k, _| k.facts.links[0].form = LinkForm::AdjointIn);
        refused("links", |k, _| k.facts.links[3].form = LinkForm::Grad);
        refused("links", |k, _| {
            k.facts.links.push(LinkFact {
                slot: "t".into(),
                what: LinkWhat::Time0,
                form: LinkForm::Value,
                of: Some("t".into()),
                reduced: false,
            });
            k.facts.links.remove(1);
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
                what: LinkWhat::Steps,
                form: LinkForm::Value,
                of: None,
                reduced: false,
            }
        });
        let guard = refused("links", |k, _| {
            k.role = DerivedRole::StateGuardVjp;
            k.facts = DerivedFacts {
                links: vec![k.facts.links[0].clone()],
                ..DerivedFacts::default()
            };
            k.facts.links[0].what = LinkWhat::Guard;
            k.facts.links[0].form = LinkForm::AdjointIn;
        });
        assert_eq!(guard.kind, RefusalKind::BadRequest);
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
                    what: LinkWhat::PrimalSlot,
                    form: LinkForm::Grad,
                    of: Some("c".into()),
                    reduced: true,
                },
                LinkFact {
                    slot: "c".into(),
                    what: LinkWhat::PrimalSlot,
                    form: LinkForm::Value,
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

    /// A link as (slot, what, form, of).
    type Stated4<'a> = (&'a str, LinkWhat, LinkForm, Option<&'a str>);

    /// A refine unit over a primal with no events.
    fn newton(links: &[Stated4], role: DerivedRole) -> Result<Derived, Refusal> {
        newton_over(links, role, 0)
    }

    /// A refine unit over a primal with a state `x`, a uniform `dt`, an integer `n` and `events` events, with the links given.
    fn newton_over(links: &[Stated4], role: DerivedRole, events: i64) -> Result<Derived, Refusal> {
        let real = WireDtype::Real;
        let mut primal = unit(&[
            ("x", real, &[]),
            ("dt", real, &[]),
            ("n", WireDtype::Int32, &[]),
        ]);
        primal.body.slots[1].role = SlotRole::Uniform;
        primal.finish = Some(crate::lower::unit_meta::Finish {
            n_events: events,
            ..Default::default()
        });
        let own = unit(&[("h", real, &[]), ("tau", real, &[]), ("w", real, &[])]);
        let facts = DerivedFacts {
            links: links
                .iter()
                .map(|(slot, what, form, of)| LinkFact {
                    slot: slot.to_string(),
                    what: *what,
                    form: *form,
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
    fn the_refine_links_and_the_runtime_slots_are_checked_by_name() {
        use LinkForm::{AdjointIn, Discard, Value};
        use LinkWhat::{Guard, PrimalSlot, Refined, Tau, TimeStep};
        let step = ("h", TimeStep, Value, Some("dt"));
        let ok = [step, ("tau", Tau, Value, None)];
        let done = newton(&ok, DerivedRole::RefineNewton).unwrap();
        assert_eq!(done.links[1].of_slot, None);
        let why = |links: &[Stated4], role| newton(links, role).unwrap_err();
        let needs = why(&[("tau", Tau, Value, None)], DerivedRole::RefineNewton);
        assert!(
            needs
                .reason
                .contains("a refine kernel needs a declared step"),
            "{}",
            needs.reason
        );
        let two = why(
            &[step, ("w", TimeStep, Value, Some("dt"))],
            DerivedRole::RefineNewton,
        );
        assert!(two.reason.contains("2 step slots"), "{}", two.reason);
        // a triple that is no row of the role names the kernel, the slot and the three names
        let row = why(&ok, DerivedRole::StepVjp);
        assert!(
            row.reason.contains(
                "derived kernel k_newton: links: the slot h is (StepVjp, TimeStep, Value)"
            ),
            "{}",
            row.reason
        );
        assert!(why(
            &[step, ("tau", Tau, Discard, None)],
            DerivedRole::RefineNewton
        )
        .reason
        .contains("(RefineNewton, Tau, Discard)"));
        let off = why(
            &[("h", TimeStep, Value, Some("x"))],
            DerivedRole::RefineNewton,
        );
        assert!(off.reason.contains("uniform or per-sample real"));
        let int = why(
            &[("h", TimeStep, Value, Some("n"))],
            DerivedRole::RefineNewton,
        );
        assert!(int.reason.contains("uniform or per-sample real"));
        let named = why(
            &[step, ("tau", Tau, Value, Some("x"))],
            DerivedRole::RefineNewton,
        );
        assert!(named.reason.contains("names no slot it is of"));
        let kinds = why(
            &[step, ("w", Refined, Value, Some("n"))],
            DerivedRole::RefineNewton,
        );
        assert!(kinds.reason.contains("differ in element type"));
        assert!(newton(
            &[step, ("w", Refined, Value, Some("x"))],
            DerivedRole::RefineNewton
        )
        .is_ok());
        // a guard link is of an event index below the primal's count; a primal without events has none to link
        let guard = |of: &'static str, events| {
            newton_over(
                &[step, ("w", Guard, Value, Some(of))],
                DerivedRole::RefineNewton,
                events,
            )
        };
        let none = guard("0", 0).unwrap_err();
        assert_eq!(none.kind, RefusalKind::BadRequest);
        assert!(
            none.reason.contains("derived kernel k_newton: links: the guard w is of event 0, and the primal has 0 events"),
            "{}",
            none.reason
        );
        let past = guard("2", 2).unwrap_err();
        assert!(past.reason.contains("event 2, and the primal has 2 events"));
        assert_eq!(guard("1", 2).unwrap_err().kind, RefusalKind::BadRequest);
        assert!(guard("0", 1).is_ok());
        // a refine-newton unit links each event once as a (Guard, Value): none missing, none twice
        let twice = newton_over(
            &[
                step,
                ("w", Guard, Value, Some("0")),
                ("tau", Guard, Value, Some("0")),
            ],
            DerivedRole::RefineNewton,
            1,
        )
        .unwrap_err();
        assert!(
            twice.reason.contains("event 0 of 1 is linked 2 times"),
            "{}",
            twice.reason
        );
        let missing = newton_over(&[step], DerivedRole::RefineNewton, 1).unwrap_err();
        assert!(
            missing.reason.contains("event 0 of 1 is linked 0 times"),
            "{}",
            missing.reason
        );
        let none = why(
            &[("w", PrimalSlot, Value, Some("x"))],
            DerivedRole::RefineEntry,
        );
        assert_eq!(none.kind, RefusalKind::NotWritten);
        assert!(none.reason.contains("RefineEntry"));
        let _ = AdjointIn;
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
                    what: PrimalSlot,
                    form: Value,
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
