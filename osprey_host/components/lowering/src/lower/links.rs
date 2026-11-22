//! The links of a derived unit: each slot of the unit says what it refers to (`LinkWhat`) and in which form the unit holds it
//! (`LinkForm`). The table of valid (role, what, form) triples is one constant of data here; a stated link whose triple is not a
//! row is refused by name, and so is a link that breaks a rule of its `what` or a cardinality rule.
use super::derived::{Context, Stated};
use super::unit_meta::{DerivedRole, Link, LinkForm, LinkWhat, SlotRole, UnitMeta, WireDtype};
use aether_writers::refuse::{bad, Refusal};

/// One slot of the derived unit and what it refers to, by name.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkFact {
    /// The slot's name in the derived unit.
    pub slot: String,
    /// What it refers to.
    pub what: LinkWhat,
    /// The form the unit holds it in.
    pub form: LinkForm,
    /// The primal's slot it is of (a step plane names a slot of the taping twin); none where `what` has no slot.
    pub of: Option<String>,
    /// Whether a gradient is summed over samples.
    pub reduced: bool,
}

use LinkForm::{AdjointIn, AdjointOut, Discard, Dot, Grad, Seed, Value};
use LinkWhat::{Guard, PrimalSlot, Refined, Steps, TapeRow, Tau, Time0, TimeStep};

/// The valid links, one row per (role, what, form). A role with no row here is not written by the library.
pub const LINK_RULES: &[(DerivedRole, LinkWhat, LinkForm)] = &[
    (DerivedRole::Tape, PrimalSlot, Value),
    (DerivedRole::Tape, Time0, Value),
    (DerivedRole::TapeFused, PrimalSlot, Value),
    (DerivedRole::TapeFused, Time0, Value),
    (DerivedRole::StepVjp, PrimalSlot, Value),
    (DerivedRole::StepVjp, PrimalSlot, AdjointIn),
    (DerivedRole::StepVjp, PrimalSlot, AdjointOut),
    (DerivedRole::StepVjp, PrimalSlot, Grad),
    (DerivedRole::StepVjp, TapeRow, Value),
    (DerivedRole::StepVjp, Steps, Value),
    (DerivedRole::StepVjp, Time0, Grad),
    (DerivedRole::FusedVjp, PrimalSlot, Value),
    (DerivedRole::FusedVjp, PrimalSlot, AdjointIn),
    (DerivedRole::FusedVjp, PrimalSlot, AdjointOut),
    (DerivedRole::FusedVjp, PrimalSlot, Grad),
    (DerivedRole::FusedVjp, TapeRow, Value),
    (DerivedRole::FusedVjp, Steps, Value),
    (DerivedRole::FusedVjp, Time0, Grad),
    (DerivedRole::RefineNewton, PrimalSlot, Value),
    (DerivedRole::RefineNewton, TapeRow, Value),
    (DerivedRole::RefineNewton, Steps, Value),
    (DerivedRole::RefineNewton, Time0, Value),
    (DerivedRole::RefineNewton, TimeStep, Value),
    (DerivedRole::RefineNewton, Tau, Value),
    (DerivedRole::RefineNewton, Tau, Seed),
    (DerivedRole::RefineNewton, Guard, Value),
    (DerivedRole::RefineNewton, Guard, Dot),
    (DerivedRole::RefineNewton, Refined, Value),
    (DerivedRole::RefineNewton, Refined, Dot),
    (DerivedRole::StateGuardVjp, PrimalSlot, Value),
    (DerivedRole::StateGuardVjp, Refined, Value),
    (DerivedRole::StateGuardVjp, Guard, AdjointIn),
    (DerivedRole::StateGuardVjp, Refined, AdjointOut),
    (DerivedRole::StateGuardVjp, PrimalSlot, Grad),
    (DerivedRole::StateGuardVjp, PrimalSlot, Discard),
    (DerivedRole::RefineVjp, PrimalSlot, Value),
    (DerivedRole::RefineVjp, TapeRow, Value),
    (DerivedRole::RefineVjp, Steps, Value),
    (DerivedRole::RefineVjp, Time0, Value),
    (DerivedRole::RefineVjp, Time0, Grad),
    (DerivedRole::RefineVjp, TimeStep, Value),
    (DerivedRole::RefineVjp, TimeStep, Grad),
    (DerivedRole::RefineVjp, Tau, Value),
    (DerivedRole::RefineVjp, Tau, Discard),
    (DerivedRole::RefineVjp, Guard, AdjointIn),
    (DerivedRole::RefineVjp, Refined, AdjointIn),
    (DerivedRole::RefineVjp, PrimalSlot, AdjointOut),
    (DerivedRole::RefineVjp, PrimalSlot, Grad),
];

/// The refusal of a stated link: the kernel and the member are named.
fn refuse(kernel: &str, why: impl std::fmt::Display) -> Refusal {
    bad(format!("derived kernel {kernel}: links: {why}"))
}

/// The slot of a name in a unit.
fn slot_in(unit: &UnitMeta, name: &str) -> Option<usize> {
    unit.body.slots.iter().position(|s| s.name == name)
}

/// Whether the library holds any link row for the role.
pub fn role_has_rows(role: DerivedRole) -> bool {
    LINK_RULES.iter().any(|r| r.0 == role)
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

/// The checks of one link that do not look at its `of`: the runtime's slots, the table's rows, the guard, `reduced`.
fn check_triple(kernel: &Stated, link: &LinkFact, runtime_owned: bool) -> Result<(), Refusal> {
    let name = kernel.name.as_str();
    if runtime_owned {
        return Err(refuse(
            name,
            format!("{} is a slot the runtime owns (the finish counter or the active set) and takes no link", link.slot),
        ));
    }
    if !LINK_RULES.contains(&(kernel.role, link.what, link.form)) {
        return Err(refuse(
            name,
            format!(
                "the slot {} is ({:?}, {:?}, {:?}), which is not a link of that role",
                link.slot, kernel.role, link.what, link.form
            ),
        ));
    }
    if link.reduced && link.form != Grad {
        return Err(refuse(
            name,
            format!("{} is reduced but is not a gradient", link.slot),
        ));
    }
    Ok(())
}

/// The slot of the primal that `of` names.
fn primal_slot<'a>(
    name: &str,
    at: &'a Context,
    of: &str,
) -> Result<(usize, &'a super::unit_meta::Slot), Refusal> {
    let found = slot_in(at.primal, of)
        .ok_or_else(|| refuse(name, format!("the primal has no slot named {of}")))?;
    Ok((found, &at.primal.body.slots[found]))
}

/// The slot a link is of, checked against what the link refers to: a primal slot whose element type the linked slot has, an
/// integer step plane, the time plane's real type, or a time step of a uniform or per-sample primal slot.
fn of_slot(
    kernel: &Stated,
    own: &UnitMeta,
    at: &Context,
    (link, slot): (&LinkFact, usize),
) -> Result<Option<usize>, Refusal> {
    let name = kernel.name.as_str();
    let mine = own.body.slots[slot].dtype;
    match (link.what, &link.of) {
        (Tau, Some(_)) => Err(refuse(
            name,
            format!("{} is a tau and names no slot it is of", link.slot),
        )),
        (Tau, None) => real_only(name, link, mine).map(|()| None),
        (Guard, of) => event_index(kernel, at, link, of.as_deref()).map(Some),
        (Steps, of) => {
            if !matches!(
                mine,
                WireDtype::Int64 | WireDtype::Int32 | WireDtype::Uint32
            ) {
                return Err(refuse(
                    name,
                    format!("the step plane {} is not an integer", link.slot),
                ));
            }
            of.as_deref()
                .map(|of| {
                    at.twin
                        .and_then(|t| slot_in(t, of))
                        .ok_or_else(|| refuse(name, format!("no step slot {of} in a taping twin")))
                })
                .transpose()
        }
        (_, None) => Err(refuse(
            name,
            format!("{} names no slot it is of", link.slot),
        )),
        (what, Some(of)) => {
            let (found, primal) = primal_slot(name, at, of)?;
            match what {
                TimeStep => {
                    let fits = primal.role == SlotRole::Uniform
                        || (primal.role == SlotRole::PlaneIn && primal.shape.is_empty());
                    if !(fits && primal.dtype == WireDtype::Real) {
                        return Err(refuse(name, format!("the time step {} is of a uniform or per-sample real slot of the primal", link.slot)));
                    }
                    real_only(name, link, mine)?;
                }
                Time0 => {
                    if primal.dtype != WireDtype::Real {
                        return Err(refuse(name, format!("{of} is not a real time plane")));
                    }
                    real_only(name, link, mine)?;
                }
                _ if primal.dtype != mine => {
                    return Err(refuse(
                        name,
                        format!("{} and {of} differ in element type", link.slot),
                    ));
                }
                _ => {}
            }
            Ok(Some(found))
        }
    }
}

/// The events of the primal: the comparisons of its stop rule, as its finish record counts them.
fn primal_events(at: &Context) -> i64 {
    at.primal.finish.as_ref().map_or(0, |f| f.n_events)
}

/// The event a guard link is of: `of` is the index, as decimal text, below the primal's event count.
fn event_index(
    kernel: &Stated,
    at: &Context,
    link: &LinkFact,
    of: Option<&str>,
) -> Result<usize, Refusal> {
    let name = kernel.name.as_str();
    let count = primal_events(at);
    let index = of.and_then(|t| t.parse::<usize>().ok()).ok_or_else(|| {
        refuse(
            name,
            format!("the guard {} names an event index in `of`", link.slot),
        )
    })?;
    if (index as i64) >= count {
        return Err(refuse(
            name,
            format!(
                "the guard {} is of event {index}, and the primal has {count} events",
                link.slot
            ),
        ));
    }
    Ok(index)
}

/// A refine-newton unit's (Guard, Value) links cover the events 0 to n_events - 1 of the primal once each.
fn events_covered(kernel: &Stated, at: &Context, links: &[Link]) -> Result<(), Refusal> {
    let count = primal_events(at);
    let mut seen = vec![0usize; count.max(0) as usize];
    for link in links.iter().filter(|l| (l.what, l.form) == (Guard, Value)) {
        if let Some(n) = link.of_slot.and_then(|i| seen.get_mut(i)) {
            *n += 1;
        }
    }
    let wrong = seen.iter().position(|&n| n != 1);
    match wrong {
        None => Ok(()),
        Some(i) => Err(refuse(
            &kernel.name,
            format!(
                "event {i} of {count} is linked {} times as a (Guard, Value); a refine unit links each event once",
                seen[i]
            ),
        )),
    }
}

/// A link of the time plane's type is a real one.
fn real_only(name: &str, link: &LinkFact, dtype: WireDtype) -> Result<(), Refusal> {
    if dtype == WireDtype::Real {
        Ok(())
    } else {
        Err(refuse(
            name,
            format!("{} is not of the time plane's real type", link.slot),
        ))
    }
}

/// The slot links of the unit, checked against the role's rows and the cardinality rules.
pub(super) fn links_of(
    kernel: &Stated,
    own: &UnitMeta,
    at: &Context,
    time0_slot: Option<usize>,
    stated: &[LinkFact],
) -> Result<Vec<Link>, Refusal> {
    let name = kernel.name.as_str();
    let twin = matches!(kernel.role, DerivedRole::Tape | DerivedRole::TapeFused);
    let runtime_owned = runtime_slots(own);
    let (mut start_copies, mut time_steps) = (0, 0);
    let mut out = vec![];
    for link in stated {
        let slot = slot_in(own, &link.slot).unwrap_or(0);
        check_triple(kernel, link, runtime_owned.contains(&slot))?;
        if (link.what, link.form) == (Time0, Value) {
            start_copies += 1;
            if start_copies > 1 {
                return Err(refuse(name, "a unit has at most one (Time0, Value) link"));
            }
            if twin && Some(slot) != time0_slot {
                return Err(refuse(
                    name,
                    format!("{} is a start copy but is not the time0 slot", link.slot),
                ));
            }
        }
        time_steps += usize::from(link.what == TimeStep);
        out.push(Link {
            slot,
            what: link.what,
            of_slot: of_slot(kernel, own, at, (link, slot))?,
            reduced: link.reduced,
            form: link.form,
        });
    }
    if kernel.role == DerivedRole::RefineNewton && time_steps != 1 {
        return Err(refuse(
            name,
            if time_steps == 0 {
                "a refine kernel needs a declared step".to_string()
            } else {
                format!("a refine kernel has {time_steps} step slots, not one")
            },
        ));
    }
    if kernel.role == DerivedRole::RefineNewton {
        events_covered(kernel, at, &out)?;
    }
    if twin && start_copies != usize::from(time0_slot.is_some()) {
        return Err(bad(format!(
            "derived kernel {name}: time0: a start copy is stated exactly when the twin has one (Time0, Value) link"
        )));
    }
    Ok(out)
}
