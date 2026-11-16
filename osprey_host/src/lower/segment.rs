//! The segmented split of a kernel: a kernel with one dispatch that splits by branch lowers to one unit per branch, each launched
//! over one run of the segmented input, and (for a selector the kernel never writes) a last unit that writes each sample's class.
//! The IR of the units comes from the wing analysis; this module turns each branch into a pending unit for the write step.
use super::request::{DeclaredSlot, KernelRequest, Member};
use super::{canonical, finished_slot, walk_field, wing, Cards, Pending};
use aether_writers::contract::types::{Access, Dtype, Segment};
use aether_writers::refuse::{fault, Refusal};
use aether_writers::spell::binding_name;
use raptor_ir::record::Ctx;
use raptor_wings::segment::{
    branch_sinks, class_sinks, group_name, offsets_name, segmented_dispatch,
};
use rcore::keytext::{parse, Val};

/// The plane of the fused-steps word, which the access class does not read.
pub(super) const FUSED_STEPS_PLANE: &str = "fused_steps";

/// Whether the selector of the walk's dispatch number `index` is one no sink of the kernel writes (the walk's own listing).
fn selector_is_static(ctx: *mut Ctx, walk: u32, index: usize) -> Result<bool, Refusal> {
    let listing = parse(&walk_field(ctx, walk, "dispatches")?).map_err(fault)?;
    let (Val::Tuple(entries) | Val::List(entries)) = listing else {
        return Err(fault("the walk's dispatches are not a tuple"));
    };
    match entries.get(index) {
        Some(Val::Tuple(entry)) if matches!(entry.last(), Some(Val::Bool(_))) => {
            Ok(entry.last() == Some(&Val::Bool(true)))
        }
        _ => Err(fault("a dispatch listing without the dispatch")),
    }
}

/// The dispatch group a branch unit belongs to.
#[derive(Clone)]
pub(crate) struct Group {
    /// The group's name.
    pub name: String,
    /// The unit that writes the classes, for a selector the kernel never writes.
    pub classify: Option<String>,
}

/// The segmented split as a step: a kernel with a segmented dispatch becomes one pending unit per branch in order, and (for a
/// selector the kernel never writes) a last unit that writes the classes; any other unit passes on.
pub(super) fn split(
    ctx: *mut Ctx,
    request: &KernelRequest,
    _cards: &Cards,
    pending: Vec<Pending>,
) -> Result<Vec<Pending>, Refusal> {
    let mut units = vec![];
    for unit in pending {
        match branches(ctx, request, &unit.member)? {
            Some(branches) => units.extend(branches),
            None => units.push(unit),
        }
    }
    Ok(units)
}

/// The units a segmented kernel lowers to, or none when the kernel has no segmented dispatch.
fn branches(
    ctx: *mut Ctx,
    request: &KernelRequest,
    member: &Member,
) -> Result<Option<Vec<Pending>>, Refusal> {
    let bare = Member {
        declared: vec![],
        force: None,
        exempt: vec![FUSED_STEPS_PLANE.to_string()],
        ..member.clone()
    };
    let (walk, _) = canonical(ctx, &bare)?;
    // SAFETY: `ctx` is the live context the caller holds.
    let context = unsafe { &*ctx };
    let Some(split) = segmented_dispatch(context, walk).map_err(wing)? else {
        return Ok(None);
    };
    let fixed = selector_is_static(ctx, walk, split.index)?;
    let group = group_name(split.index);
    let offsets = offsets_name(&group);
    let classify = format!("{}__cls", request.name);
    let ours = Group {
        name: group.clone(),
        classify: fixed.then(|| classify.clone()),
    };
    let mut units = vec![];
    for branch in 0..split.branches {
        // SAFETY: the caller's context is not read elsewhere while the units are made.
        let made = branch_sinks(unsafe { &mut *ctx }, &member.sinks, &split, branch, !fixed);
        let segment = Segment {
            offsets_binding: binding_name("lookup", &offsets)?,
            j: branch as i64,
        };
        let name = format!("{}__k{branch}", request.name);
        units.push(Pending {
            member: unit_member(ctx, name, made.map_err(wing)?, Some((&offsets, !fixed))),
            segment: Some(segment),
            one_step: None,
            active: None,
            lanes: None,
            boxable: false,
            note: None,
            team: None,
            warp: None,
            group: Some(ours.clone()),
        });
    }
    if fixed {
        // SAFETY: the caller's context is not read elsewhere while the unit is made.
        let sinks = class_sinks(unsafe { &mut *ctx }, &split).map_err(wing)?;
        units.push(Pending {
            member: unit_member(ctx, classify, sinks, None),
            segment: None,
            one_step: None,
            active: None,
            lanes: None,
            boxable: false,
            note: None,
            team: None,
            warp: None,
            group: None,
        });
    }
    Ok(Some(units))
}

/// The body of one split unit: its walk declares the offsets plane for a branch unit (floored to the cross-sample access class
/// when the branch is guarded), and the finished-count plane for a finishing one.
fn unit_member(
    ctx: *mut Ctx,
    name: String,
    sinks: Vec<u32>,
    branch: Option<(&str, bool)>,
) -> Member {
    let mut member = Member {
        name,
        sinks,
        declared: branch
            .iter()
            .map(|(offsets, _)| DeclaredSlot::lookup(offsets, Dtype::I64))
            .collect(),
        force: branch
            .filter(|(_, guarded)| *guarded)
            .map(|_| Access::CrossSampleWrite),
        exempt: vec![FUSED_STEPS_PLANE.to_string()],
    };
    member.declared.extend(finished_slot(ctx, &member));
    member
}
