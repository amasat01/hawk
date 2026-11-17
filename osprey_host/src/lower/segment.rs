//! The segmented split of a kernel: a kernel with one dispatch that splits by branch lowers to one unit per branch, each launched
//! over one run of the segmented input, and (for a selector the kernel never writes) a last unit that writes each sample's class.
//! The IR of the units comes from the wing analysis; this module turns each branch into a pending unit for the write step.
use super::request::{Body, DeclaredSlot, KernelRequest, Member};
use super::{canonical, finished_slot, walk_of, wing, Cards, Pending};
use aether_writers::contract::types::{Access, Dtype, Segment};
use aether_writers::refuse::{fault, not_written, Refusal};
use aether_writers::spell::binding_name;
use raptor_ir::record::Ctx;
use raptor_ir::value::{ttype_val, Val};
use raptor_wings::segment::{
    branch_sinks, class_sinks, group_name, offsets_name, segmented_dispatch, SegmentedDispatch,
};

/// The plane of the fused-steps word, which the access class does not read.
pub(super) const FUSED_STEPS_PLANE: &str = raptor_wings::cells::FUSED_STEPS_PLANE;

/// Whether the selector of the walk's dispatch number `index` is one no sink of the kernel writes (the walk's own listing).
fn selector_is_static(ctx: *mut Ctx, walk: u32, index: usize) -> Result<bool, Refusal> {
    walk_of(ctx, walk)?
        .static_selector
        .get(index)
        .copied()
        .ok_or_else(|| fault("a dispatch listing without the dispatch"))
}

/// The dispatch group a branch unit belongs to.
#[derive(Clone)]
pub(crate) struct Group {
    /// The group's name.
    pub name: String,
    /// The unit that writes the classes, for a selector the kernel never writes.
    pub classify: Option<String>,
    /// The group's number of branches.
    pub k: usize,
    /// The branch units in selector-value order; none on the record of the classify unit.
    pub branches: Vec<String>,
    /// The unit that sweeps the stale samples, for a selector the kernel writes.
    pub sweep: Option<String>,
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

/// The segmented dispatch of a body and whether its selector is one no sink of the kernel writes, or none when the body has no
/// segmented dispatch.
fn split_of(ctx: *mut Ctx, member: &Member) -> Result<Option<(SegmentedDispatch, bool)>, Refusal> {
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
    Ok(Some((split, fixed)))
}

/// The units a segmented kernel lowers to, or none when the kernel has no segmented dispatch.
fn branches(
    ctx: *mut Ctx,
    request: &KernelRequest,
    member: &Member,
) -> Result<Option<Vec<Pending>>, Refusal> {
    let Some((split, fixed)) = split_of(ctx, member)? else {
        return Ok(None);
    };
    if !fixed && !matches!(request.body, Body::Plain(_)) {
        return Err(not_written(format!(
            "kernel {:?}: the sweep unit of a segmented group with a dynamic selector in a chain or a fused kernel",
            request.name
        )));
    }
    let group = group_name(split.index);
    let offsets = offsets_name(&group);
    let classify = format!("{}__cls", request.name);
    let ours = Group {
        name: group.clone(),
        classify: fixed.then(|| classify.clone()),
        k: split.branches,
        branches: (0..split.branches)
            .map(|branch| format!("{}__k{branch}", request.name))
            .collect(),
        sweep: (!fixed).then(|| sweep_name(&request.name)),
    };
    let mut units = vec![];
    for branch in 0..split.branches {
        // SAFETY: the caller's context is not read elsewhere while the units are made.
        let made = branch_sinks(unsafe { &mut *ctx }, &member.sinks, &split, branch, !fixed);
        let segment = Segment {
            offsets_binding: binding_name("lookup", &offsets)?,
            j: branch as i64,
        };
        let name = ours.branches[branch].clone();
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
            exports_block: true,
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
            group: Some(Group {
                classify: None,
                branches: vec![],
                sweep: None,
                ..ours
            }),
            exports_block: true,
        });
    }
    Ok(Some(units))
}

/// The name of the sweep unit of the kernel `name`.
fn sweep_name(name: &str) -> String {
    format!("{name}__sweep")
}

/// The request of the sweep unit of a segmented group whose selector the kernel writes: the same kernel with its dispatch lowered
/// as a switch and its kind's guard an active set, which the lowering then writes like any such kernel. None for any other kernel.
pub(super) fn sweep_request(
    ctx: *mut Ctx,
    request: &KernelRequest,
) -> Result<Option<KernelRequest>, Refusal> {
    let Body::Plain(member) = &request.body else {
        return Ok(None);
    };
    let Some((split, false)) = split_of(ctx, member)? else {
        return Ok(None);
    };
    // SAFETY: the caller's context is not read elsewhere while the sweep's IR is made.
    let context = unsafe { &mut *ctx };
    let node = &context.nodes[split.node as usize];
    let operands = node.ops.clone();
    let tt = ttype_val(&node.dtype, &node.shape, node.tag.as_deref());
    let class = context
        .table
        .find("Dispatch")
        .ok_or_else(|| fault("no plain dispatch class"))?;
    let branches = operands[1..].iter().map(|&b| Val::Node(b)).collect();
    let given = [
        ("selector".to_string(), Val::Node(operands[0])),
        ("branches".to_string(), Val::Tuple(branches)),
        ("ttype".to_string(), tt),
    ];
    let switch = context.new_from(class, &given).map_err(wing)?;
    let swapped = [(split.node, switch)].into_iter().collect();
    let sinks = context.substitute(&member.sinks, &swapped).map_err(wing)?;
    let name = sweep_name(&request.name);
    let mut twin = request.clone();
    twin.name = name.clone();
    twin.one_step = None;
    twin.kind.active_set = true;
    twin.kind.runtime_set = true;
    twin.body = Body::Plain(Member {
        name,
        sinks,
        ..member.clone()
    });
    Ok(Some(twin))
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
