//! The typed meta of the units, from what the lowering already knows: each unit's walk, request and texts, and what the split into
//! stages decided for the kernel. A member nothing here knows is left absent (see `NOT_COMPUTED`), never given a value.
use super::active::{COUNT_PLANE, MAP_PLANE};
use super::card::{BuildNumbers, ContractionRoute};
use super::request::{Body, KernelRequest, LaneAxis, Steps, TargetText, UnitAnswer};
use super::segment::FUSED_STEPS_PLANE;
use super::stage_block::{KeyedRecord, PlaneRecord, StageBlock, TeamRecord};
use super::stages::cell_records;
use super::unit_meta as m;
use super::{walk_of, wing, Pending, FINISHED_PLANE};
use aether_writers::contract::types::{Access, Dtype, Role, Scalar, Target, Walk};
use aether_writers::refuse::{bad, fault, not_written, Refusal};
use raptor_ir::record::Ctx;
use raptor_wings::contraction::recognize;
use raptor_wings::cost::{dispatch_costs, BranchCost};
use raptor_wings::gemm::{gemm_plan, Extent, GemmPlan};
use raptor_wings::segment::{class_name, offsets_name, stale_name, strategies, Strategy};
use raptor_wings::stages::stored_cells;

/// What the meta of one unit is written from.
pub(super) struct Parts<'a> {
    /// The context that holds the kernel's IR.
    pub ctx: *mut Ctx,
    /// The request.
    pub request: &'a KernelRequest,
    /// The numbers the lowering decided on.
    pub card: &'a BuildNumbers,
    /// The command that makes the device image, when the request names the device.
    pub image: Option<&'a super::image::ImageCommand>,
    /// The form of the artifact the backend builds, when the request names a device.
    pub artifact: Option<&'a str>,
    /// The unit.
    pub pending: &'a Pending,
    /// The unit's canonical walk and its digest.
    pub walk: u32,
    /// The digest of that walk: 64 hex digits.
    pub digest: &'a str,
    /// The walk's arguments, access and finish.
    pub typed: &'a Walk,
    /// The unit's text per target.
    pub texts: &'a [TargetText],
    /// How the card runs a dense contraction of the request's scalar type on the device; `Gemm` without a device card.
    pub route: ContractionRoute,
}

/// The header's value of a count the runner chooses.
const AUTO_STEPS: i64 = -1;

/// The role of a slot in the header's closed set.
fn role(role: Role) -> m::SlotRole {
    match role {
        Role::Mutable => m::SlotRole::Mutable,
        Role::PlaneIn => m::SlotRole::PlaneIn,
        Role::Lookup => m::SlotRole::Lookup,
        Role::PlaneOut => m::SlotRole::PlaneOut,
        Role::Uniform => m::SlotRole::Uniform,
        Role::Terminated => m::SlotRole::Terminated,
        Role::WideIn => m::SlotRole::WideIn,
        Role::Nsamples => m::SlotRole::Nsamples,
        Role::Out => m::SlotRole::Out,
        Role::Extent => m::SlotRole::Extent,
    }
}

/// The type of a slot on the wire: the sample count is always the unsigned 32-bit the runtime passes, whatever the request states
/// (the text reads it as the runtime's index type either way). An index leaf is bound as the 64-bit `Int` by the entry's writer, a float is `Real`. A
/// by-value integer (a uniform) is bound as that `Int` whatever its width: no unit that has one narrows `Int`.
fn dtype(role: Role, dtype: Dtype) -> m::WireDtype {
    match dtype {
        _ if role == Role::Nsamples => m::WireDtype::Uint32,
        Dtype::Int32 | Dtype::Uint32 if role == Role::Uniform => m::WireDtype::Int64,
        Dtype::F64 | Dtype::F32 => m::WireDtype::Real,
        Dtype::I32 | Dtype::I64 => m::WireDtype::Int64,
        Dtype::Int32 => m::WireDtype::Int32,
        Dtype::Uint32 => m::WireDtype::Uint32,
        Dtype::Bool => m::WireDtype::Bool,
    }
}

/// The scalar type of a request.
pub(super) fn scalar(scalar: Scalar) -> m::ScalarType {
    match scalar {
        Scalar::Float64 => m::ScalarType::Float64,
        Scalar::Float32 => m::ScalarType::Float32,
        Scalar::Banded => m::ScalarType::Banded,
    }
}

/// The access class of a walk in the header's closed set.
pub(super) fn access(access: Access) -> m::Access {
    match access {
        Access::SampleLocal => m::Access::SampleLocal,
        Access::CrossSampleRead => m::Access::CrossSampleRead,
        Access::CrossSampleWrite => m::Access::CrossSampleWrite,
    }
}

/// The index of the slot of that name.
fn slot_at(slots: &[m::Slot], name: &str) -> Option<usize> {
    slots.iter().position(|slot| slot.name == name)
}

/// The slots of a walk: position is the index.
pub(super) fn slots(
    ctx: *mut Ctx,
    walk: u32,
    typed: &Walk,
    request: &KernelRequest,
) -> Result<Vec<m::Slot>, Refusal> {
    let shared = &request.shared;
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let gathered = raptor_wings::gather_mats(unsafe { &*ctx }, walk).map_err(wing)?;
    let patterns = patterns(request, typed)?;
    let prior = super::prior_reads(ctx, walk)?;
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let params = raptor_wings::param_cells(unsafe { &*ctx }, walk).map_err(wing)?;
    typed
        .slots
        .iter()
        .enumerate()
        .map(|(at, s)| {
            let planar = s.role == Role::Lookup && params.contains(&s.name);
            let shape = s.shape.clone();
            // the door types GATHER on a rank-2 per-sample plane and on a rank-0 LOOKUP
            let compact_matrix = s.role == Role::PlaneIn && s.shape.len() == 2;
            // a table is a broadcast table when it states axes and every one of them is a lane axis of the request (its label is
            // one of the lane space's): the caller then sizes it by its payload and it varies along those lane axes only. A table
            // with an axis of its own, or with none stated (one run-time axis of its own), is read at computed indices and stays
            // the caller's. The door types a gathered LOOKUP of rank 0 (a table of scalars) only, so a table with components
            // stays the caller's own table; a shared or parameter plane is never gathered.
            let scalar_table = s.role == Role::Lookup
                && !planar
                && s.axes.as_ref().is_some_and(|axes| {
                    !axes.is_empty()
                        && axes.iter().all(|axis| {
                            request
                                .lane_space
                                .iter()
                                .any(|lane| lane.label.as_deref() == Some(axis.label.as_str()))
                        })
                })
                && !shared.contains(&s.name)
                && payload_of_table(request, &s.name).is_some_and(|payload| payload.is_empty());
            let gather = gather_mask(
                compact_matrix,
                scalar_table,
                patterns,
                &s.name,
                gathered.contains(&s.name),
                request.lane_space.len(),
            );
            // the writers cross the boundary by the rank of the walk's slot; the record's shape is the caller's payload, which a
            // sink plane may state wider than the slot the entry binds
            let mirror =
                aether_writers::ir::runtime_names::mirror_of(s.role.name(), s.shape.len())?;
            Ok(m::Slot {
                flags: if planar { m::SLOT_PARAM_PLANE } else { 0 }
                    | if s.zeroed { m::SLOT_ZEROED } else { 0 }
                    | if gather.is_some() { m::SLOT_GATHER } else { 0 }
                    | if prior.contains(&s.name) {
                        m::SLOT_PRIOR
                    } else {
                        0
                    }
                    | if shared.contains(&s.name) {
                        m::SLOT_SHARED
                    } else {
                        0
                    },
                role: role(s.role),
                dtype: dtype(s.role, s.dtype),
                like_slot: like_slot(typed, at),
                gather,
                written: is_written(s.role, &s.name, typed.finish.as_deref()),
                shape,
                name: s.name.clone(),
                mirror: mirror.into(),
            })
        })
        .collect()
}

/// Whether the unit writes a slot of this role and name. A mask is only read unless the walk finishes samples through it (the
/// `finish` record names the mask it stores); every other output role is written by its declaration.
fn is_written(role: Role, name: &str, finish_mask: Option<&str>) -> bool {
    match role {
        Role::Terminated => finish_mask == Some(name),
        Role::Mutable | Role::Out | Role::PlaneOut => true,
        _ => false,
    }
}

/// The prefix of the name of the tangent of a plane.
const TANGENT_PREFIX: &str = "dot_";

/// The request's broadcast pattern, checked against the unit: each plane is one the unit binds (or the primal of a tangent plane it
/// binds) and states one flag for each lane axis.
fn patterns<'a>(
    request: &'a KernelRequest,
    typed: &Walk,
) -> Result<&'a [(String, Vec<bool>)], Refusal> {
    let axes = request.lane_space.len().max(1);
    for (plane, flags) in &request.broadcast {
        let tangent = format!("{TANGENT_PREFIX}{plane}");
        let bound = typed
            .slots
            .iter()
            .any(|s| s.name == *plane || s.name == tangent);
        if !bound || flags.len() != axes {
            return Err(bad(format!(
                "request key broadcast: the plane {plane:?} is not a plane of the kernel, or its flags are not one per lane axis ({axes})"
            )));
        }
    }
    Ok(&request.broadcast)
}

/// The lane axes a gathered plane varies along: the pattern of its own name, else of its primal plane (a tangent plane takes its
/// primal's); none when the request states none.
fn pattern_of(patterns: &[(String, Vec<bool>)], name: &str) -> Option<Vec<bool>> {
    let primal = name.strip_prefix(TANGENT_PREFIX).unwrap_or(name);
    patterns
        .iter()
        .find(|(plane, _)| plane == name)
        .or_else(|| patterns.iter().find(|(plane, _)| plane == primal))
        .map(|(_, flags)| flags.clone())
}

/// The payload the request states for the broadcast table `name` (the primal's for a tangent plane): the extents of its own leading
/// dimensions, empty for a table of scalars; none when the request states a pattern with no payload, or no pattern.
fn payload_of_table(request: &KernelRequest, name: &str) -> Option<Vec<i64>> {
    let primal = name.strip_prefix(TANGENT_PREFIX).unwrap_or(name);
    let stated = |plane: &str| {
        request
            .table_payloads
            .iter()
            .find(|(p, _)| p == plane)
            .map(|(_, payload)| payload.clone())
    };
    stated(name).or_else(|| stated(primal))
}

/// The keep mask of a gathered slot, one flag per lane axis, and none for a slot that is not gathered. A compact matrix (a rank-2
/// per-sample plane) is gathered when the request states a broadcast pattern for it, and also when the walk reads it only through a gather
/// with no pattern stated: that is the caller's full per-lane matrix, which varies along every axis. A table (a LOOKUP that is
/// neither shared nor a parameter plane) of scalars is gathered only by a stated pattern and a stated empty payload: with either
/// missing, or with components, it is read as one table for every lane. A gathered slot never states less.
fn gather_mask(
    compact_matrix: bool,
    scalar_table: bool,
    patterns: &[(String, Vec<bool>)],
    name: &str,
    read_through_a_gather: bool,
    lane_axes: usize,
) -> Option<Vec<bool>> {
    if compact_matrix {
        pattern_of(patterns, name).or_else(|| read_through_a_gather.then(|| vec![true; lane_axes]))
    } else if scalar_table {
        pattern_of(patterns, name)
    } else {
        None
    }
}

/// The prefix of the name of the accumulated gradient of a plane.
const ADJOINT_PREFIX: &str = "bar_";

/// The lookup or wide input slot a gradient slot is sized like: the accumulated output `bar_<plane>` of a plane the unit reads by
/// index, whatever the lane count.
fn like_slot(typed: &Walk, at: usize) -> Option<usize> {
    let slot = &typed.slots[at];
    let plane = slot.name.strip_prefix(ADJOINT_PREFIX)?;
    (slot.role == Role::PlaneOut && slot.zeroed).then_some(())?;
    typed
        .slots
        .iter()
        .position(|s| s.name == plane && matches!(s.role, Role::Lookup | Role::WideIn))
}

/// The thread box a unit's text of a target states.
fn thread_box(texts: &[TargetText], target: Target) -> Option<m::ThreadBox> {
    let held = texts
        .iter()
        .find(|t| t.target == target)?
        .thread_box
        .as_ref()?;
    Some(m::ThreadBox {
        axis: held.axis.unwrap_or(0),
        t: held.t,
        stride: held.stride,
    })
}

/// Whether the unit is a stage after the first of a cut plain kernel: only the first names the kernel's sinks.
fn later_stage(parts: &Parts) -> bool {
    matches!(parts.request.body, Body::Plain(_)) && parts.pending.member.name != parts.request.name
}

/// The declared axes of the planes the unit binds, each at its slot, as the plane states them. A runtime axis names the slot that
/// carries its extent.
pub(super) fn plane_axes(typed: &Walk, slots: &[m::Slot]) -> Vec<m::PlaneAxes> {
    let mut found: Vec<m::PlaneAxes> = vec![];
    for plane in typed.slots.iter().filter(|s| s.axes.is_some()) {
        let axes = plane.axes.as_deref().unwrap_or_default();
        let Some(slot) = slot_at(slots, &plane.name) else {
            continue;
        };
        found.push(m::PlaneAxes {
            slot,
            labels: axes.iter().map(|a| Some(a.label.clone())).collect(),
            extents: axes.iter().map(|a| a.extent.unwrap_or(-1)).collect(),
            extent_slots: axes
                .iter()
                .map(|a| match a.extent {
                    Some(_) => -1,
                    None => slot_at(
                        slots,
                        &aether_writers::unit::extent_name(&plane.name, &a.label),
                    )
                    .map_or(-1, |at| at as i64),
                })
                .collect(),
        });
    }
    found
}

/// The device threads of one block: the tile the warp step fixed (its lanes, one warp each, or the warps of a single lane), else
/// the request's launch bounds, else 0 for the runtime's choice. No recording states it: the old build's sidecars hold no block.
fn block_threads(parts: &Parts) -> i64 {
    match &parts.pending.warp {
        Some(warp) => warp.threads() * warp.warps,
        None => parts
            .request
            .settings
            .launch_bounds
            .map_or(0, |(threads, _)| threads),
    }
}

/// Marks the launch with the slots of the kernel's index set. A set of lanes the caller builds is the launch's `index_*` pair; the
/// set the runtime fills for a run-until-done runner is its `active_*` pair. A count alone marks a prefix set.
pub(super) fn mark_set(launch: &mut m::Launch, request: &KernelRequest, slots: &[m::Slot]) {
    let map =
        slot_at(slots, MAP_PLANE).filter(|_| request.kind.lanes.is_none_or(|lanes| !lanes.prefix));
    let count = slot_at(slots, COUNT_PLANE);
    if request.kind.lanes.is_some() {
        (launch.index_map_slot, launch.index_count_slot) = (map, count);
    } else {
        (launch.active_map_slot, launch.active_count_slot) = (map, count);
    }
}

/// The launch record: the group, the boxes and the slots of the index maps the unit runs over.
fn launch(parts: &Parts, slots: &[m::Slot]) -> m::Launch {
    // a later stage that runs on a team runs over its blocks, not over the caller's lanes: its space is one axis with
    // neither label nor extent
    let anonymous = [LaneAxis {
        label: None,
        extent: None,
    }];
    let teamed = later_stage(parts) && parts.pending.team.is_some();
    let space: &[LaneAxis] = if teamed {
        &anonymous
    } else {
        &parts.request.lane_space
    };
    let mut found = m::Launch {
        group: parts.request.group,
        host_box: thread_box(parts.texts, Target::Host),
        device_box: thread_box(parts.texts, Target::Device),
        block: block_threads(parts),
        lane_labels: space.iter().map(|a| a.label.clone()).collect(),
        lane_extents: space.iter().map(|a| a.extent.unwrap_or(-1)).collect(),
        plane_axes: if later_stage(parts) {
            vec![]
        } else {
            plane_axes(parts.typed, slots)
        },
        ..m::Launch::default()
    };
    if parts.pending.active.is_some() {
        mark_set(&mut found, parts.request, slots);
    }
    found
}

/// The finish record of a unit that finishes samples: its mask and its counter.
fn finish(parts: &Parts, slots: &[m::Slot]) -> Result<Option<m::Finish>, Refusal> {
    let Some(mask) = &parts.typed.finish else {
        return Ok(None);
    };
    let slot = |name: &str| {
        slot_at(slots, name)
            .ok_or_else(|| fault(format!("a finishing unit without the slot {name}")))
    };
    let (steps, steps_max) = match parts.request.steps {
        Some(Steps::Fixed(n)) => (n, Some(0)),
        Some(Steps::Auto { max }) => (AUTO_STEPS, Some(max)),
        None => {
            return Err(bad(
                "request key steps: a kernel that finishes samples needs it",
            ))
        }
    };
    let step = parts.texts.first().and_then(|t| t.facts.step.as_ref());
    let device = parts.texts.iter().find(|t| t.target == Target::Device);
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let n_events = super::events::count(
        unsafe { &*parts.ctx },
        &super::events::request_sinks(parts.request),
    );
    super::events::limit(&parts.pending.member.name, n_events)?;
    Ok(Some(m::Finish {
        flags: if step.is_some_and(|s| s.entry_counts) {
            m::FINISH_ENTRY_COUNTS
        } else {
            0
        },
        step_ops: step.map(|s| s.ops),
        fp64_ops_ir: match step {
            // SAFETY: `ctx` is the live context the caller holds; it is only read here.
            Some(_) => raptor_wings::cost::step_fp64_ops(unsafe { &*parts.ctx }, parts.walk)
                .map_err(wing)?,
            None => None,
        },
        // the device writer names its fast entries; the record copies the names
        device_persist_function: device.and_then(|t| t.facts.persist_function.clone()),
        device_range_function: device.and_then(|t| t.facts.range_function.clone()),
        mask_slot: slot(mask)?,
        counter_slot: slot(FINISHED_PLANE)?,
        n_events,
        host_steps_reverse: parts
            .request
            .targets
            .contains(&Target::Host)
            .then(|| parts.card.steps_reverse.clone())
            .flatten(),
        device_steps_reverse: parts
            .request
            .targets
            .contains(&Target::Device)
            .then(|| parts.card.device_steps_reverse.clone())
            .flatten(),
        steps,
        steps_max,
        steps_slot: slot_at(slots, FUSED_STEPS_PLANE).filter(|_| steps == AUTO_STEPS),
    }))
}

/// A unit reference by name, its digest filled once every unit of the answer is written.
fn by_name(name: &str) -> m::UnitRef {
    m::UnitRef {
        name: name.to_string(),
        digest: String::new(),
    }
}

/// The record of one branch's cost: the model's counts, its stores when it states them.
fn branch_cost(cost: &BranchCost) -> m::BranchCost {
    m::BranchCost {
        arith32: cost.arith32,
        arith64: cost.arith64,
        special32: cost.special32,
        special64: cost.special64,
        loads: cost.loads,
        gathers: cost.gathers,
        chain32: cost.chain32,
        chain64: cost.chain64,
        chain_special32: cost.chain_special32,
        chain_special64: cost.chain_special64,
        stores: cost.stores,
    }
}

/// The matrix products of a unit that is one dense contraction, whatever route the card chooses for it, or none. A banded unit
/// and a contraction whose shapes the strides do not state have none.
fn recognised_products(parts: &Parts, slots: &[m::Slot]) -> Result<Option<m::Gemm>, Refusal> {
    if parts.request.scalar == Scalar::Banded {
        return Ok(None);
    }
    // SAFETY: `ctx` is the live context the caller holds; the recogniser only reads it.
    let ctx = unsafe { &mut *parts.ctx };
    let Some(found) = recognize(ctx, &parts.pending.member.sinks, &[]).map_err(wing)? else {
        return Ok(None);
    };
    let Some(plan) = gemm_plan(&found) else {
        return Ok(None);
    };
    Ok(gemm_member(&plan, slots))
}

/// The record of a product over the unit's slots, by plane name; none when the unit lacks a slot of that name.
fn gemm_member(plan: &GemmPlan, slots: &[m::Slot]) -> Option<m::Gemm> {
    let call = &plan.forward;
    let (Some(a), Some(b), Some(c)) = (
        slot_at(slots, &call.a),
        slot_at(slots, &call.b),
        slot_at(slots, &call.c),
    ) else {
        return None;
    };
    let number = |extent: Extent| match extent {
        Extent::Fixed(n) => n,
        Extent::PerLane => m::GEMM_PER_LANE,
    };
    let [(a_rows, a_cols), (b_rows, b_cols), (c_rows, c_cols)] =
        call.shapes.map(|(rows, cols)| (number(rows), number(cols)));
    let bit = |on: bool, flag: u32| if on { flag } else { 0 };
    Some(m::Gemm {
        lane_fanin: plan.lane_fanin,
        calls: vec![m::GemmCall {
            flags: bit(call.trans_a, m::GEMM_TRANS_A)
                | bit(call.trans_b, m::GEMM_TRANS_B)
                | bit(call.trans_c, m::GEMM_TRANS_C)
                | bit(call.accumulate, m::GEMM_ACCUMULATE),
            a_slot: a,
            b_slot: b,
            c_slot: c,
            a_rows,
            a_cols,
            b_rows,
            b_cols,
            c_rows,
            c_cols,
        }],
    })
}

/// The cost of each branch of each dispatch group of a walk, in the walk's dispatch listing order (an auto group only: the runtime
/// prices its entry choice from them; any other group states none).
fn group_costs(
    ctx: &Ctx,
    order: &[u32],
    found: &[(String, Strategy)],
) -> Result<Vec<Vec<m::BranchCost>>, Refusal> {
    // the model prices every group of the walk, in the same listing the strategies come in
    let priced = if found
        .iter()
        .any(|(_, strategy)| *strategy == Strategy::Auto)
    {
        dispatch_costs(ctx, order).map_err(wing)?
    } else {
        vec![]
    };
    Ok(found
        .iter()
        .enumerate()
        .map(|(at, (_, strategy))| match (strategy, priced.get(at)) {
            (Strategy::Auto, Some(group)) => group.branches.iter().map(branch_cost).collect(),
            _ => vec![],
        })
        .collect())
}

/// The dispatch groups of a unit's walk: each group's name, strategy and branches, and for an auto group the slot of its
/// branch-mix plane.
fn dispatches(parts: &Parts, slots: &[m::Slot]) -> Result<Vec<m::Dispatch>, Refusal> {
    if let Some(group) = &parts.pending.group {
        let at = |name: String| slot_at(slots, &name);
        // the classify unit's record states the class plane it writes and nothing else
        let lists = !group.branches.is_empty();
        return Ok(vec![m::Dispatch {
            strategy: m::DispatchStrategy::Segmented,
            class_slot: at(class_name(&group.name)),
            stale_slot: at(stale_name(&group.name)).filter(|_| lists),
            offsets_slot: at(offsets_name(&group.name)).filter(|_| lists),
            classify: group.classify.as_deref().map(by_name),
            name: group.name.clone(),
            k: group.k,
            branches: group.branches.iter().map(|name| by_name(name)).collect(),
            sweep: group.sweep.as_deref().map(by_name),
            ..m::Dispatch::default()
        }]);
    }
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let ctx = unsafe { &*parts.ctx };
    let found = strategies(ctx, parts.walk).map_err(wing)?;
    let order = &walk_of(parts.ctx, parts.walk)?.order;
    let nodes = ctx.dispatch_nodes(order).map_err(wing)?;
    let costs = group_costs(ctx, order, &found)?;
    found
        .into_iter()
        .zip(nodes)
        .zip(costs)
        .map(|(((name, strategy), node), branch_costs)| {
            let mix_slot = match strategy {
                Strategy::Auto => {
                    let plane = parts
                        .texts
                        .first()
                        .and_then(|t| t.facts.mix_planes.iter().find(|(group, _)| *group == name));
                    let held = plane.and_then(|(_, plane)| slot_at(slots, plane));
                    Some(held.ok_or_else(|| {
                        fault(format!("the branch-mix plane of the group {name}"))
                    })?)
                }
                _ => None,
            };
            let set = parts.pending.auto_set.as_ref().filter(|g| g.name == name);
            Ok(m::Dispatch {
                strategy: match strategy {
                    Strategy::Predicated => m::DispatchStrategy::Predicated,
                    Strategy::Switch => m::DispatchStrategy::Switch,
                    Strategy::Segmented => m::DispatchStrategy::Segmented,
                    Strategy::Auto => m::DispatchStrategy::Auto,
                },
                mix_slot,
                branch_costs,
                // the segmented set of an automatic group, when the request asked for it: the refs of the fused unit's record
                classify: set.and_then(|g| g.classify.as_deref().map(by_name)),
                branches: set.map_or(vec![], |g| g.branches.iter().map(|b| by_name(b)).collect()),
                name,
                k: ctx.nodes[node as usize].ops.len() - 1,
                ..m::Dispatch::default()
            })
        })
        .collect()
}

/// Fills the digest of each unit a dispatch group names (its classify unit, its branch units and its sweep unit), once every unit
/// of the answer is written.
pub(super) fn unit_digests(units: &mut [UnitAnswer]) {
    let digests: Vec<(String, String)> = units
        .iter()
        .map(|u| (u.meta.name.clone(), u.meta.digest.clone()))
        .collect();
    for group in units.iter_mut().flat_map(|u| &mut u.meta.body.dispatches) {
        let named = group
            .classify
            .iter_mut()
            .chain(group.branches.iter_mut())
            .chain(group.sweep.iter_mut());
        for unit in named {
            if let Some((_, digest)) = digests.iter().find(|(name, _)| *name == unit.name) {
                unit.digest = digest.clone();
            }
        }
    }
}

/// The meta of one unit, without what only its kernel's stages tell (see `stages`), and the contraction the unit is when it is
/// one dense contraction: its products and the route word the card chose. The meta keeps the products only when the route runs
/// them as matrix products; the unit keeps its full kernel in either case, so a host plan runs the loop and a device plan the
/// products.
pub(super) fn unit(parts: &Parts) -> Result<(m::UnitMeta, Option<m::Contraction>), Refusal> {
    let slots = slots(parts.ctx, parts.walk, parts.typed, parts.request)?;
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let stored = stored_cells(unsafe { &*parts.ctx }, &parts.pending.member.sinks).map_err(wing)?;
    let stores = !stored.is_empty();
    // what the writer decided is the same in every text of the unit
    let facts = &parts
        .texts
        .first()
        .ok_or_else(|| fault("a unit without a text"))?
        .facts;
    let rows = &facts.row_planes;
    let cells = cell_records(stored, &parts.request.planes)
        .iter()
        .map(|cell| {
            let mine: Vec<usize> = rows
                .iter()
                .filter(|r| r.cell == cell.name)
                .map(|r| r.slot)
                .collect();
            Ok(m::Cell {
                op: reduce_op(&cell.op)?,
                internal: cell.internal,
                columns: cell.columns,
                name: cell.name.clone(),
                slots: Some(mine).filter(|s| !s.is_empty()),
                comp_slots: None,
                position: cell.op.starts_with("arg"),
            })
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    let name = &parts.pending.member.name;
    let entry = |target: Target| parts.texts.iter().find(|t| t.target == target);
    let device = entry(Target::Device).map(|t| t.entry.clone());
    let image = parts.image.filter(|_| device.is_some());
    let device_format = device
        .as_ref()
        .and(parts.artifact)
        .map_or_else(m::ArtifactForm::default, |form| {
            m::ArtifactForm(form.to_string())
        });
    let host = entry(Target::Host);
    let device_text = entry(Target::Device);
    // the symbol that folds a companion plane, as the target's text states it
    let fold_of = |text: Option<&TargetText>, plane: &String| {
        text.and_then(|t| t.facts.companions.iter().find(|h| h.plane == *plane))
            .map(|h| h.fold_entry.clone())
    };
    let dispatches = dispatches(parts, &slots)?;
    let products = recognised_products(parts, &slots)?;
    let gemm = products
        .clone()
        .filter(|_| parts.route == ContractionRoute::Gemm);
    let contraction = products.map(|products| m::Contraction {
        route: parts.route.word().into(),
        products,
    });
    let companions = facts
        .companions
        .iter()
        .map(|c| {
            let (comp, target) = (&c.plane, &c.target);
            let at = |plane: &str| {
                slot_at(&slots, plane).ok_or_else(|| {
                    // SAFETY: `parts.ctx` is the live context the caller holds; it is only read here.
                    let ctx = unsafe { &*parts.ctx };
                    if super::group::group_of(ctx, &parts.pending.member.sinks).is_some() {
                        return not_written(format!(
                            "{:?}: a compensated keyed add inside a group is not written yet: the kernel declares no companion plane {plane} for it; state the add with the plain contract",
                            parts.pending.member.name
                        ));
                    }
                    fault(format!("a companion pair over the plane {plane}"))
                })
            };
            Ok(m::Companion {
                slot: at(comp)?,
                target_slot: at(target)?,
                host_fold_entry: fold_of(host, comp),
                device_fold_function: fold_of(device_text, comp),
            })
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    let counts = super::walk_of(parts.ctx, parts.walk)?
        .counts
        .iter()
        .map(|(count, of)| {
            let at = |plane: &str| {
                slot_at(&slots, plane)
                    .ok_or_else(|| fault(format!("a count over the plane {plane}")))
            };
            Ok(m::Counted {
                slot: at(count)?,
                of_slot: at(of)?,
            })
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    let winners = super::walk_of(parts.ctx, parts.walk)?
        .winners
        .iter()
        .map(|(winner, of)| {
            let at = |plane: &str| {
                slot_at(&slots, plane)
                    .ok_or_else(|| fault(format!("a winner plane over the plane {plane}")))
            };
            Ok(m::Ranked {
                slot: at(winner)?,
                of_slot: at(of)?,
            })
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let made =
        raptor_wings::cells::walk_companions(unsafe { &*parts.ctx }, parts.walk).map_err(wing)?;
    let cells = compensated(cells, &companions, &slots, &made)?;
    let meta = m::UnitMeta {
        flags: if parts.request.cards.is_empty() {
            0
        } else {
            m::UNIT_CALLER_CARDS
        },
        name: name.clone(),
        digest: parts.digest.to_string(),
        scalar_type: scalar(parts.request.scalar),
        host_entry: entry(Target::Host).map(|t| t.facts.symbol.clone()),
        device_image: device.as_ref().map(|_| m::image_symbols(name).0),
        device_image_size: device.as_ref().map(|_| m::image_symbols(name).1),
        device_image_format: device_format,
        device_compiler: image.map(|i| i.compiler.clone()),
        device_flags: image.map(|i| i.flags.clone()),
        device_ptx_isa: image.map_or(0, |i| i.isa_version),
        gemm,
        group: super::group::record(parts)?,
        device_function: device,
        launch: launch(parts, &slots),
        finish: finish(parts, &slots)?,
        body: m::Body {
            access: access(parts.typed.access),
            cell_row: if stores { parts.card.cells_row } else { 0 },
            cells,
            companions,
            counts,
            winners,
            dispatches,
            slots,
            ..m::Body::default()
        },
        ..m::UnitMeta::default()
    };
    Ok((meta, contraction))
}

/// The cells with the compensation slots of their sums: each row plane of a sum cell names the companion slot that compensates it,
/// a companion pair's or else the compensation plane the walk made for that row (`made`, the walk's (row, plane) pairs), never a
/// slot found by its name; a cell none of whose rows has one keeps none, one with only some of them is a fault.
fn compensated(
    cells: Vec<m::Cell>,
    companions: &[m::Companion],
    slots: &[m::Slot],
    made: &[(String, String)],
) -> Result<Vec<m::Cell>, Refusal> {
    cells
        .into_iter()
        .map(|mut cell| {
            let rows = cell.slots.clone().unwrap_or_default();
            let found: Vec<usize> = rows
                .iter()
                .filter_map(|&row| {
                    companions
                        .iter()
                        .find(|c| c.target_slot == row)
                        .map(|c| c.slot)
                        .or_else(|| {
                            made.iter()
                                .find(|(sum, _)| *sum == slots[row].name)
                                .and_then(|(_, plane)| slot_at(slots, plane))
                        })
                })
                .collect();
            if !found.is_empty() {
                if found.len() != rows.len() {
                    return Err(fault(format!(
                        "the cell {} has a companion for {} of its {} rows",
                        cell.name,
                        found.len(),
                        rows.len()
                    )));
                }
                cell.comp_slots = Some(found);
            }
            Ok(cell)
        })
        .collect()
}

/// The index of a keyed plane in the table, which holds each plane once by name.
fn intern(table: &mut Vec<m::Keyed>, plane: &KeyedRecord) -> usize {
    if let Some(at) = table.iter().position(|k| k.name == plane.name) {
        return at;
    }
    table.push(m::Keyed {
        width: plane.width,
        axis_labels: plane.axes.iter().map(|(label, _)| label.clone()).collect(),
        axis_extents: plane
            .axes
            .iter()
            .map(|(_, extent)| extent.unwrap_or(-1))
            .collect(),
        payload: plane.payload.clone(),
        name: plane.name.clone(),
    });
    table.len() - 1
}

/// The reduction of a cell as the header's closed set.
fn reduce_op(op: &str) -> Result<m::ReduceOp, Refusal> {
    match op {
        "sum" => Ok(m::ReduceOp::Sum),
        "max" | "argmax" => Ok(m::ReduceOp::Max),
        "min" | "argmin" => Ok(m::ReduceOp::Min),
        "all" | "land" => Ok(m::ReduceOp::All),
        other => Err(not_written(format!("a cell folded by {other}"))),
    }
}

/// The element type of a team plane as the header's closed set.
fn plane_dtype(name: &str) -> Result<m::WireDtype, Refusal> {
    match name {
        "real" => Ok(m::WireDtype::Real),
        "int32" => Ok(m::WireDtype::Int32),
        "band_partial" => Ok(m::WireDtype::BandPartial),
        other => Err(not_written(format!("a team plane of type {other}"))),
    }
}

/// The slots a kernel's scratch planes and scatters name: those of the unit that lists the stages, then those of each stage unit.
struct View<'a> {
    /// The slots of the unit that lists the stages.
    carrier: &'a [m::Slot],
    /// The slots of each stage unit, in stage order.
    stages: &'a [Vec<m::Slot>],
}

impl View<'_> {
    /// The slot of that name: its index and, when it is not the lister's, the index of the stage whose unit holds it.
    fn find(&self, name: &str) -> Option<(usize, Option<usize>)> {
        if let Some(at) = slot_at(self.carrier, name) {
            return Some((at, None));
        }
        self.stages
            .iter()
            .enumerate()
            .find_map(|(stage, slots)| slot_at(slots, name).map(|at| (at, Some(stage))))
    }
}

/// A scratch plane of the stages as a record, its keyed plane and target slot looked up.
fn team_plane(
    plane: &PlaneRecord,
    keyed: &mut Vec<m::Keyed>,
    view: &View,
) -> Result<m::TeamPlane, Refusal> {
    let unit = match (&plane.keys, &plane.per_target, plane.once) {
        (Some(_), _, _) => m::PlaneUnit::Key,
        (_, Some(_), _) => m::PlaneUnit::Target,
        (_, _, true) => m::PlaneUnit::Once,
        _ => m::PlaneUnit::Lane,
    };
    let target = plane.per_target.as_deref().and_then(|name| view.find(name));
    Ok(m::TeamPlane {
        dtype: plane_dtype(&plane.dtype)?,
        unit,
        layout: match plane.layout.as_deref() {
            Some("plane_major") => m::PlaneLayout::PlaneMajor,
            _ => m::PlaneLayout::Flat,
        },
        keyed: plane.keys.as_ref().map(|k| intern(keyed, k)),
        target_slot: target.map(|(at, _)| at),
        target_stage: target.and_then(|(_, stage)| stage),
        per_lane: plane.per_lane,
        fill: plane.fill.unwrap_or(0),
        fill_promised: plane.fill.is_some(),
        // Every scratch plane the placement makes is tiled by warp: the host text keeps its tile on the stack and never
        // touches the plane, the device text writes a whole warp.
        wide: if plane.wide {
            m::WideLayout::WarpTiled
        } else {
            m::WideLayout::None
        },
        name: plane.name.clone(),
    })
}

/// The team of a stage as a record; the index of its keyed plane, scatter target and status plane are looked up in the lister.
fn team(
    team: &TeamRecord,
    keyed: &mut Vec<m::Keyed>,
    carrier: (&View, &[m::TeamPlane]),
) -> m::Team {
    let (grid, of) = match (&team.keys, &team.scatter) {
        (Some(_), _) => (m::TeamGrid::Keys, None),
        (_, Some(s)) if s.grid == "chunks" => (m::TeamGrid::Chunks, Some(s)),
        (_, Some(s)) => (m::TeamGrid::Targets, Some(s)),
        _ => (m::TeamGrid::Lanes, None),
    };
    let target = of.and_then(|s| carrier.0.find(&s.target));
    m::Team {
        grid,
        threads: team.threads,
        warps: team.warps,
        lanes: team.lanes,
        groups: team.keys.as_ref().map_or(1, |k| k.groups),
        keyed: team.keys.as_ref().map(|k| intern(keyed, &k.keyed)),
        of_slot: target.map(|(at, _)| at),
        of_stage: target.and_then(|(_, stage)| stage),
        status_plane: of
            .and_then(|s| s.status.as_ref())
            .and_then(|name| carrier.1.iter().position(|p| &p.name == name)),
    }
}

/// The output of a stage that runs one block per key is one value per kept key, so the unit that lists the stages states the kept
/// axes of its keyed plane for the wide output the stage writes: the plane's extent is the keys' extents, as a declared sink states
/// its own. The runtime reads the keys of a wide output from the stage that writes it through the listing unit's slot of that
/// name, so the axes belong on that slot and on no stage unit. An output the request already declared axes for keeps them.
fn state_keyed_outputs(units: &mut [UnitAnswer], listing: usize, block: &StageBlock) {
    for stage in &block.stages {
        let Some(keys) = stage
            .team
            .as_ref()
            .and_then(|t| t.keys.as_ref())
            .filter(|k| !k.keyed.axes.is_empty())
        else {
            // a sum over every lane keeps no axis: one key, an output with no named axes
            continue;
        };
        let written: Vec<String> = units
            .iter()
            .find(|u| u.meta.name == stage.kernel)
            .map(|u| {
                let slots = &u.meta.body.slots;
                let outputs = slots.iter().filter(|s| {
                    // a scratch plane the runtime owns is no output of the kernel
                    s.role == m::SlotRole::PlaneOut
                        && s.flags & m::SLOT_ZEROED == 0
                        && !block.team_planes.iter().any(|p| p.name == s.name)
                });
                outputs.map(|s| s.name.clone()).collect()
            })
            .unwrap_or_default();
        let meta = &mut units[listing].meta;
        for name in written {
            let at = slot_at(&meta.body.slots, &name);
            let wide = at.is_some_and(|at| {
                let slot = &meta.body.slots[at];
                slot.role == m::SlotRole::PlaneOut && slot.flags & m::SLOT_ZEROED == 0
            });
            let stated = at.is_some_and(|at| meta.launch.plane_axes.iter().any(|p| p.slot == at));
            if let (Some(at), true, false) = (at, wide, stated) {
                meta.launch.plane_axes.push(m::PlaneAxes {
                    slot: at,
                    labels: keys.keyed.axes.iter().map(|(l, _)| l.clone()).collect(),
                    extents: keys
                        .keyed
                        .axes
                        .iter()
                        .map(|(_, e)| e.unwrap_or(-1))
                        .collect(),
                    extent_slots: vec![-1; keys.keyed.axes.len()],
                });
            }
        }
    }
}

/// Writes what the split into stages decided into the unit that lists the stages: the kernel's entry-less source unit when the
/// kernel was cut (put first in the answer), else the first stage. It gets the stages, cells, keyed planes and scratch planes of
/// the kernel, and the chunk cap of its ordered scatters.
pub(super) fn stages(
    units: &mut Vec<UnitAnswer>,
    block: &StageBlock,
    scatter_chunks: i64,
) -> Result<(), Refusal> {
    let first = block.stages.first().map(|s| s.kernel.as_str());
    let refs: Vec<(String, String)> = units
        .iter()
        .map(|u| (u.meta.name.clone(), u.meta.digest.clone()))
        .collect();
    let stage_slots: Vec<Vec<m::Slot>> = block
        .stages
        .iter()
        .map(|s| {
            units
                .iter()
                .find(|u| u.meta.name == s.kernel)
                .map(|u| u.meta.body.slots.clone())
                .ok_or_else(|| {
                    fault(format!(
                        "a stage {} that is no unit of the answer",
                        s.kernel
                    ))
                })
        })
        .collect::<Result<_, _>>()?;
    let at = match &block.source {
        Some(source) => {
            units.insert(
                0,
                UnitAnswer {
                    texts: vec![],
                    meta: source.clone(),
                    defines: vec![],
                    contraction: None,
                },
            );
            0
        }
        None => units
            .iter()
            .position(|u| Some(u.meta.name.as_str()) == first)
            .ok_or_else(|| fault("a stage block whose first stage is not a unit of the answer"))?,
    };
    let meta = &mut units[at].meta;
    let mut keyed = vec![];
    for plane in &block.keyed {
        intern(&mut keyed, plane);
    }
    let carrier = meta.body.slots.clone();
    let view = View {
        carrier: &carrier,
        stages: &stage_slots,
    };
    let mut planes = vec![];
    for plane in &block.team_planes {
        planes.push(team_plane(plane, &mut keyed, &view)?);
    }
    for stage in &block.stages {
        let digest = refs
            .iter()
            .find(|(name, _)| *name == stage.kernel)
            .map(|(_, digest)| digest.clone())
            .ok_or_else(|| {
                fault(format!(
                    "a stage {} that is no unit of the answer",
                    stage.kernel
                ))
            })?;
        meta.stages.push(m::Stage {
            unit: m::UnitRef {
                name: stage.kernel.clone(),
                digest,
            },
            cells_in: stage.cells_in.clone(),
            cells_out: stage.cells_out.clone(),
            team: stage
                .team
                .as_ref()
                .map(|t| team(t, &mut keyed, (&view, &planes))),
        });
    }
    meta.body.cells.clear();
    for cell in &block.cells {
        meta.body.cells.push(m::Cell {
            op: reduce_op(&cell.op)?,
            internal: cell.internal,
            columns: cell.columns,
            name: cell.name.clone(),
            slots: None,
            comp_slots: None,
            position: cell.op.starts_with("arg"),
        });
    }
    meta.body.keyed = keyed;
    // The chunk cap belongs to an ordered scatter alone: the unit states it when it has a plane sliced by
    // the scatter's target or a stage that runs on the chunk grid, and states none (0) otherwise.
    let scatters = planes.iter().any(|p| p.unit == m::PlaneUnit::Target)
        || meta.stages.iter().any(|s| {
            s.team
                .as_ref()
                .is_some_and(|t| t.grid == m::TeamGrid::Chunks)
        });
    meta.body.scatter_chunks = if scatters { scatter_chunks } else { 0 };
    meta.body.team_planes = planes;
    state_keyed_outputs(units, at, block);
    // a scratch plane is bound by the stage units under its own name: those slots hold the plane's element type
    for unit in units.iter_mut() {
        for slot in &mut unit.meta.body.slots {
            let plane = block
                .team_planes
                .iter()
                .find(|p| p.name == slot.name)
                .map(|p| plane_dtype(&p.dtype))
                .transpose()?;
            slot.dtype = plane.unwrap_or(slot.dtype);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! A compensation plane is the one the walk made for the row, never one a name suggests.
    use super::*;
    use raptor_ir::door::{rx_ctx_free, rx_ctx_new, RX_OK};
    use raptor_ir::dump::rx_load;

    /// A slot of the given name.
    fn slot(name: &str) -> m::Slot {
        m::Slot {
            name: name.into(),
            ..m::Slot::default()
        }
    }

    /// A sum cell over the slot 0.
    fn sum() -> Vec<m::Cell> {
        vec![m::Cell {
            op: m::ReduceOp::Sum,
            columns: 1,
            name: "c".into(),
            slots: Some(vec![0]),
            ..m::Cell::default()
        }]
    }

    #[test]
    fn a_mask_is_written_only_when_the_walk_finishes_samples_through_it() {
        // an active-set kernel that never stores to its mask reads it; one that finishes samples through it stores to it
        assert!(!is_written(Role::Terminated, "terminated", None));
        assert!(!is_written(Role::Terminated, "terminated", Some("other")));
        assert!(is_written(
            Role::Terminated,
            "terminated",
            Some("terminated")
        ));
        assert!(is_written(Role::Mutable, "y", None));
        assert!(!is_written(Role::PlaneIn, "x", None));
    }

    #[test]
    fn the_unit_listing_a_keys_stage_states_the_kept_axes_of_the_wide_output_the_stage_writes() {
        use super::super::stage_block::{KeyedRecord, KeysGrid, StageRecord, TeamRecord};
        let written = |name: &str| m::Slot {
            name: name.into(),
            written: true,
            role: m::SlotRole::PlaneOut,
            ..m::Slot::default()
        };
        let unit = |name: &str, slots: Vec<m::Slot>| UnitAnswer {
            texts: vec![],
            meta: m::UnitMeta {
                name: name.into(),
                body: m::Body {
                    slots,
                    ..m::Body::default()
                },
                ..m::UnitMeta::default()
            },
            defines: vec![],
            contraction: None,
        };
        let keyed = KeyedRecord {
            name: "tmb0_gp".into(),
            width: 1,
            axes: vec![(Some("a".into()), Some(2)), (Some("b".into()), None)],
            payload: vec![],
        };
        let team = |keys: bool| TeamRecord {
            threads: 128,
            lanes: 1,
            warps: 1,
            keys: keys.then(|| KeysGrid {
                groups: 1,
                keyed: keyed.clone(),
            }),
            scatter: None,
        };
        let stage = |kernel: &str, keys: bool| StageRecord {
            kernel: kernel.into(),
            cells_in: vec![],
            cells_out: vec![],
            team: Some(team(keys)),
        };
        let scratch = crate::lower::stage_block::PlaneRecord {
            name: "tmb0_gp".into(),
            dtype: "real".into(),
            per_lane: 64,
            layout: None,
            keys: Some(keyed.clone()),
            per_target: None,
            once: false,
            wide: false,
            fill: None,
        };
        let mut units = vec![
            unit("k", vec![slot("x"), written("out"), written("tmb0_gp")]),
            unit("k_2", vec![written("out"), written("tmb0_gp")]),
        ];
        let block = StageBlock {
            stages: vec![stage("k", true), stage("k_2", true)],
            team_planes: vec![scratch],
            ..StageBlock::default()
        };
        state_keyed_outputs(&mut units, 0, &block);
        // the scratch plane and the stage units carry none; the listing unit's wide output carries the keys' axes
        assert!(units[1].meta.launch.plane_axes.is_empty());
        let [axes] = &units[0].meta.launch.plane_axes[..] else {
            panic!("one output: {:?}", units[0].meta.launch.plane_axes);
        };
        assert_eq!(units[0].meta.body.slots[axes.slot].name, "out");
        assert_eq!(axes.labels, [Some("a".into()), Some("b".into())]);
        assert_eq!(axes.extents, [2, -1]);
        // a sum over every lane keeps no axis, and the door refuses named axes of none
        let mut whole = block.clone();
        for stage in &mut whole.stages {
            if let Some(grid) = stage.team.as_mut().and_then(|t| t.keys.as_mut()) {
                grid.keyed.axes.clear();
            }
        }
        let mut units = vec![unit("k", vec![written("out")])];
        state_keyed_outputs(&mut units, 0, &whole);
        assert!(units[0].meta.launch.plane_axes.is_empty());
    }

    #[test]
    fn a_user_slot_named_like_a_compensation_plane_is_not_taken() {
        // the argument list holds a user plane `mu_comp` and the plane the walk made, `mu_low`
        let slots = [slot("mu"), slot("mu_comp"), slot("mu_low")];
        let made = ("mu".to_string(), "mu_low".to_string());
        let cells = compensated(sum(), &[], &slots, &[made]).unwrap();
        assert_eq!(cells[0].comp_slots, Some(vec![2]));
        let cells = compensated(sum(), &[], &slots, &[]).unwrap();
        assert_eq!(cells[0].comp_slots, None);
    }

    /// The kernel with two dispatch groups of two branches each over one sink, as an IR dump, and its walk order as dump positions.
    /// Both groups select on the plane `k`. The groups' branches cost different numbers, so the order of the records shows.
    const TWO_GROUPS: &str = include_str!("../../tests/data/two_groups.dump");
    const TWO_GROUPS_ORDER: [usize; 10] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10];

    /// Runs `check` on the context of the two-group kernel and its walk order; the context is freed after.
    fn with_two_groups(check: impl FnOnce(&Ctx, &[u32])) {
        let c = rx_ctx_new();
        let mut roots = vec![0u32; 4096];
        let mut n = 0;
        // SAFETY: `c` is the live context just made, the text and the buffers outlive the call.
        let status = unsafe {
            rx_load(
                c,
                TWO_GROUPS.as_ptr().cast(),
                TWO_GROUPS.len(),
                roots.as_mut_ptr(),
                roots.len(),
                &mut n,
            )
        };
        assert_eq!(status, RX_OK);
        let order: Vec<u32> = TWO_GROUPS_ORDER.iter().map(|&p| roots[p]).collect();
        // SAFETY: `c` is the live context; nothing else holds it.
        check(unsafe { &*c }, &order);
        // SAFETY: `c` is the live context made above and is not used after.
        unsafe { rx_ctx_free(c) };
    }

    #[test]
    fn the_groups_of_a_two_group_kernel_state_their_own_branch_costs_in_the_walks_order() {
        // oracle: the per-branch counts the cost model recorded for the kernel (the wing's cost oracle, case two_groups_one_sink)
        let auto = ("g".to_string(), Strategy::Auto);
        let found = [auto.clone(), ("h".to_string(), Strategy::Auto)];
        with_two_groups(|ctx, order| {
            let read = group_costs(ctx, order, &found).unwrap();
            let [first, second] = &read[..] else {
                panic!("two groups: {read:?}");
            };
            let counts = |g: &[m::BranchCost]| -> Vec<(f64, f64, f64, f64)> {
                g.iter()
                    .map(|b| (b.arith64, b.special64, b.loads, b.chain_special64))
                    .collect()
            };
            assert_eq!(counts(first), [(1.0, 0.0, 2.0, 0.0), (0.0, 1.0, 1.0, 1.0)]);
            assert_eq!(counts(second), [(0.0, 0.0, 1.0, 0.0), (0.0, 1.0, 1.0, 1.0)]);
            assert!(first.iter().chain(second).all(|b| b.stores.is_none()));
        });
    }

    #[test]
    fn a_group_that_is_not_auto_states_no_costs() {
        let found = [
            ("g".to_string(), Strategy::Switch),
            ("h".to_string(), Strategy::Segmented),
        ];
        with_two_groups(|ctx, order| {
            let costs = group_costs(ctx, order, &found).unwrap();
            assert_eq!(costs, [vec![], vec![]]);
        });
    }

    /// The recorded kernels of the gemm tests: the IR dump of each and the positions of its sinks in it.
    const KERNELS: &str = include_str!("../../tests/data/gemm_kernels.json");

    /// The matrix products of the recorded kernel `name`, from the recogniser over its dump.
    fn plan_of(name: &str) -> GemmPlan {
        let kernels = rcore::json::parse(KERNELS).unwrap();
        let case = kernels
            .get("cases")
            .and_then(|c| c.get(name))
            .expect("a recorded kernel");
        let dump = case.get("dump").and_then(rcore::json::J::str).unwrap();
        let c = rx_ctx_new();
        let mut roots = vec![0u32; 4096];
        let mut n = 0;
        // SAFETY: `c` is the live context made above, `dump` and `roots` outlive the call.
        let status = unsafe {
            rx_load(
                c,
                dump.as_ptr().cast(),
                dump.len(),
                roots.as_mut_ptr(),
                roots.len(),
                &mut n,
            )
        };
        assert_eq!(status, RX_OK);
        let sinks: Vec<u32> = case
            .get("sinks")
            .unwrap()
            .arr()
            .iter()
            .map(|p| match p {
                rcore::json::J::Num(at) => roots[*at as usize],
                _ => panic!("a sink position"),
            })
            .collect();
        // SAFETY: `c` is the live context made above; nothing else holds it.
        let found = recognize(unsafe { &mut *c }, &sinks, &[])
            .unwrap()
            .expect("a contraction");
        // SAFETY: `c` is the live context made above and is not used after.
        unsafe { rx_ctx_free(c) };
        gemm_plan(&found).expect("a product")
    }

    /// Oracle: the product the old route planned for `contract_staged` (the plane counted by the launch is the lane-major operand,
    /// so A is it, stored (8, lanes / 6) and read transposed; B is `w`, stored (8, 6); C is `y`, (lanes / 6, 6)), over six lanes
    /// per slot.
    #[test]
    fn a_contraction_with_a_plane_counted_by_the_launch_states_one_per_lane_extent_and_the_fan_in()
    {
        let slots = [slot("w"), slot("feat"), slot("y")];
        let gemm = gemm_member(&plan_of("contract_staged"), &slots).expect("a member");
        assert_eq!(gemm.lane_fanin, 6);
        let [call] = &gemm.calls[..] else {
            panic!("one call")
        };
        assert_eq!((call.a_slot, call.b_slot, call.c_slot), (1, 0, 2));
        assert_eq!(call.flags, m::GEMM_TRANS_A);
        let lane = m::GEMM_PER_LANE;
        assert_eq!(
            [
                call.a_rows,
                call.a_cols,
                call.b_rows,
                call.b_cols,
                call.c_rows,
                call.c_cols
            ],
            [8, lane, 8, 6, lane, 6]
        );
    }

    /// Oracle: the same planned for `contract`, whose planes have declared sizes: no extent is the run's, so there is no fan-in.
    #[test]
    fn a_contraction_over_planes_of_declared_sizes_states_numbers_only() {
        let slots = [slot("w"), slot("f"), slot("y")];
        let gemm = gemm_member(&plan_of("contract"), &slots).expect("a member");
        assert_eq!(gemm.lane_fanin, 0);
        let call = &gemm.calls[0];
        assert_eq!(
            [
                call.a_rows,
                call.a_cols,
                call.b_rows,
                call.b_cols,
                call.c_rows,
                call.c_cols
            ],
            [8, 5, 8, 6, 5, 6]
        );
    }

    #[test]
    fn a_unit_without_a_slot_of_an_operands_name_states_no_product() {
        assert!(gemm_member(&plan_of("contract"), &[slot("w"), slot("y")]).is_none());
    }
}
