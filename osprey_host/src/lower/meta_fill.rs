//! The typed meta of the units, from what the lowering already knows: each unit's walk, request and texts, and what the split into
//! stages decided for the kernel. A member nothing here knows is left absent (see `NOT_COMPUTED`), never given a value.
use super::active::{COUNT_PLANE, MAP_PLANE};
use super::card::BuildCard;
use super::request::{Body, KernelRequest, LaneAxis, Steps, TargetText, UnitAnswer};
use super::segment::FUSED_STEPS_PLANE;
use super::stage_block::{KeyedRecord, PlaneRecord, StageBlock, TeamRecord};
use super::stages::cell_records;
use super::unit_meta as m;
use super::{wing, Pending, FINISHED_PLANE};
use aether_writers::contract::types::{Access, Dtype, Role, Scalar, Target, Walk};
use aether_writers::refuse::{bad, fault, not_written, Refusal};
use raptor_ir::record::Ctx;
use raptor_wings::segment::{class_name, offsets_name, stale_name, strategies, Strategy};
use raptor_wings::stages::stored_cells;

/// What the meta of one unit is written from.
pub(super) struct Parts<'a> {
    /// The context that holds the kernel's IR.
    pub ctx: *mut Ctx,
    /// The request.
    pub request: &'a KernelRequest,
    /// The numbers the lowering decided on.
    pub card: &'a BuildCard,
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
}

/// The header's value of a count the runner chooses.
const AUTO_STEPS: i64 = -1;

/// The role of a slot in the header's closed set.
fn role(role: Role) -> m::SlotRole {
    match role {
        Role::Mutable => m::SlotRole::Mutable,
        Role::PerSample => m::SlotRole::PerSample,
        Role::VecIn => m::SlotRole::VecIn,
        Role::Lookup => m::SlotRole::Lookup,
        Role::WideOut => m::SlotRole::WideOut,
        Role::Uniform => m::SlotRole::Uniform,
        Role::MatIn => m::SlotRole::MatIn,
        Role::AccumOut => m::SlotRole::AccumOut,
        Role::Terminated => m::SlotRole::Terminated,
        Role::WideIn => m::SlotRole::WideIn,
        Role::Nsamples => m::SlotRole::Nsamples,
        Role::Out => m::SlotRole::Out,
    }
}

/// The type of a slot on the wire: an index leaf is bound as the 64-bit `Int` by the entry's writer, a float is `Real`. A
/// by-value integer (a uniform) is bound as that `Int` whatever its width: no unit that has one narrows `Int`.
fn dtype(role: Role, dtype: Dtype) -> m::WireDtype {
    match dtype {
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
            Ok(m::Slot {
                flags: if planar { m::SLOT_PARAM_PLANE } else { 0 }
                    | if gathered.contains(&s.name) {
                        m::SLOT_GATHER
                    } else {
                        0
                    }
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
                gather: gathered
                    .contains(&s.name)
                    .then(|| pattern_of(patterns, &s.name))
                    .flatten(),
                shape: s.shape.clone(),
                name: s.name.clone(),
            })
        })
        .collect()
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

/// The prefix of the name of the accumulated gradient of a plane.
const ADJOINT_PREFIX: &str = "bar_";

/// The lookup or wide input slot a gradient slot is sized like: the accumulated output `bar_<plane>` of a plane the unit reads by
/// index, whatever the lane count.
fn like_slot(typed: &Walk, at: usize) -> Option<usize> {
    let slot = &typed.slots[at];
    let plane = slot.name.strip_prefix(ADJOINT_PREFIX)?;
    (slot.role == Role::AccumOut).then_some(())?;
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

/// The declared axes of the sink planes the unit binds, each at its slot.
pub(super) fn plane_axes(request: &KernelRequest, slots: &[m::Slot]) -> Vec<m::PlaneAxes> {
    request
        .sink_axes
        .iter()
        .filter_map(|sink| {
            Some(m::PlaneAxes {
                slot: slot_at(slots, &sink.name)?,
                labels: sink.axes.iter().map(|a| a.label.clone()).collect(),
                extents: sink.axes.iter().map(|a| a.extent.unwrap_or(-1)).collect(),
            })
        })
        .collect()
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
            plane_axes(parts.request, slots)
        },
        ..m::Launch::default()
    };
    if let Some(active) = &parts.pending.active {
        let map = active
            .map_binding
            .as_ref()
            .and_then(|_| slot_at(slots, MAP_PLANE));
        let count = slot_at(slots, COUNT_PLANE);
        if parts.request.kind.lanes.is_some() {
            (found.index_map_slot, found.index_count_slot) = (map, count);
        } else {
            (found.active_map_slot, found.active_count_slot) = (map, count);
        }
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
    Ok(Some(m::Finish {
        flags: if step.is_some_and(|s| s.entry_counts) {
            m::FINISH_ENTRY_COUNTS
        } else {
            0
        },
        step_ops: step.map(|s| s.ops),
        mask_slot: slot(mask)?,
        counter_slot: slot(FINISHED_PLANE)?,
        steps,
        steps_max,
        steps_slot: slot_at(slots, FUSED_STEPS_PLANE).filter(|_| steps == AUTO_STEPS),
        ..m::Finish::default()
    }))
}

/// The dispatch groups of a unit's walk: each group's name and strategy, and for an auto group the slot of its branch-mix plane.
fn dispatches(parts: &Parts, slots: &[m::Slot]) -> Result<Vec<m::Dispatch>, Refusal> {
    if let Some(group) = &parts.pending.group {
        let at = |name: String| slot_at(slots, &name);
        return Ok(vec![m::Dispatch {
            strategy: m::DispatchStrategy::Segmented,
            class_slot: at(class_name(&group.name)),
            stale_slot: at(stale_name(&group.name)),
            offsets_slot: at(offsets_name(&group.name)),
            classify: group.classify.as_ref().map(|name| m::UnitRef {
                name: name.clone(),
                digest: String::new(),
            }),
            name: group.name.clone(),
            ..m::Dispatch::default()
        }]);
    }
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let found = strategies(unsafe { &*parts.ctx }, parts.walk).map_err(wing)?;
    found
        .into_iter()
        .map(|(name, strategy)| {
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
            Ok(m::Dispatch {
                strategy: match strategy {
                    Strategy::Predicated => m::DispatchStrategy::Predicated,
                    Strategy::Switch => m::DispatchStrategy::Switch,
                    Strategy::Segmented => m::DispatchStrategy::Segmented,
                    Strategy::Auto => m::DispatchStrategy::Auto,
                },
                mix_slot,
                name,
                ..m::Dispatch::default()
            })
        })
        .collect()
}

/// Fills the digest of each classify unit a dispatch group names, once every unit of the answer is written.
pub(super) fn classify_digests(units: &mut [UnitAnswer]) {
    let digests: Vec<(String, String)> = units
        .iter()
        .map(|u| (u.meta.name.clone(), u.meta.digest.clone()))
        .collect();
    for group in units.iter_mut().flat_map(|u| &mut u.meta.body.dispatches) {
        if let Some(unit) = &mut group.classify {
            if let Some((_, digest)) = digests.iter().find(|(name, _)| *name == unit.name) {
                unit.digest = digest.clone();
            }
        }
    }
}

/// The meta of one unit, without what only its kernel's stages tell (see `stages`).
pub(super) fn unit(parts: &Parts) -> Result<m::UnitMeta, Refusal> {
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
            })
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    let name = &parts.pending.member.name;
    let entry = |target: Target| parts.texts.iter().find(|t| t.target == target);
    let device = entry(Target::Device).map(|t| t.entry.clone());
    let device_format = device
        .as_ref()
        .and(parts.request.image_format)
        .unwrap_or(m::ImageFormat::None);
    let host = entry(Target::Host);
    let dispatches = dispatches(parts, &slots)?;
    let companions = facts
        .companions
        .iter()
        .map(|c| {
            let (comp, target) = (&c.plane, &c.target);
            let at = |plane: &str| {
                slot_at(&slots, plane)
                    .ok_or_else(|| fault(format!("a companion pair over the plane {plane}")))
            };
            Ok(m::Companion {
                slot: at(comp)?,
                target_slot: at(target)?,
                host_fold_entry: host
                    .and_then(|t| t.facts.companions.iter().find(|h| h.plane == *comp))
                    .and_then(|h| h.fold_entry.clone()),
                ..m::Companion::default()
            })
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    Ok(m::UnitMeta {
        name: name.clone(),
        digest: parts.digest.to_string(),
        scalar_type: scalar(parts.request.scalar),
        host_entry: entry(Target::Host).map(|t| t.facts.symbol.clone()),
        device_image: device.as_ref().map(|_| format!("{name}__image")),
        device_image_size: device.as_ref().map(|_| format!("{name}__image_size")),
        device_image_format: device_format,
        device_function: device,
        launch: launch(parts, &slots),
        finish: finish(parts, &slots)?,
        body: m::Body {
            access: access(parts.typed.access),
            cell_row: if stores { parts.card.cells_row } else { 0 },
            cells,
            companions,
            dispatches,
            slots,
            ..m::Body::default()
        },
        ..m::UnitMeta::default()
    })
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
        "max" => Ok(m::ReduceOp::Max),
        "min" => Ok(m::ReduceOp::Min),
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
        wide: if plane.wide {
            m::WideLayout::Plain
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
        });
    }
    meta.body.keyed = keyed;
    // the chunks of an ordered scatter are stated when the unit has scratch planes or a stage runs on a team
    let teamed = !planes.is_empty() || meta.stages.iter().any(|s| s.team.is_some());
    meta.body.scatter_chunks = if teamed { scatter_chunks } else { 0 };
    meta.body.team_planes = planes;
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
