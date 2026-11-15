//! The typed meta of the units, from what the lowering already knows: each unit's walk, request and texts, and what the split into
//! stages decided for the kernel. A member nothing here knows is left absent (see `NOT_COMPUTED`), never given a value.
use super::active::{COUNT_PLANE, MAP_PLANE};
use super::card::BuildCard;
use super::request::{KernelRequest, TargetText, UnitAnswer};
use super::stage_block::{KeyedRecord, PlaneRecord, StageBlock, TeamRecord};
use super::unit_meta as m;
use super::{wing, Pending, FINISHED_PLANE};
use aether_writers::contract::types::{Access, Dtype, Role, Scalar, Target, Walk};
use aether_writers::ir::analysis::param_plane;
use aether_writers::refuse::{fault, not_written, Refusal};
use aether_writers::RxCtx;
use raptor_ir::record::Ctx;
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

/// The type of a slot on the wire: an index leaf is bound as the 64-bit `Int` by the entry's writer, a float is `Real`.
fn dtype(dtype: Dtype) -> m::WireDtype {
    match dtype {
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
pub(super) fn slots(ctx: *mut Ctx, walk: u32, typed: &Walk) -> Result<Vec<m::Slot>, Refusal> {
    let ctx = ctx as *mut RxCtx;
    typed
        .slots
        .iter()
        .map(|s| {
            let planar = s.role == Role::Lookup && param_plane(ctx, walk, &s.name)?;
            Ok(m::Slot {
                flags: if planar { m::SLOT_PARAM_PLANE } else { 0 },
                role: role(s.role),
                dtype: dtype(s.dtype),
                shape: s.shape.clone(),
                name: s.name.clone(),
                ..m::Slot::default()
            })
        })
        .collect()
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

/// The launch record: the group, the boxes and the slots of the index maps the unit runs over.
fn launch(parts: &Parts, slots: &[m::Slot]) -> m::Launch {
    let mut found = m::Launch {
        group: parts.request.group,
        host_box: thread_box(parts.texts, Target::Host),
        device_box: thread_box(parts.texts, Target::Device),
        block: parts
            .request
            .settings
            .launch_bounds
            .map_or(0, |(threads, _)| threads),
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
    Ok(Some(m::Finish {
        mask_slot: slot(mask)?,
        counter_slot: slot(FINISHED_PLANE)?,
        ..m::Finish::default()
    }))
}

/// The meta of one unit, without what only its kernel's stages tell (see `stages`).
pub(super) fn unit(parts: &Parts) -> Result<m::UnitMeta, Refusal> {
    let slots = slots(parts.ctx, parts.walk, parts.typed)?;
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let stores = !stored_cells(unsafe { &*parts.ctx }, &parts.pending.member.sinks)
        .map_err(wing)?
        .is_empty();
    let name = &parts.pending.member.name;
    let entry = |target: Target| parts.texts.iter().find(|t| t.target == target);
    let device = entry(Target::Device).map(|t| t.entry.clone());
    Ok(m::UnitMeta {
        name: name.clone(),
        digest: parts.digest.to_string(),
        scalar_type: scalar(parts.request.scalar),
        host_entry: entry(Target::Host).map(|t| t.symbol.clone()),
        device_image: device.as_ref().map(|_| format!("{name}__image")),
        device_image_size: device.as_ref().map(|_| format!("{name}__image_size")),
        device_function: device,
        launch: launch(parts, &slots),
        finish: finish(parts, &slots)?,
        body: m::Body {
            access: access(parts.typed.access),
            cell_row: if stores { parts.card.cells_row } else { 0 },
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
        name: plane.name.clone(),
        ..m::TeamPlane::default()
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
    meta.body.scatter_chunks = if planes.is_empty() { 0 } else { scatter_chunks };
    meta.body.team_planes = planes;
    Ok(())
}
