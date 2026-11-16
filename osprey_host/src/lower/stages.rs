//! The split of a kernel into launches. A kernel that reads a value reduced over samples is cut at each such reduction into stages
//! run in sequence, and a kernel the host runs with interchanged loops has its wide values placed in scratch planes (shared by
//! name across the stages). The IR decisions are the wing's; this module turns each stage into a pending unit and notes what the
//! answer's stage block says of it.
use super::card::BuildCard;
use super::meta_fill;
use super::placed::{interchanged, place_sinks, place_unit};
use super::request::{DeclaredSlot, KernelRequest, Member, DTYPES, ROLES};
use super::stage_block::{
    CellRecord, KeyedRecord, KeysGrid, PlaneRecord, ScatterGrid, StageBlock, StageRecord,
    TeamRecord,
};
use super::unit_meta::UnitMeta;
use super::unit_meta::{Body, Launch};
use super::{canonical, finished_slot, typed_walk, Cards, Pending, FINISHED_PLANE};
use aether_writers::contract::types::{
    Active, Form, Grid, Scalar, Step, StepOp, Target, TeamShape, TeamStage,
};
use aether_writers::refuse::{bad, fault, nib, Refusal};
use raptor_ir::record::{Ctx, Fault};
use raptor_wings::scatter;
use raptor_wings::stages::Keyed;
use raptor_wings::stages::{
    names_under, scratch_planes, stages, stored_cells, Cell, Cut, StageFacts, Stages,
};
use raptor_wings::team::{Grid as TeamGrid, Plane, TeamLimits, TeamSpec, Unit};

/// What the answer's stage block holds of one unit.
pub(crate) struct Note {
    /// The unit as a stage of its kernel.
    pub stage: StageRecord,
    /// The cells of the kernel (carried by its first unit).
    pub cells: Vec<CellRecord>,
    /// The scratch planes the unit's placement made.
    pub planes: Vec<PlaneRecord>,
    /// The keyed planes of the kernel (carried by its first unit).
    pub keyed: Vec<KeyedRecord>,
    /// The unit that lists the stages of a cut kernel (carried by its first unit).
    pub source: Option<UnitMeta>,
}

/// A wing refusal as the refusal of the request.
fn wing(why: Fault) -> Refusal {
    match why {
        Fault::Refused(reason) => bad(reason),
        other => fault(format!("stages: {other:?}")),
    }
}

/// The step: each unit is cut at its reductions over samples and has its wide values placed where the target needs it.
pub(super) fn split(
    ctx: *mut Ctx,
    request: &KernelRequest,
    cards: &Cards,
    pending: Vec<Pending>,
) -> Result<Vec<Pending>, Refusal> {
    let mut out = vec![];
    for unit in pending {
        out.extend(one_unit(ctx, request, &cards.deciding, unit)?);
    }
    Ok(out)
}

/// What the split reads besides the IR: the request's settings and the deciding card's limits.
pub(super) fn stage_facts(request: &KernelRequest, card: &BuildCard) -> StageFacts {
    StageFacts {
        routed: request.settings.route_to_ordered,
        team: TeamLimits {
            form: request.settings.linalg_form,
            max_dim: card.linalg_max_dim,
            team_max_dim: card.linalg_team_max_dim,
            copy_threads: card.copy_threads,
            team_threads: card.team_threads,
            team_threads_large: card.team_threads_large,
            forced: request.settings.team_threads,
        },
        in_place: !request.kind.active_set && request.kind.lanes.is_none(),
        scatter: (card.scatter_max_chunks, card.scatter_threads),
    }
}

/// The units one pending unit lowers to.
fn one_unit(
    ctx: *mut Ctx,
    request: &KernelRequest,
    card: &BuildCard,
    mut unit: Pending,
) -> Result<Vec<Pending>, Refusal> {
    // A warp per lane is one launch with no stages and no placed value: the slicing already decided where its values live.
    if unit.warp.is_some() {
        return Ok(vec![unit]);
    }
    let facts = stage_facts(request, card);
    // SAFETY: `ctx` is the live context the caller holds; nothing else reads it while the stages are made.
    let found = stages(unsafe { &mut *ctx }, &unit.member.sinks, &facts).map_err(wing)?;
    match found {
        Stages::LoopLog => Err(nib(format!(
            "{}: an ordered accumulate inside a loop is not written yet; the log of a loop's accumulates needs one slot per iteration",
            unit.member.name
        ))),
        Stages::Unwritten => Err(nib(
            "the block stages of a sum over the lanes of a key of this shape",
        )),
        Stages::Whole => {
            if placement_wanted(ctx, request, (&unit.member, unit.lanes.is_some()))?
                && place_unit(ctx, request, card, &mut unit)?
            {
                unit.note = Some(whole_note(ctx, &unit)?);
            }
            Ok(vec![unit])
        }
        Stages::Cut(cut) => {
            if unit.one_step.is_some() || unit.segment.is_some() {
                return Err(bad(format!(
                    "{} reads a value reduced over samples inside a stepping or segmented kernel",
                    unit.member.name
                )));
            }
            if !cut.keyed.is_empty() && request.group != 1 {
                return Err(bad(format!(
                    "{} reduces over some lane axes and declares a group",
                    unit.member.name
                )));
            }
            staged(ctx, request, card, (unit, cut))
        }
    }
}

/// Whether wide values are placed in scratch planes for the body: when a target is the device (a fused group is never placed), else
/// when the host runs its loops interchanged. The body is given with whether it is a fused group.
fn placement_wanted(
    ctx: *mut Ctx,
    request: &KernelRequest,
    (member, fused): (&Member, bool),
) -> Result<bool, Refusal> {
    let device = request.targets.contains(&Target::Device) && !fused;
    Ok(device || interchanged(ctx, member)?)
}

/// What a placed whole kernel says of itself: one stage, its cells and the planes its placement made.
fn whole_note(ctx: *mut Ctx, unit: &Pending) -> Result<Note, Refusal> {
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let context = unsafe { &*ctx };
    let mut sinks = unit.member.sinks.clone();
    sinks.extend(unit.one_step.iter().flat_map(|step| step.sinks.clone()));
    Ok(Note {
        stage: StageRecord {
            kernel: unit.member.name.clone(),
            cells_in: vec![],
            cells_out: vec![],
            team: None,
        },
        cells: cell_records(
            stored_cells(context, &unit.member.sinks).map_err(wing)?,
            &[],
        ),
        planes: planes_under(context, &sinks)?,
        keyed: vec![],
        source: None,
    })
}

/// The cells as the answer records them: one a later stage reads is the runtime's own unless the caller declares it.
pub(super) fn cell_records(cells: Vec<Cell>, declared: &[String]) -> Vec<CellRecord> {
    cells
        .into_iter()
        .map(|c| CellRecord {
            internal: c.internal && !declared.contains(&c.name),
            name: c.name,
            columns: c.columns,
            op: c.op,
        })
        .collect()
}

/// The scratch planes of the placed values under the sinks.
fn planes_under(context: &Ctx, sinks: &[u32]) -> Result<Vec<PlaneRecord>, Refusal> {
    let planes = scratch_planes(context, sinks).map_err(wing)?;
    Ok(planes
        .into_iter()
        .map(|p| PlaneRecord {
            name: p.name,
            dtype: p.dtype,
            per_lane: p.per_lane,
            layout: None,
            keys: None,
            per_target: None,
            once: false,
            wide: true,
            fill: None,
        })
        .collect())
}

/// The slots a stage declares beyond its kernel's: the compensation plane of each summing cell it stores, and the finished-count
/// plane when it finishes samples.
fn stage_declared(
    ctx: *mut Ctx,
    base: &Member,
    sinks: &[u32],
) -> Result<Vec<DeclaredSlot>, Refusal> {
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let context = unsafe { &*ctx };
    // the kernel's declared slots hold the compensation planes of its own sums: each stage declares those of its own sinks
    let mut own = vec![];
    for &sink in &base.sinks {
        let node = &context.nodes[sink as usize];
        if context.table.is_a(node.class, "SampleSum") {
            own.push(context.str_field(sink, "name").map_err(wing)?);
        }
    }
    let mut declared: Vec<DeclaredSlot> = base
        .declared
        .iter()
        .filter(|slot| {
            slot.name != FINISHED_PLANE
                && !slot
                    .compensates
                    .as_ref()
                    .is_some_and(|row| own.contains(row))
        })
        .cloned()
        .collect();
    for &sink in sinks {
        let is_sum = context
            .table
            .is_a(context.nodes[sink as usize].class, "SampleSum")
            && context.str_field(sink, "op").map_err(wing)? == "sum";
        if !is_sum {
            continue;
        }
        let node = &context.nodes[sink as usize];
        let role = context.str_field(sink, "role").map_err(wing)?;
        let name = context.str_field(sink, "name").map_err(wing)?;
        declared.push(DeclaredSlot {
            role: *ROLES
                .iter()
                .find(|r| r.name() == role)
                .ok_or_else(|| fault("a cell with a role the request does not know"))?,
            name: format!("{name}_comp"),
            dtype: *DTYPES
                .iter()
                .find(|d| d.name() == node.dtype)
                .ok_or_else(|| fault("a cell with an element type the request does not know"))?,
            shape: node.shape.clone(),
            tag: node.tag.clone(),
            compensates: Some(name),
        });
    }
    let mut member = Member {
        declared,
        ..base.clone()
    };
    member.declared.extend(finished_slot(ctx, &member));
    Ok(member.declared)
}

/// The unit of a cut kernel that lists its stages and has no entry of its own: its slots are what the caller binds over the whole
/// kernel and its digest is that of the kernel's walk. Its launch is the caller's view too: the kernel's lane space and the declared
/// axes of its sink planes.
fn source_unit(
    ctx: *mut Ctx,
    request: &KernelRequest,
    member: &Member,
) -> Result<UnitMeta, Refusal> {
    let (walk, digest) = canonical(ctx, member)?;
    let typed = typed_walk(ctx, walk)?;
    let slots = meta_fill::slots(ctx, walk, &typed, request)?;
    let launch = Launch {
        group: request.group,
        lane_labels: request.lane_space.iter().map(|a| a.label.clone()).collect(),
        lane_extents: request
            .lane_space
            .iter()
            .map(|a| a.extent.unwrap_or(-1))
            .collect(),
        plane_axes: meta_fill::plane_axes(request, &slots),
        ..Launch::default()
    };
    Ok(UnitMeta {
        name: format!("{}_source", member.name),
        digest,
        scalar_type: meta_fill::scalar(request.scalar),
        body: Body {
            access: meta_fill::access(typed.access),
            slots,
            ..Body::default()
        },
        launch,
        ..UnitMeta::default()
    })
}

/// The pending units of a cut kernel, its wide values placed in scratch planes shared across the stages.
fn staged(
    ctx: *mut Ctx,
    request: &KernelRequest,
    card: &BuildCard,
    (unit, cut): (Pending, Cut),
) -> Result<Vec<Pending>, Refusal> {
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let mut taken: Vec<String> = vec![];
    for stage in &cut.stages {
        taken.extend(names_under(unsafe { &*ctx }, &stage.sinks).map_err(wing)?);
    }
    let source = (cut.stages.len() > 1)
        .then(|| source_unit(ctx, request, &unit.member))
        .transpose()?;
    let cells = cell_records(cut.cells.clone(), &request.planes);
    let keyed: Vec<KeyedRecord> = cut
        .keyed
        .iter()
        .map(|k| KeyedRecord {
            name: k.name.clone(),
            width: k.width,
            axes: k.kept.clone(),
            payload: k.payload.clone(),
        })
        .collect();
    let mut wide = false;
    for stage in cut.stages.iter().filter(|stage| stage.team.is_none()) {
        let declared = stage_declared(ctx, &unit.member, &stage.sinks)?;
        let member = Member {
            sinks: stage.sinks.clone(),
            declared,
            ..unit.member.clone()
        };
        wide |= placement_wanted(ctx, request, (&member, unit.lanes.is_some()))?;
    }
    let mut out = vec![];
    for (index, stage) in cut.stages.iter().enumerate() {
        let name = match index {
            0 => unit.member.name.clone(),
            n => format!("{}_{}", unit.member.name, n + 1),
        };
        let mut sinks = stage.sinks.clone();
        let mut planes = team_planes(stage.team.as_ref(), request.scalar == Scalar::Banded);
        if wide && stage.team.is_none() {
            if let Some(placed) = place_sinks(ctx, request, card, &sinks, &taken)? {
                // SAFETY: `ctx` is the live context the caller holds; it is only read here.
                planes = planes_under(unsafe { &*ctx }, &placed)?;
                taken.extend(planes.iter().map(|p| p.name.clone()));
                sinks = placed;
            }
        }
        let first = index == 0;
        let team = stage.team.clone();
        let declared = match &team {
            Some(spec) if !spec.steps.is_empty() => {
                let mut held: Vec<DeclaredSlot> = unit
                    .member
                    .declared
                    .iter()
                    .filter(|_| first)
                    .cloned()
                    .collect();
                held.extend(read_slots(ctx, &spec.reads)?);
                held
            }
            _ => stage_declared(ctx, &unit.member, &sinks)?,
        };
        out.push(Pending {
            member: Member {
                name: name.clone(),
                sinks,
                declared,
                ..unit.member.clone()
            },
            segment: None,
            one_step: None,
            active: unit.active.clone(),
            lanes: unit.lanes.clone(),
            boxable: unit.boxable && team.is_none(),
            team: team.clone(),
            note: Some(Note {
                stage: StageRecord {
                    kernel: name,
                    cells_in: stage.cells_in.clone(),
                    cells_out: stage.cells_out.clone(),
                    team: team.as_ref().map(team_record),
                },
                cells: if first { cells.clone() } else { vec![] },
                planes,
                keyed: if first { keyed.clone() } else { vec![] },
                source: if first { source.clone() } else { None },
            }),
            warp: None,
            group: None,
            exports_block: true,
        });
    }
    Ok(out)
}

/// The stage blocks from the notes of the units. The units that run one after the other (the stages of a cut kernel, the
/// pieces of a chain) share one block. A unit that covers one run of a segmented input is an alternative of its dispatch
/// group, never a stage after another branch: its note is a block of its own, with its own scratch planes.
pub(super) fn blocks(pending: &[Pending]) -> Vec<StageBlock> {
    let mut sequence = StageBlock::default();
    let mut in_sequence = false;
    let mut alternatives = vec![];
    for unit in pending {
        let Some(note) = unit.note.as_ref() else {
            continue;
        };
        if unit.segment.is_some() {
            let mut own = StageBlock::default();
            add(&mut own, note);
            alternatives.push(own);
        } else {
            in_sequence = true;
            add(&mut sequence, note);
        }
    }
    let mut out = alternatives;
    if in_sequence {
        out.insert(0, sequence);
    }
    out
}

/// Adds what one unit's note says to a block; a scratch plane two stages share is recorded once.
fn add(block: &mut StageBlock, note: &Note) {
    block.stages.push(note.stage.clone());
    block.cells.extend(note.cells.iter().cloned());
    block.keyed.extend(note.keyed.iter().cloned());
    block.source = block.source.take().or_else(|| note.source.clone());
    for plane in &note.planes {
        if !block.team_planes.iter().any(|p| p.name == plane.name) {
            block.team_planes.push(plane.clone());
        }
    }
}

/// The scratch planes of a team stage as the answer records them. A partial sum is the build's scalar, or a band of them when the
/// scalar is banded.
fn team_planes(team: Option<&TeamSpec>, banded: bool) -> Vec<PlaneRecord> {
    let Some(spec) = team else { return vec![] };
    let keys = match &spec.grid {
        TeamGrid::Keys { keyed, .. } => Some(keyed_record(keyed)),
        _ => None,
    };
    let target = match &spec.grid {
        TeamGrid::Chunks { target } | TeamGrid::Targets { target } => Some(target.clone()),
        _ => None,
    };
    let dtype = |plane: &Plane| match (plane.dtype, banded) {
        ("partial", true) => "band_partial",
        ("partial", false) => "real",
        (other, _) => other,
    };
    spec.planes
        .iter()
        .map(|plane| PlaneRecord {
            name: plane.name.clone(),
            dtype: dtype(plane).to_string(),
            per_lane: plane.per_lane,
            layout: None,
            keys: keys.clone().filter(|_| plane.unit == Unit::Key),
            per_target: (plane.unit == Unit::Target)
                .then(|| target.clone())
                .flatten(),
            once: plane.unit == Unit::Once,
            wide: false,
            fill: plane.fill,
        })
        .collect()
}

/// A keyed plane as the answer records it.
fn keyed_record(keyed: &Keyed) -> KeyedRecord {
    KeyedRecord {
        name: keyed.name.clone(),
        width: keyed.width,
        axes: keyed.kept.clone(),
        payload: keyed.payload.clone(),
    }
}

/// The team of a stage as the answer records it.
fn team_record(spec: &TeamSpec) -> TeamRecord {
    let keys = match &spec.grid {
        TeamGrid::Keys { keyed, groups } => Some(KeysGrid {
            groups: *groups,
            keyed: keyed_record(keyed),
        }),
        _ => None,
    };
    let scatter = match &spec.grid {
        TeamGrid::Chunks { target } => Some(("chunks", target)),
        TeamGrid::Targets { target } => Some(("targets", target)),
        _ => None,
    };
    let scatter = scatter.map(|(grid, target)| ScatterGrid {
        grid: grid.to_string(),
        target: target.clone(),
        status: (grid == "chunks").then(|| scatter::names(target)[3].clone()),
    });
    TeamRecord {
        threads: spec.threads,
        lanes: 1,
        warps: 1,
        keys,
        scatter,
    }
}

/// The slots of the input planes a team stage reads in place: each is bound as an argument of the stage.
fn read_slots(ctx: *mut Ctx, reads: &[u32]) -> Result<Vec<DeclaredSlot>, Refusal> {
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let context = unsafe { &*ctx };
    let missing = || fault("a plane read in place with a role or a type the request does not know");
    reads
        .iter()
        .map(|&leaf| {
            let (role, node) = (
                context.str_field(leaf, "role").map_err(wing)?,
                &context.nodes[leaf as usize],
            );
            Ok(DeclaredSlot {
                role: *ROLES
                    .iter()
                    .find(|r| r.name() == role)
                    .ok_or_else(missing)?,
                name: context.str_field(leaf, "name").map_err(wing)?,
                dtype: *DTYPES
                    .iter()
                    .find(|d| d.name() == node.dtype)
                    .ok_or_else(missing)?,
                shape: node.shape.clone(),
                tag: node.tag.clone(),
                compensates: None,
            })
        })
        .collect()
}

/// The writers' form of a team unit: a stage with calls is written from them, one that only copies is a plain body on a team. The
/// unit's team shape comes with it.
pub(super) fn team_form(
    spec: &TeamSpec,
    active: Option<Active>,
) -> Result<(Option<Form>, TeamShape), Refusal> {
    let shape = TeamShape {
        threads: spec.threads,
        warps: 1,
    };
    if spec.steps.is_empty() {
        return Ok((None, shape));
    }
    let mut steps = vec![];
    for step in &spec.steps {
        steps.push(Step {
            op: StepOp::named(step.op).ok_or_else(|| fault(format!("a team step {}", step.op)))?,
            n: step.n,
            k: step.k,
            m: step.m,
            max_chunks: step.max_chunks,
            planes: step.planes.clone(),
        });
    }
    let planes = spec
        .planes
        .iter()
        .map(|p| (p.name.clone(), p.per_lane))
        .collect();
    let grid = match spec.grid {
        TeamGrid::Lanes => Grid::Lanes,
        TeamGrid::Keys { .. } => Grid::Keys,
        TeamGrid::Chunks { .. } => Grid::Chunks,
        TeamGrid::Targets { .. } => Grid::Targets,
    };
    let stage = TeamStage {
        grid,
        planes,
        reads: spec.reads.clone(),
        steps,
        active,
    };
    Ok((Some(Form::TeamStage(stage)), shape))
}
