//! The warp form of a kernel built for the device: one warp serves a lane, and every wide value is held sliced across the warp's
//! lanes. The slicing is the wing's; this step decides whether a kernel takes the form, how many lanes one block serves (the
//! rows of the tile its vector outputs go through) and what the answer's stage block says of the launch. The numbers come from
//! the device card.
use super::card::DeviceCard;
use super::request::{Body, KernelRequest, Member, WideForm};
use super::stage_block::TeamRecord;
use super::stages::{stage_facts, Note};
use super::{canonical, leaf_slots, real_bytes, wing, Cards, Pending};
use aether_writers::contract::types::{Form, Scalar, Target, TeamShape, Unit};
use aether_writers::refuse::{nib, Refusal};
use raptor_ir::record::Ctx;
use raptor_wings::lanes::slice::{slice_sinks, SliceSpec};
use raptor_wings::stages::{stages, Stages};

/// The rows a block of the warp form serves, from the largest the tile allows.
const TILE_ROWS: [i64; 3] = [8, 4, 2];

/// A unit written as a warp per lane.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Warp {
    /// The lanes one block serves, one warp each.
    pub rows: i64,
    /// The warps of one block when it serves a single lane.
    pub warps: i64,
    /// The lanes of a warp.
    pub size: i64,
}

impl Warp {
    /// The threads of one block.
    pub fn threads(&self) -> i64 {
        self.size * self.rows
    }

    /// The writers' unit as a warp per lane: its form, the lanes a block serves and the team of its block.
    pub fn shape(&self, unit: &mut Unit) {
        unit.form = Form::WarpStage;
        unit.rows = self.rows;
        unit.team = Some(TeamShape {
            threads: self.threads(),
            warps: self.warps,
        });
    }
}

/// Whether the request asks for the warp form, or leaves it to the automatic choice: a bundle for the device alone.
fn asked(request: &KernelRequest) -> Result<Option<bool>, Refusal> {
    let device = request.targets.contains(&Target::Device);
    let host = request.targets.contains(&Target::Host);
    Ok(match (request.settings.wide_form, device) {
        (WideForm::Team, true) if host => {
            return Err(nib("a warp per lane in a bundle with a host target"))
        }
        (WideForm::Team, true) => Some(true),
        (WideForm::Auto, true) if !host => Some(false),
        _ => None,
    })
}

/// The step: a kernel that is one plain launch of wide values the form has a rule for becomes a warp per lane.
pub(super) fn slice(
    ctx: *mut Ctx,
    request: &KernelRequest,
    cards: &Cards,
    pending: Vec<Pending>,
) -> Result<Vec<Pending>, Refusal> {
    let (Some(explicit), Some(device)) = (asked(request)?, cards.device.as_ref()) else {
        return Ok(pending);
    };
    let plain = matches!(request.body, Body::Plain(_)) && request.group == 1;
    pending
        .into_iter()
        .map(|unit| match plain && unit.segment.is_none() {
            true => one_unit(ctx, request, (&cards.deciding, device, explicit), unit),
            false => Ok(unit),
        })
        .collect()
}

/// One unit as a warp per lane, or as it is when the form has no rule for it.
fn one_unit(
    ctx: *mut Ctx,
    request: &KernelRequest,
    (deciding, device, explicit): (&super::card::BuildCard, &DeviceCard, bool),
    mut unit: Pending,
) -> Result<Pending, Refusal> {
    let facts = stage_facts(request, deciding);
    let bodies = std::iter::once(&unit.member).chain(unit.one_step.as_ref());
    for body in bodies {
        // SAFETY: `ctx` is the live context the caller holds; nothing else reads it while the stages are planned.
        let found = stages(unsafe { &mut *ctx }, &body.sinks, &facts).map_err(wing)?;
        if found != Stages::Whole {
            return Ok(unit);
        }
    }
    let scalar = match real_bytes(request.scalar) {
        4 => "float32",
        _ => "float64",
    };
    if device.wide_rows.iter().any(|row| row == scalar) {
        return Err(nib("staged rows over a card row or a forced wide setting"));
    }
    let spec = SliceSpec {
        warp: device.warp_size,
        slice_min: device.slice_min,
        team_min: device.team_min,
        explicit,
    };
    let Some(main) = sliced(ctx, &unit.member, spec)? else {
        return Ok(unit);
    };
    let single = match &unit.one_step {
        Some(step) => match sliced(ctx, step, spec)? {
            Some(sinks) => Some(sinks),
            None => return Ok(unit),
        },
        None => None,
    };
    let kept = kept_slots(ctx, &unit, &main)?;
    let bodies = 1 + i64::from(single.is_some());
    let plain = !request.kind.active_set && request.kind.lanes.is_none();
    let rows = if plain {
        tile_rows(ctx, &main, request.scalar, bodies, device)
    } else {
        1
    };
    unit.member.sinks = main;
    unit.member.declared.extend(kept);
    if let (Some(step), Some(sinks)) = (unit.one_step.as_mut(), single) {
        step.sinks = sinks;
    }
    let warp = Warp {
        rows,
        warps: if rows == 1 { request.settings.warps } else { 1 },
        size: device.warp_size,
    };
    unit.note = Some(note(ctx, &unit.member, &warp)?);
    unit.boxable = false;
    unit.warp = Some(warp);
    Ok(unit)
}

/// The sinks of a body in the warp form, when it has one.
fn sliced(ctx: *mut Ctx, body: &Member, spec: SliceSpec) -> Result<Option<Vec<u32>>, Refusal> {
    // SAFETY: `ctx` is the live context the caller holds; nothing else reads it while the sinks are made.
    slice_sinks(unsafe { &mut *ctx }, &body.sinks, spec).map_err(wing)
}

/// The slots the kernel as written binds that its warp form no longer reads (a value the slicing dropped): they stay bound.
fn kept_slots(
    ctx: *mut Ctx,
    unit: &Pending,
    sinks: &[u32],
) -> Result<Vec<super::request::DeclaredSlot>, Refusal> {
    let mut written = vec![];
    for body in std::iter::once(&unit.member).chain(unit.one_step.as_ref()) {
        written.extend(leaf_slots(ctx, canonical(ctx, body)?.0)?);
    }
    let sliced = Member {
        sinks: sinks.to_vec(),
        ..unit.member.clone()
    };
    let held = leaf_slots(ctx, canonical(ctx, &sliced)?.0)?;
    let mut kept: Vec<super::request::DeclaredSlot> = vec![];
    for slot in written {
        let bound = |other: &super::request::DeclaredSlot| {
            other.role == slot.role && other.name == slot.name
        };
        if !held.iter().any(bound) && !kept.iter().any(bound) {
            kept.push(slot);
        }
    }
    Ok(kept)
}

/// The lanes a block serves: the most of 8, 4 and 2 whose output tiles fit the card's bytes (a kernel with two bodies declares
/// them twice), else 1 (a warp per block, storing straight to the planes). Only vector outputs are staged.
fn tile_rows(ctx: *mut Ctx, sinks: &[u32], scalar: Scalar, bodies: i64, card: &DeviceCard) -> i64 {
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let context = unsafe { &*ctx };
    let widths: Vec<&Vec<i64>> = sinks
        .iter()
        .filter(|&&sink| {
            context
                .table
                .is_a(context.nodes[sink as usize].class, "WarpStore")
        })
        .map(|&sink| &context.nodes[sink as usize].shape)
        .filter(|shape| shape.first().is_some_and(|&w| w >= card.slice_min))
        .collect();
    if widths.is_empty() || widths.iter().any(|shape| shape.len() != 1) {
        return 1;
    }
    let real = real_bytes(scalar);
    let total: i64 = widths.iter().map(|shape| shape[0]).sum();
    TILE_ROWS
        .into_iter()
        .find(|rows| bodies * total * (rows + 1) * real <= card.tile_bytes)
        .unwrap_or(1)
}

/// The refusal of a warp form over a body that stores a reduction over samples: a construct this build does not write.
fn stores_cells() -> Refusal {
    nib("a warp form that stores a reduction over samples")
}

/// What the answer's stage block says of the launch: one stage on a team of the block's threads.
fn note(ctx: *mut Ctx, member: &Member, warp: &Warp) -> Result<Note, Refusal> {
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let context = unsafe { &*ctx };
    let cells = raptor_wings::stages::stored_cells(context, &member.sinks).map_err(wing)?;
    if !cells.is_empty() {
        return Err(stores_cells());
    }
    Ok(Note {
        stage: super::stage_block::StageRecord {
            kernel: member.name.clone(),
            cells_in: vec![],
            cells_out: vec![],
            team: Some(TeamRecord {
                threads: warp.threads(),
                lanes: warp.rows,
                warps: warp.warps,
                keys: None,
                scatter: None,
            }),
        },
        cells: vec![],
        planes: vec![],
        keyed: vec![],
        source: None,
    })
}

#[cfg(test)]
mod tests {
    //! A warp form over cells is refused as a construct, not as a fault.
    use super::*;
    use aether_writers::refuse::RefusalKind;

    #[test]
    fn a_warp_form_that_stores_cells_is_not_written() {
        assert_eq!(stores_cells().kind, RefusalKind::NotWritten);
    }
}
