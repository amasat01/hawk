//! The chain step: a chain of kernels is one kernel when the fusion rule fuses every boundary between its members, else a unit per
//! piece, each piece reading what an earlier piece handed on in a plane the runtime makes per lane. The IR decisions (the
//! composition, the cuts, the pieces) come from the wing analysis; this module reads the fusion row of the build's target from the facts and
//! hands the pieces to the steps that follow as pending units.
use super::request::{DeclaredSlot, KernelRequest, Member};
use super::stage_block::{PlaneRecord, StageRecord};
use super::stages::Note;
use super::{carries_no_forced_access, held_exempt, merge_slot, real_bytes, wing, Pending};
use crate::facts;
use aether_writers::contract::types::{Scalar, Target};
use aether_writers::refuse::{bad, fault, nib, Refusal};
use raptor_ir::record::Ctx;
use raptor_wings::chain::{bind, cuts, pieces, Member as ChainMember, Piece};
use std::collections::HashSet;
use std::path::Path;

/// The lanes the cut rule weighs a cut at: the extent is not known when the kernel is built.
const CHAIN_SAMPLES: i64 = 1 << 16;

/// The name of the unit of piece `index`: the chain's own name, then the name and the piece's place from the second on.
fn piece_name(name: &str, index: usize) -> String {
    match index {
        0 => name.to_string(),
        _ => format!("{name}_{}", index + 1),
    }
}

/// The pieces a chain is cut into: the whole chain composed as one piece when no boundary is cut. A compaction group runs as one
/// kernel, so it is never cut.
fn cut_pieces(
    ctx: &mut Ctx,
    request: &KernelRequest,
    members: &[ChainMember],
    row: &facts::FusionRow,
) -> Result<Vec<Piece>, Refusal> {
    let cutting = members.len() > 1 && request.group == 1;
    if cutting && request.scalar == Scalar::Banded {
        return Err(nib("the cuts of a chain of a banded kernel"));
    }
    let cut = match cutting {
        true => cuts(
            ctx,
            members,
            row,
            (real_bytes(request.scalar) == 4, CHAIN_SAMPLES),
        )
        .map_err(wing)?,
        false => vec![],
    };
    if cut.is_empty() {
        let sinks = bind(ctx, members, &HashSet::new()).map_err(wing)?.sinks;
        return Ok(vec![Piece {
            sinks,
            planes: vec![],
        }]);
    }
    pieces(ctx, members, &cut).map_err(wing)
}

/// The slots the members declare, each name once: a piece of the chain keeps the slots its members stated.
fn declared_by_members(members: &[Member]) -> Result<Vec<DeclaredSlot>, Refusal> {
    let mut declared: Vec<DeclaredSlot> = vec![];
    for slot in members.iter().flat_map(|m| &m.declared) {
        merge_slot(&mut declared, slot.clone())
            .map_err(|why| bad(format!("chain members: {why}")))?;
    }
    Ok(declared)
}

/// The pending units of a chain, one per piece, each noting the planes it hands to the next piece.
pub(super) fn expand(
    ctx: *mut Ctx,
    request: &KernelRequest,
    members: &[Member],
    facts_dir: &Path,
) -> Result<Vec<Pending>, Refusal> {
    // the build's one fusion row: the device's when a target is a device, else this host's
    let row = if request.targets.contains(&Target::Device) {
        facts::device_fusion(facts_dir)
    } else {
        facts::host_fusion(facts_dir, request.facts_for(Target::Host)?)
    }
    .map_err(fault)?;
    let named: Vec<ChainMember> = members
        .iter()
        .map(|m| ChainMember {
            name: &m.name,
            sinks: &m.sinks,
        })
        .collect();
    // SAFETY: the caller's context is not read elsewhere while the pieces are made.
    let parts = cut_pieces(unsafe { &mut *ctx }, request, &named, &row)?;
    for member in members {
        carries_no_forced_access(member, "a chain member")?;
    }
    let declared = declared_by_members(members)?;
    let cut = parts.len() > 1;
    let mut units = vec![];
    for (index, piece) in parts.into_iter().enumerate() {
        let exempt = held_exempt(ctx, members, &piece.sinks, &declared)?;
        let member = Member {
            name: piece_name(&request.name, index),
            sinks: piece.sinks,
            declared: declared.clone(),
            force: None,
            exempt,
        };
        let mut unit = Pending::of(&member, None);
        unit.note = cut.then_some(Note {
            stage: StageRecord {
                kernel: member.name.clone(),
                cells_in: vec![],
                cells_out: vec![],
                team: None,
            },
            cells: vec![],
            planes: piece
                .planes
                .into_iter()
                .map(|plane| PlaneRecord {
                    name: plane.name,
                    dtype: plane.dtype.to_string(),
                    per_lane: plane.per_lane,
                    layout: Some("plane_major".into()),
                    keys: None,
                    per_target: None,
                    once: false,
                    fill: None,
                })
                .collect(),
            keyed: vec![],
            source: None,
        });
        units.push(unit);
    }
    Ok(units)
}
