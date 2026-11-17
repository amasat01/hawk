//! The fused step: lanes written one after the other in one body, each in a scope of its own, as one unit. The unit's walk binds
//! every slot each lane binds, so that the fused entry has the signature of all its lanes.
use super::request::{DeclaredSlot, Member};
use super::{canonical, carries_no_forced_access, held_exempt, merge_slot, typed_walk, Pending};
use aether_writers::contract::types::FusedLane;
use aether_writers::refuse::{not_written, Refusal};
use raptor_ir::record::Ctx;

/// The slots the lane's own walk binds, as slots the fused walk declares.
fn bound_slots(ctx: *mut Ctx, lane: &Member) -> Result<Vec<DeclaredSlot>, Refusal> {
    let (walk, _) = canonical(ctx, lane)?;
    Ok(typed_walk(ctx, walk)?
        .slots
        .into_iter()
        .map(|slot| DeclaredSlot {
            role: slot.role,
            name: slot.name,
            dtype: slot.dtype,
            shape: slot.shape,
            tag: None,
        })
        .collect())
}

/// The pending unit of fused lanes: the lanes' sinks over one walk that declares what each lane binds, and the lanes as the
/// unit's form.
pub(super) fn expand(ctx: *mut Ctx, lanes: &[Member], name: &str) -> Result<Pending, Refusal> {
    let mut declared: Vec<DeclaredSlot> = vec![];
    for lane in lanes {
        carries_no_forced_access(lane, "a fused lane")?;
        for slot in bound_slots(ctx, lane)? {
            merge_slot(&mut declared, slot)
                .map_err(|why| not_written(format!("fused lanes: {why}")))?;
        }
    }
    let sinks: Vec<u32> = lanes.iter().flat_map(|lane| lane.sinks.clone()).collect();
    let exempt = held_exempt(ctx, lanes, &sinks, &declared)?;
    let merged = Member {
        name: name.to_string(),
        sinks,
        declared,
        force: None,
        exempt,
    };
    let mut unit = Pending::of(&merged, None);
    unit.boxable = false;
    unit.lanes = Some(
        lanes
            .iter()
            .map(|lane| FusedLane {
                name: lane.name.clone(),
                sinks: lane.sinks.clone(),
            })
            .collect(),
    );
    Ok(unit)
}
