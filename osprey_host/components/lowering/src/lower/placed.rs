//! The placement of wide values: a kernel for a device, or whose loops the host runs interchanged, keeps each wide value that
//! needs a plane in a scratch plane, so the lane rows hold what a loop carries. The IR rewrite is the wing's; this module decides
//! when it runs, with which numbers of the deciding card, and how a two-path kernel's single step shares the planes of the fused
//! body.
use super::card::BuildNumbers;
use super::real_bytes;
use super::request::{KernelRequest, Member, WideForm};
use super::{canonical, leaf_slots, Pending};
use aether_writers::door;
use aether_writers::ir::analysis::interchanged_loops;
use aether_writers::refuse::{bad, fault, Refusal};
use aether_writers::RxCtx;
use raptor_ir::door::RX_OK;
use raptor_ir::record::Ctx;
use raptor_wings::place::rw_wide_place;
use std::ffi::{c_int, CString};

/// Whether the host runs the member's loops interchanged, which is when its wide values are placed in scratch planes.
pub(super) fn interchanged(ctx: *mut Ctx, member: &Member) -> Result<bool, Refusal> {
    let (walk, _) = canonical(ctx, member)?;
    Ok(!interchanged_loops(ctx as *mut RxCtx, walk)?.is_empty())
}

/// The unit's body placed, and a two-path unit's single step placed beside it in planes of its own; whether anything changed.
pub(super) fn place_unit(
    ctx: *mut Ctx,
    request: &KernelRequest,
    card: &BuildNumbers,
    unit: &mut Pending,
) -> Result<bool, Refusal> {
    let (walk, _) = canonical(ctx, &unit.member)?;
    let held: Vec<String> = leaf_slots(ctx, walk)?.into_iter().map(|s| s.name).collect();
    let mut changed = false;
    if let Some(sinks) = place_sinks(ctx, request, card, &unit.member.sinks, &[])? {
        unit.member.sinks = sinks;
        changed = true;
    }
    if unit.one_step.is_some() {
        let (walk, _) = canonical(ctx, &unit.member)?;
        let made: Vec<String> = leaf_slots(ctx, walk)?
            .into_iter()
            .map(|slot| slot.name)
            .filter(|name| !held.contains(name))
            .collect();
        let step = unit
            .one_step
            .as_mut()
            .ok_or_else(|| fault("no single step"))?;
        if let Some(sinks) = place_sinks(ctx, request, card, &step.sinks, &made)? {
            step.sinks = sinks;
            changed = true;
        }
    }
    if changed {
        share_planes(ctx, unit)?;
    }
    Ok(changed)
}

/// Declares on the unit's body every argument its single step binds that the body does not, so both bodies read the planes of
/// both.
fn share_planes(ctx: *mut Ctx, unit: &mut Pending) -> Result<(), Refusal> {
    let Some(step) = &unit.one_step else {
        return Ok(());
    };
    let (own, _) = canonical(ctx, &unit.member)?;
    let (other, _) = canonical(ctx, step)?;
    let held = leaf_slots(ctx, own)?;
    for slot in leaf_slots(ctx, other)? {
        if !held
            .iter()
            .any(|h| (h.role, &h.name) == (slot.role, &slot.name))
        {
            unit.member.declared.push(slot);
        }
    }
    Ok(())
}

/// The sinks with each wide value that needs a plane placed, or none when the pass leaves them as they are. `taken` are the plane
/// names it must not reuse.
pub(super) fn place_sinks(
    ctx: *mut Ctx,
    request: &KernelRequest,
    card: &BuildNumbers,
    sinks: &[u32],
    taken: &[String],
) -> Result<Option<Vec<u32>>, Refusal> {
    let limits = card
        .wide_limits(real_bytes(request.scalar))
        .ok_or_else(|| fault("the build card has no wide thresholds for the scalar"))?;
    let text = |text: String| CString::new(text).map_err(|_| bad("a NUL in a plane name"));
    let form = text(format!("{:?}", request.settings.fold_form).to_lowercase())?;
    let taken = text(taken.join("\n"))?;
    let thread_form = (request.settings.wide_form == WideForm::Thread) as c_int;
    let (mut made, mut count, mut changed) = (vec![0u32; sinks.len()], 0usize, 0 as c_int);
    // SAFETY: `ctx` is the caller's live context; the buffers are sized as the header states.
    let status = unsafe {
        rw_wide_place(
            ctx,
            sinks.as_ptr(),
            sinks.len(),
            thread_form,
            limits.runtime_min.as_ptr(),
            limits.runtime_min.len(),
            limits.wide_min,
            &card.rolled_fold_min,
            form.as_ptr(),
            taken.as_ptr(),
            made.as_mut_ptr(),
            made.len(),
            &mut count,
            &mut changed,
        )
    };
    if status != RX_OK {
        return Err(door::failed(ctx as *mut RxCtx, status));
    }
    made.truncate(count);
    Ok((changed != 0).then_some(made))
}
