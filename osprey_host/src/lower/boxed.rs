//! The thread box of a kernel: whether its plain entry is written with each thread serving several lanes along a non-fastest
//! axis, and the box's shape. The IR facts (which axis, whether the kernel's control and commits are lane-local) come from the
//! wing analysis; the lanes per thread come from the build card; the request states what rules a box out.
use super::request::KernelRequest;
use aether_writers::contract::types::{Role, SinkPolicy, ThreadBox, Walk};
use aether_writers::refuse::{fault, Refusal};
use raptor_ir::record::Ctx;
use raptor_wings::boxing::choose_box;

/// The thread box for the kernel whose sinks are `sinks` in `ctx`, or none. A compensated kind, a scalar type the card has no
/// entry for and a walk that reads a terminated flag are written without a box.
pub(super) fn thread_box(
    ctx: *mut Ctx,
    request: &KernelRequest,
    sinks: &[u32],
    walk: &Walk,
    box_lanes: &[(String, i64)],
) -> Result<Option<ThreadBox>, Refusal> {
    let scalar = format!("{:?}", request.scalar).to_lowercase();
    let Some(&(_, card_lanes)) = box_lanes.iter().find(|(name, _)| *name == scalar) else {
        return Ok(None);
    };
    let compensated = !matches!(request.kind.sink, SinkPolicy::Plain);
    if compensated || walk.slots.iter().any(|slot| slot.role == Role::Terminated) {
        return Ok(None);
    }
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let context = unsafe { ctx.as_ref() }.ok_or_else(|| fault("a null context"))?;
    let chosen = choose_box(context, sinks, request.group, card_lanes)
        .map_err(|why| fault(format!("thread box: {why:?}")))?;
    Ok(chosen.map(|found| ThreadBox {
        axis: Some(found.axis as i64),
        t: found.lanes,
        stride: found.stride,
    }))
}
