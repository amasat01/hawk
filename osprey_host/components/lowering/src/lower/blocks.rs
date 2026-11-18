//! The block sums of a unit made plain sums when the build's linear algebra form does not keep them for a team of threads: the IR
//! rewrite and its decision are the wing's; this module runs it over each unit's body and its single step with the deciding card's limits.
use super::request::{KernelRequest, Member};
use super::Cards;
use super::Pending;
use aether_writers::refuse::{bad, fault, Refusal};
use raptor_ir::record::{Ctx, Fault};
use raptor_wings::blocks::{plain_blocks, BlockLimits};

/// The step: every unit not cut from a segmented input has its block sums made plain.
pub(super) fn plain(
    ctx: *mut Ctx,
    request: &KernelRequest,
    cards: &Cards,
    pending: Vec<Pending>,
) -> Result<Vec<Pending>, Refusal> {
    let limits = BlockLimits {
        rolled_fold_min: cards.deciding.rolled_fold_min,
        team_max_dim: cards.deciding.linalg_team_max_dim,
    };
    let form = request.settings.linalg_form;
    let mut out = vec![];
    for mut unit in pending {
        if unit.segment.is_none() {
            plain_member(ctx, &mut unit.member, form, &limits)?;
            if let Some(step) = unit.one_step.as_mut() {
                plain_member(ctx, step, form, &limits)?;
            }
        }
        out.push(unit);
    }
    Ok(out)
}

/// One body with its block sums made plain.
fn plain_member(
    ctx: *mut Ctx,
    member: &mut Member,
    form: raptor_wings::blocks::LinalgForm,
    limits: &BlockLimits,
) -> Result<(), Refusal> {
    // SAFETY: `ctx` is the live context the caller holds; nothing else reads it while the sinks are rewritten.
    let context = unsafe { &mut *ctx };
    match plain_blocks(context, &member.sinks, form, limits) {
        Ok(Some(sinks)) => member.sinks = sinks,
        Ok(None) => {}
        Err(Fault::Refused(reason)) => return Err(bad(reason)),
        Err(other) => return Err(fault(format!("block sums: {other:?}"))),
    }
    Ok(())
}
