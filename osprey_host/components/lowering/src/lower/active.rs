//! The active set of a kernel: when the kind reads an index map (a thinning guard or a set of lanes) the unit's walk also binds
//! the map plane and the count plane, and the unit carries the binding of each so the launch runs only the listed samples.
use super::request::{DeclaredSlot, IndexSet, KernelRequest};
use super::{canonical, leaf_slots, Cards, Pending};
use aether_writers::contract::types::{Active, Dtype, Guard, MapBits, Role};
use aether_writers::refuse::{bad, not_written, Refusal};
use aether_writers::spell::binding_name;
use raptor_ir::record::Ctx;

/// The plane of the index map.
pub(super) const MAP_PLANE: &str = "active_map";
/// The plane that holds the number of listed samples.
pub(super) const COUNT_PLANE: &str = "active_count";

/// The roles a walk's leaves are re-declared with: a written output and a row sum's companion are not leaves.
const LEAF_ROLES: [Role; 9] = [
    Role::VecIn,
    Role::MatIn,
    Role::WideIn,
    Role::PerSample,
    Role::Lookup,
    Role::Terminated,
    Role::Uniform,
    Role::Nsamples,
    Role::Mutable,
];

/// The step: every unit of a kernel whose kind reads an index map gets the map and count planes declared and its active set.
pub(super) fn set(
    ctx: *mut Ctx,
    request: &KernelRequest,
    _cards: &Cards,
    pending: Vec<Pending>,
) -> Result<Vec<Pending>, Refusal> {
    if !request.kind.active_set && request.kind.lanes.is_none() {
        return Ok(pending);
    }
    pending
        .into_iter()
        .map(|unit| with_map(ctx, request, unit))
        .collect()
}

/// The refusal of a kernel that accumulates across samples under an active set: a construct this build does not write.
fn accumulates(name: &str) -> Refusal {
    not_written(format!(
        "{name:?}: an active set refuses a kernel that accumulates across samples"
    ))
}

/// One unit with its walk widened by the map and count planes. A segmented unit, a unit that accumulates across samples under a
/// thinning guard, a thinning guard a caller asks for none of whose masks the unit binds (a set the runtime fills follows no mask)
/// and a unit that already uses a reserved plane name are refused.
fn with_map(ctx: *mut Ctx, request: &KernelRequest, mut unit: Pending) -> Result<Pending, Refusal> {
    let name = &request.name;
    let what = if request.kind.lanes.is_some() {
        "a lane set"
    } else {
        "an active set"
    };
    if unit.segment.is_some() {
        return Err(bad(format!(
            "{name:?}: {what} cannot read an index map in a segmented unit"
        )));
    }
    // A caller that declares the map and count planes itself has them re-declared below.
    unit.member
        .declared
        .retain(|slot| slot.name != MAP_PLANE && slot.name != COUNT_PLANE);
    let (walk, _) = canonical(ctx, &unit.member)?;
    let leaves = leaf_slots(ctx, walk)?;
    let lane_set = request.kind.lanes;
    if lane_set.is_none() {
        let default = ["terminated".to_string()];
        let masks: &[String] = match &request.kind.guard {
            Guard::Masks(masks) => masks,
            Guard::Default => &default,
            Guard::DataOnly => &[],
        };
        if leaves.iter().any(|slot| slot.role == Role::AccumOut) {
            return Err(accumulates(name));
        }
        let binds = |mask: &String| {
            leaves
                .iter()
                .any(|s| s.role == Role::Terminated && &s.name == mask)
        };
        // the mask rule belongs to a set a caller asks for, which is built from the caller's mask and follows it; a set the runtime
        // fills lists the live samples whatever they are, and a kernel with no mask has every sample live
        if !request.kind.runtime_set && !masks.iter().any(binds) {
            return Err(bad(format!(
                "{name:?}: an active set needs the kernel to bind the mask the map follows"
            )));
        }
    }
    let IndexSet { bits, prefix } = lane_set.unwrap_or(IndexSet {
        bits: MapBits::B32,
        prefix: false,
    });
    let reserved: &[&str] = if prefix {
        &[COUNT_PLANE]
    } else {
        &[MAP_PLANE, COUNT_PLANE]
    };
    if let Some(taken) = leaves
        .iter()
        .find(|slot| reserved.contains(&slot.name.as_str()))
    {
        return Err(bad(format!(
            "{name:?}: {:?} is reserved for the index map under {what}; rename that parameter",
            taken.name
        )));
    }
    let mut declared: Vec<DeclaredSlot> = leaves
        .into_iter()
        .filter(|slot| LEAF_ROLES.contains(&slot.role))
        .collect();
    if !prefix {
        let word = if bits == MapBits::B64 {
            Dtype::I64
        } else {
            Dtype::Int32
        };
        declared.push(DeclaredSlot::lookup(MAP_PLANE, word));
    }
    declared.push(DeclaredSlot::lookup(COUNT_PLANE, Dtype::Uint32));
    // The companions the caller declared (a compensated total's low word) follow the map planes.
    declared.extend(
        unit.member
            .declared
            .iter()
            .filter(|slot| !LEAF_ROLES.contains(&slot.role))
            .cloned(),
    );
    unit.member.declared = declared;
    unit.active = Some(Active {
        map_binding: if prefix {
            None
        } else {
            Some(binding_name("lookup", MAP_PLANE)?)
        },
        count_binding: binding_name("lookup", COUNT_PLANE)?,
        group: request.group,
        map_bits: if prefix { MapBits::B32 } else { bits },
        lane_set: lane_set.is_some() && (prefix || bits == MapBits::B32),
    });
    Ok(unit)
}

#[cfg(test)]
mod tests {
    //! The refusals of an active set are of the kind of what they refuse.
    use super::*;
    use aether_writers::refuse::RefusalKind;

    #[test]
    fn a_kernel_that_accumulates_under_an_active_set_is_a_construct_not_written() {
        let why = accumulates("k");
        assert_eq!(why.kind, RefusalKind::NotWritten);
        assert!(!why.reason.contains("Guard("), "{}", why.reason);
    }
}
