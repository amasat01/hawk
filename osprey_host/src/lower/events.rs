//! The events of a kernel: its stop rule, read as the comparisons it is made of. A kernel has no events interface; the value its
//! finish reads (`terminated = a or b or ...`) is flattened through `or` in operand order, and when every leaf is a comparison
//! (`<`, `<=`, `>`, `>=`) the i-th leaf is event i. A stop rule with any other leaf (an `and`, a `not`, an equality, a flag) has
//! no events. A launch takes at most `MAX_EVENTS` of them.
use super::request::{Body, KernelRequest};
use super::unit_meta::DerivedRole;
use aether_writers::refuse::{bad, Refusal};
use raptor_ir::record::Ctx;
use raptor_ir::Val;
use std::collections::HashMap;

/// The most events a unit may state (the runtime's limit).
pub const MAX_EVENTS: i64 = 8;

/// The refusal of a stop rule with more comparisons than a unit has events: the kernel and the count are named.
pub fn limit(kernel: &str, count: i64) -> Result<(), Refusal> {
    if count > MAX_EVENTS {
        return Err(bad(format!(
            "kernel {kernel}: the stop rule has {count} comparisons, and a unit has at most {MAX_EVENTS} events"
        )));
    }
    Ok(())
}

/// The kind of an operation node, `None` for a node that is not one.
fn op_kind(ctx: &Ctx, node: u32) -> Option<String> {
    let class = ctx.nodes.get(node as usize)?.class;
    if !ctx.table.is_a(class, "Op") {
        return None;
    }
    match ctx.field(node, "kind") {
        Ok(Val::Str(kind)) => Some(kind),
        _ => None,
    }
}

/// The leaves of the `or` chain under `node`, in operand order.
fn leaves(ctx: &Ctx, node: u32, out: &mut Vec<u32>) {
    if op_kind(ctx, node).as_deref() == Some("lor") {
        for &operand in &ctx.nodes[node as usize].ops {
            leaves(ctx, operand, out);
        }
    } else {
        out.push(node);
    }
}

/// The number of events of the stop rule of a kernel's sinks: the comparisons of its finish value, 0 when it has no finish or its
/// stop rule is not made of comparisons only. The kernel's own sinks are read, not a unit's walk: a split kernel's later unit
/// holds the stop rule as a plane it reads, and the comparisons are in the kernel.
pub fn count(ctx: &Ctx, sinks: &[u32]) -> i64 {
    let finish = sinks
        .iter()
        .find(|&&n| ctx.table.is_a(ctx.nodes[n as usize].class, "Finish"));
    let Some(&value) = finish.and_then(|&f| ctx.nodes[f as usize].ops.first()) else {
        return 0;
    };
    let mut all = vec![];
    leaves(ctx, value, &mut all);
    let comparison =
        |n: &u32| matches!(op_kind(ctx, *n).as_deref(), Some("lt" | "le" | "gt" | "ge"));
    if all.iter().all(comparison) {
        all.len() as i64
    } else {
        0
    }
}

/// The sinks of the kernel a request states, whatever its units: a split kernel's later unit holds the stop rule as a plane it
/// reads, and the comparisons are in the kernel's own sinks.
pub(super) fn request_sinks(request: &KernelRequest) -> Vec<u32> {
    match &request.body {
        Body::Plain(member) => member.sinks.clone(),
        Body::Chain(members) | Body::Fused(members) => {
            members.iter().flat_map(|m| m.sinks.clone()).collect()
        }
    }
}

/// The real operations that are constant between jumps: a smooth guard never passes through one.
const STEPWISE: [&str; 6] = ["floor", "ceil", "trunc", "round", "rint", "sign"];

/// Whether a node depends smoothly on a floating-point input: `Some(true)` when it does, `Some(false)` when it does not, and `None`
/// when a stepwise operation sits on a floating-point path (a refusal). A leaf is smooth when it is real; a constant, a sample
/// index, a plane extent and a loop index are not; neither is anything that is not real-valued (a comparison, a connective, an
/// integer). A select reads its two arms, a dispatch its branches, any other operation its operands.
pub fn smooth(ctx: &Ctx, node: u32, memo: &mut HashMap<u32, Option<bool>>) -> Option<bool> {
    if let Some(done) = memo.get(&node) {
        return *done;
    }
    let held = &ctx.nodes[node as usize];
    let is = |class: &str| ctx.table.is_a(held.class, class);
    let real = matches!(held.dtype.as_str(), "f64" | "f32");
    let mut all = |from: &[u32]| {
        let parts: Vec<Option<bool>> = from.iter().map(|&o| smooth(ctx, o, memo)).collect();
        if parts.contains(&None) {
            None
        } else {
            Some(parts.contains(&Some(true)))
        }
    };
    let out = if is("Leaf") {
        Some(real)
    } else if ["Const", "SampleIndex", "PlaneExtent", "LoopIndex"]
        .iter()
        .any(|c| is(c))
        || !real
    {
        Some(false)
    } else if is("Select") {
        all(held.ops.get(1..3).unwrap_or(&[]))
    } else if is("Dispatch") {
        all(held.ops.get(1..).unwrap_or(&[]))
    } else if op_kind(ctx, node).is_some_and(|k| STEPWISE.contains(&k.as_str())) {
        match held.ops.first().map(|&o| smooth(ctx, o, memo)) {
            Some(Some(true)) | Some(None) => None,
            _ => Some(false),
        }
    } else {
        all(&held.ops)
    };
    memo.insert(node, out);
    out
}

/// The comparisons of a kernel's stop rule and why the first one that cannot be an event of a refine run cannot: a leaf of the `or`
/// chain that is not a comparison, a comparison on a stepwise operation, or one whose two sides carry no smooth dependence on a
/// real input (a count, a rounding, a sign, a class test). `Ok` holds the number of leaves; a kernel with no finish has no stop rule.
pub fn refine_verdict(ctx: &Ctx, sinks: &[u32]) -> Result<usize, (usize, &'static str)> {
    let finish = sinks
        .iter()
        .find(|&&n| ctx.table.is_a(ctx.nodes[n as usize].class, "Finish"));
    let Some(&value) = finish.and_then(|&f| ctx.nodes[f as usize].ops.first()) else {
        return Err((0, "the kernel has no stop rule"));
    };
    let mut all = vec![];
    leaves(ctx, value, &mut all);
    let mut memo = HashMap::new();
    for (at, &leaf) in all.iter().enumerate() {
        if !matches!(
            op_kind(ctx, leaf).as_deref(),
            Some("lt" | "le" | "gt" | "ge")
        ) {
            return Err((at, "it is not a comparison (<, <=, >, >=)"));
        }
        let sides: Vec<Option<bool>> = ctx.nodes[leaf as usize]
            .ops
            .iter()
            .map(|&o| smooth(ctx, o, &mut memo))
            .collect();
        if sides.contains(&None) {
            return Err((at, "a comparison on a stepwise operation (floor, ceil, trunc, round, rint, sign) is not a crossing"));
        }
        if !sides.contains(&Some(true)) {
            return Err((at, "its two sides carry no smooth dependence on a real input (a count, a rounding, a sign or a class test)"));
        }
    }
    Ok(all.len())
}

/// Whether a role is one of a refine run.
fn refine_role(role: DerivedRole) -> bool {
    use DerivedRole::{RefineEntry, RefineNewton, RefineVjp, StateGuard, StateGuardVjp};
    matches!(
        role,
        RefineEntry | RefineNewton | RefineVjp | StateGuard | StateGuardVjp
    )
}

/// Refuses a request that asks for a refine unit of a kernel whose stop rule cannot be refined: the kernel, the event index and the
/// reason are named. A request with no refine unit is not looked at.
pub fn check_refine(ctx: &Ctx, request: &KernelRequest) -> Result<(), Refusal> {
    if !request.derived.iter().any(|d| refine_role(d.role)) {
        return Ok(());
    }
    refine_verdict(ctx, &request_sinks(request))
        .map(|_| ())
        .map_err(|(event, why)| {
            bad(format!(
                "kernel {}: event {event} cannot be refined: {why}",
                request.name
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nine_comparisons_are_refused_naming_the_kernel_and_the_count() {
        assert!(limit("k", 8).is_ok());
        let why = limit("k", 9).unwrap_err().reason;
        assert!(
            why.contains("kernel k") && why.contains("9 comparisons"),
            "{why}"
        );
    }
}
