//! The events of a kernel: its stop rule, read as the comparisons it is made of. A kernel has no events interface; the value its
//! finish reads (`terminated = a or b or ...`) is flattened through `or` in operand order, and when every leaf is a comparison
//! (`<`, `<=`, `>`, `>=`) the i-th leaf is event i. A stop rule with any other leaf (an `and`, a `not`, an equality, a flag) has
//! no events. A launch takes at most `MAX_EVENTS` of them.
use aether_writers::refuse::{bad, Refusal};
use raptor_ir::record::Ctx;
use raptor_ir::Val;

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
