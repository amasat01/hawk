//! The function table of a unit: the functions its walk calls, each after the functions its body calls, as the writers take them.
use super::request::DeclaredSlot;
use super::{declared_leaf, fault, typed_walk, walk_of, walk_refusal, DTYPES};
use aether_writers::contract::types::{FunctionParam, FunctionUnit, Role};
use aether_writers::refuse::{nib, Refusal};
use raptor_ir::canon::Declared;
use raptor_ir::dump::refs;
use raptor_ir::record::Ctx;
use raptor_ir::value::Val;
use std::collections::HashSet;

/// The `(name, digest)` a node names as its function when it is a call.
fn callee(c: &Ctx, h: u32) -> Option<(String, String)> {
    if !c.table.is_a(c.nodes[h as usize].class, "Call") {
        return None;
    }
    match c.field(h, "function").ok()? {
        Val::Tuple(pair) => match pair.as_slice() {
            [Val::Str(name), Val::Str(digest)] => Some((name.clone(), digest.clone())),
            _ => None,
        },
        _ => None,
    }
}

/// The calls among the nodes reached from `roots` through operands and through the nodes a node's fields hold (a loop reaches its
/// body that way), each once.
fn calls_under(c: &Ctx, roots: &[u32]) -> Vec<u32> {
    let (mut seen, mut stack, mut calls) = (HashSet::new(), roots.to_vec(), vec![]);
    while let Some(h) = stack.pop() {
        if !seen.insert(h) {
            continue;
        }
        if callee(c, h).is_some() {
            calls.push(h);
        }
        let node = &c.nodes[h as usize];
        stack.extend(node.ops.iter().rev());
        node.fields.iter().for_each(|v| refs(v, &mut stack));
    }
    calls
}

/// The nodes a field holds, none when it holds none.
fn nodes_of(c: &Ctx, h: u32, name: &str) -> Result<Vec<u32>, Refusal> {
    let held = c.field(h, name).map_err(|e| fault(format!("{e:?}")))?;
    Ok(match held {
        Val::Node(n) => vec![n],
        Val::Tuple(items) | Val::List(items) => items
            .iter()
            .filter_map(|v| if let Val::Node(n) = v { Some(*n) } else { None })
            .collect(),
        _ => vec![],
    })
}

/// The result plane a function's body walk stores its value into: no text is ever written for it, the frame ends the body in
/// `return <value>;` instead.
const RESULT_PLANE: &str = "hawk_result";

/// A function the unit calls: the node, its digest and the root of its traced body.
struct Called {
    /// The `Function` node.
    function: u32,
    /// The digest the calls name it by.
    digest: String,
    /// The root of the traced body.
    body: u32,
}

/// Adds the function a call names, after the functions its body calls.
fn add(c: &Ctx, call: u32, done: &mut Vec<Called>) -> Result<(), Refusal> {
    let Some((name, digest)) = callee(c, call) else {
        return Ok(());
    };
    let function = c.function_named(&name, &digest).ok_or_else(|| {
        fault(format!(
            "a call of the function {name:?} that the table does not hold"
        ))
    })?;
    if done.iter().any(|d| d.function == function) {
        return Ok(());
    }
    let Some(&body) = nodes_of(c, function, "body")?.first() else {
        return Err(nib(format!("the native function {name:?}")));
    };
    for inner in calls_under(c, &[body]) {
        add(c, inner, done)?;
    }
    done.push(Called {
        function,
        digest,
        body,
    });
    Ok(())
}

/// The function as the writers take it: its parameters, and a walk of its own over a result sink that stores the body's value.
fn unit_of(ctx: *mut Ctx, called: &Called) -> Result<FunctionUnit, Refusal> {
    // SAFETY: `ctx` is the live context the caller holds; nothing else reads it during the call.
    let c = unsafe { &mut *ctx };
    let name = c
        .str_field(called.function, "name")
        .map_err(|e| fault(format!("{e:?}")))?;
    let params = nodes_of(c, called.function, "params")?
        .into_iter()
        .map(|p| {
            let node = &c.nodes[p as usize];
            Ok(FunctionParam {
                name: c
                    .str_field(p, "name")
                    .map_err(|e| fault(format!("{e:?}")))?,
                dtype: node.dtype.clone(),
                shape: node.shape.clone(),
                node: p,
            })
        })
        .collect::<Result<_, Refusal>>()?;
    let result = &c.nodes[called.function as usize];
    let (dtype, shape) = (result.dtype.clone(), result.shape.clone());
    let sink = c
        .new_node(
            "Assign",
            &format!("(('name', '{RESULT_PLANE}'), ('value', #0))"),
            &[called.body],
        )
        .map_err(|e| fault(format!("{e:?}")))?;
    let slot = DeclaredSlot {
        role: Role::Mutable,
        name: RESULT_PLANE.to_string(),
        dtype: *DTYPES
            .iter()
            .find(|d| d.name() == dtype)
            .ok_or_else(|| nib(format!("a function that returns {dtype}")))?,
        shape: shape.clone(),
        tag: None,
        axes: vec![],
        zeroed: false,
        counts: None,
        winner_of: None,
    };
    let declared = Declared {
        slots: vec![declared_leaf(&slot)],
        ..Declared::default()
    };
    let walk =
        raptor_wings::cells::kernel_walk(c, &[sink], &declared, None, &[]).map_err(walk_refusal)?;
    Ok(FunctionUnit {
        name,
        digest: called.digest.clone(),
        params,
        result: dtype,
        result_shape: shape.clone(),
        walk,
        sink,
        slots: typed_walk(ctx, walk)?,
    })
}

/// The functions the walk `walk` calls, callee before caller; none when it makes no call.
pub(super) fn table(ctx: *mut Ctx, walk: u32) -> Result<Option<Vec<FunctionUnit>>, Refusal> {
    let mut done = vec![];
    {
        // SAFETY: `ctx` is the live context the caller holds; it is only read here.
        let c = unsafe { &*ctx };
        for call in calls_under(c, &walk_of(ctx, walk)?.order) {
            add(c, call, &mut done)?;
        }
    }
    let units = done
        .iter()
        .map(|called| unit_of(ctx, called))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((!units.is_empty()).then_some(units))
}
