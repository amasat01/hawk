//! The segmented split of a kernel: a kernel with one dispatch that splits by branch lowers to one unit per branch, each launched
//! over one run of the segmented input, and (for a selector the kernel never writes) a last unit that writes each sample's class.
//! The IR of the units comes from the wing analysis; this module turns each branch into a pending unit for the write step. A kernel
//! with one automatic dispatch whose selector no sink writes gets the same units beside its fused unit when its request asks.
use super::request::{Body, DeclaredSlot, KernelRequest, Member};
use super::{canonical, finished_slot, walk_of, wing, Cards, Pending};
use aether_writers::contract::types::{Access, Dtype, Segment};
use aether_writers::refuse::{fault, not_written, Refusal};
use aether_writers::spell::binding_name;
use raptor_ir::record::Ctx;
use raptor_ir::value::{ttype_val, Val};
use raptor_wings::segment::{
    auto_dispatch, branch_sinks, class_sinks, group_name, offsets_name, segmented_dispatch,
    SegmentedDispatch,
};

/// The plane of the fused-steps word, which the access class does not read.
pub(super) const FUSED_STEPS_PLANE: &str = raptor_wings::cells::FUSED_STEPS_PLANE;

/// Whether the selector of the walk's dispatch number `index` is one no sink of the kernel writes (the walk's own listing).
fn selector_is_static(ctx: *mut Ctx, walk: u32, index: usize) -> Result<bool, Refusal> {
    walk_of(ctx, walk)?
        .static_selector
        .get(index)
        .copied()
        .ok_or_else(|| fault("a dispatch listing without the dispatch"))
}

/// The dispatch group a branch unit belongs to.
#[derive(Clone)]
pub(crate) struct Group {
    /// The group's name.
    pub name: String,
    /// The unit that writes the classes, for a selector the kernel never writes.
    pub classify: Option<String>,
    /// The group's number of branches.
    pub k: usize,
    /// The branch units in selector-value order; none on the record of the classify unit.
    pub branches: Vec<String>,
    /// The unit that sweeps the stale samples, for a selector the kernel writes.
    pub sweep: Option<String>,
}

/// The segmented split as a step: a kernel with a segmented dispatch becomes one pending unit per branch in order, and (for a
/// selector the kernel never writes) a last unit that writes the classes. A kernel whose request asks for the segmented set of its
/// automatic group (`auto_segmented_set`) keeps its fused unit, which then names those units, and gets them after it. Any other
/// unit passes on.
pub(super) fn split(
    ctx: *mut Ctx,
    request: &KernelRequest,
    _cards: &Cards,
    pending: Vec<Pending>,
) -> Result<Vec<Pending>, Refusal> {
    let mut units = vec![];
    for unit in pending {
        if let Some((branches, _)) = branches(ctx, request, &unit.member, Find::Segmented)? {
            units.extend(branches);
            continue;
        }
        let asked = request.settings.auto_segmented_set && matches!(request.body, Body::Plain(_));
        let found = match asked {
            true => branches(ctx, request, &unit.member, Find::Auto)?,
            false => None,
        };
        match found {
            Some((branches, group)) => {
                units.push(Pending {
                    auto_set: Some(group),
                    ..unit
                });
                units.extend(branches);
            }
            None => units.push(unit),
        }
    }
    Ok(units)
}

/// Which dispatch of a walk a split is made of.
#[derive(Clone, Copy, PartialEq)]
enum Find {
    /// The dispatch pinned to the segmented strategy: its units replace the kernel's.
    Segmented,
    /// The one automatic dispatch: its units come beside the fused kernel.
    Auto,
}

/// The segmented dispatch of a body and whether its selector is one no sink of the kernel writes, or none when the body has no
/// segmented dispatch.
fn split_of(
    ctx: *mut Ctx,
    member: &Member,
    find: Find,
) -> Result<Option<(SegmentedDispatch, bool)>, Refusal> {
    let bare = Member {
        declared: vec![],
        force: None,
        exempt: vec![FUSED_STEPS_PLANE.to_string()],
        ..member.clone()
    };
    let (walk, _) = canonical(ctx, &bare)?;
    // SAFETY: `ctx` is the live context the caller holds.
    let context = unsafe { &*ctx };
    let found = match find {
        Find::Segmented => segmented_dispatch(context, walk),
        Find::Auto => auto_dispatch(context, walk),
    };
    let Some(split) = found.map_err(wing)? else {
        return Ok(None);
    };
    let fixed = selector_is_static(ctx, walk, split.index)?;
    // an automatic group gets units only when nothing writes its selector: the sweep of a moving selector is the segmented kernel's own
    Ok(Some((split, fixed)).filter(|(_, fixed)| *fixed || find == Find::Segmented))
}

/// The units a split lowers to and the group they form, or none when the kernel has no dispatch of the kind asked for.
fn branches(
    ctx: *mut Ctx,
    request: &KernelRequest,
    member: &Member,
    find: Find,
) -> Result<Option<(Vec<Pending>, Group)>, Refusal> {
    let Some((split, fixed)) = split_of(ctx, member, find)? else {
        return Ok(None);
    };
    if !fixed && !matches!(request.body, Body::Plain(_)) {
        return Err(not_written(format!(
            "kernel {:?}: the sweep unit of a segmented group with a dynamic selector in a chain or a fused kernel",
            request.name
        )));
    }
    let group = group_name(split.index);
    let offsets = offsets_name(&group);
    let classify = format!("{}__cls", request.name);
    let ours = Group {
        name: group.clone(),
        classify: fixed.then(|| classify.clone()),
        k: split.branches,
        branches: (0..split.branches)
            .map(|branch| format!("{}__k{branch}", request.name))
            .collect(),
        sweep: (!fixed).then(|| sweep_name(&request.name)),
    };
    let mut units = vec![];
    for branch in 0..split.branches {
        // SAFETY: the caller's context is not read elsewhere while the units are made.
        let made = branch_sinks(unsafe { &mut *ctx }, &member.sinks, &split, branch, !fixed);
        let segment = Segment {
            offsets_binding: binding_name("lookup", 0, &offsets)?,
            j: branch as i64,
        };
        let name = ours.branches[branch].clone();
        units.push(Pending {
            member: unit_member(ctx, name, made.map_err(wing)?, Some((&offsets, !fixed))),
            segment: Some(segment),
            one_step: None,
            active: None,
            lanes: None,
            boxable: false,
            note: None,
            team: None,
            warp: None,
            group: Some(ours.clone()),
            auto_set: None,
            exports_block: true,
        });
    }
    if fixed {
        // SAFETY: the caller's context is not read elsewhere while the unit is made.
        let sinks = class_sinks(unsafe { &mut *ctx }, &split).map_err(wing)?;
        units.push(Pending {
            member: unit_member(ctx, classify, sinks, None),
            segment: None,
            one_step: None,
            active: None,
            lanes: None,
            boxable: false,
            note: None,
            team: None,
            warp: None,
            group: Some(Group {
                classify: None,
                branches: vec![],
                sweep: None,
                ..ours.clone()
            }),
            auto_set: None,
            exports_block: true,
        });
    }
    Ok(Some((units, ours)))
}

/// The name of the sweep unit of the kernel `name`.
fn sweep_name(name: &str) -> String {
    format!("{name}__sweep")
}

/// The request of the sweep unit of a segmented group whose selector the kernel writes: the same kernel with its dispatch lowered
/// as a switch and its kind's guard an active set, which the lowering then writes like any such kernel. None for any other kernel.
pub(super) fn sweep_request(
    ctx: *mut Ctx,
    request: &KernelRequest,
) -> Result<Option<KernelRequest>, Refusal> {
    let Body::Plain(member) = &request.body else {
        return Ok(None);
    };
    let Some((split, false)) = split_of(ctx, member, Find::Segmented)? else {
        return Ok(None);
    };
    // SAFETY: the caller's context is not read elsewhere while the sweep's IR is made.
    let context = unsafe { &mut *ctx };
    let node = &context.nodes[split.node as usize];
    let operands = node.ops.clone();
    let tt = ttype_val(&node.dtype, &node.shape, node.tag.as_deref());
    let class = context
        .table
        .find("Dispatch")
        .ok_or_else(|| fault("no plain dispatch class"))?;
    let branches = operands[1..].iter().map(|&b| Val::Node(b)).collect();
    let given = [
        ("selector".to_string(), Val::Node(operands[0])),
        ("branches".to_string(), Val::Tuple(branches)),
        ("ttype".to_string(), tt),
    ];
    let switch = context.new_from(class, &given).map_err(wing)?;
    let swapped = [(split.node, switch)].into_iter().collect();
    let sinks = context.substitute(&member.sinks, &swapped).map_err(wing)?;
    let name = sweep_name(&request.name);
    let mut twin = request.clone();
    twin.name = name.clone();
    twin.one_step = None;
    twin.kind.active_set = true;
    twin.kind.runtime_set = true;
    twin.body = Body::Plain(Member {
        name,
        sinks,
        ..member.clone()
    });
    Ok(Some(twin))
}

/// The body of one split unit: its walk declares the offsets plane for a branch unit (floored to the cross-sample access class
/// when the branch is guarded), and the finished-count plane for a finishing one.
fn unit_member(
    ctx: *mut Ctx,
    name: String,
    sinks: Vec<u32>,
    branch: Option<(&str, bool)>,
) -> Member {
    let mut member = Member {
        name,
        sinks,
        declared: branch
            .iter()
            .map(|(offsets, _)| DeclaredSlot::lookup(offsets, Dtype::I64))
            .collect(),
        force: branch
            .filter(|(_, guarded)| *guarded)
            .map(|_| Access::CrossSampleWrite),
        exempt: vec![FUSED_STEPS_PLANE.to_string()],
    };
    member.declared.extend(finished_slot(ctx, &member));
    member
}

#[cfg(test)]
mod tests {
    //! The units a kernel's segmented form gets, on a kernel of one two-branch dispatch over a plane `k`: asked of an automatic group
    //! with a static selector it names the set; with a selector the kernel writes it names none, as for any other kernel.
    use super::*;
    use raptor_ir::door::{rx_ctx_free, rx_ctx_new, RX_OK};
    use raptor_ir::dump::rx_load;

    /// The dump of the kernel `y = dispatch(k; x + 1, x * 2)` with the dispatch of the given class, and (when `moving`) a second sink
    /// that writes the plane `k`, so that no sink-free selector is left. The sinks are the last roots.
    fn dump(class: &str, moving: bool) -> String {
        let mut lines = vec![
            "raptor.ir.dump/1".to_string(),
            "n|Leaf|i32|()|None|()|(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', 'k'), ('quantity', None), ('wire_index', 0), ('sorted', False))".into(),
            "n|Leaf|f64|()|None|()|(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', 'x'), ('quantity', None), ('wire_index', 0), ('sorted', False))".into(),
            "n|Op|f64|()|None|(#1, #1)|(('kind', 'add'), ('literal', None))".into(),
            "n|Op|f64|()|None|(#1, #1)|(('kind', 'mul'), ('literal', None))".into(),
            format!("n|{class}|f64|()|None|(#0, #2, #3)|()"),
            "n|Assign|f64|()|None|(#4,)|(('kind', 'assign'), ('role', 'mutable'), ('name', 'y'))".into(),
        ];
        let sinks = if moving {
            lines.push("n|Assign|i32|()|None|(#0,)|(('kind', 'assign'), ('role', 'mutable'), ('name', 'k'))".into());
            vec![5, 6]
        } else {
            vec![5]
        };
        lines.extend(sinks.iter().map(|at| format!("root|#{at}")));
        lines.join("\n") + "\n"
    }

    /// The request of a plain kernel named `k` and its member over the sinks of the dump, run by `check` with the context.
    fn with_kernel(
        class: &str,
        moving: bool,
        check: impl FnOnce(*mut Ctx, &KernelRequest, &Member),
    ) {
        let text = dump(class, moving);
        let ctx = rx_ctx_new();
        // SAFETY: `ctx` is the live context just made and nothing else holds it.
        raptor_wings::declare::declare_kinds(unsafe { &mut *ctx }).unwrap();
        let mut roots = vec![0u32; 16];
        let mut count = 0;
        // SAFETY: `ctx` is the live context just made; the text and the buffers outlive the call.
        let status = unsafe {
            rx_load(
                ctx,
                text.as_ptr().cast(),
                text.len(),
                roots.as_mut_ptr(),
                roots.len(),
                &mut count,
            )
        };
        assert_eq!(status, RX_OK, "{:?}", unsafe { (*ctx).err.clone() });
        let request = KernelRequest::from_json(
            r#"{"v": 1, "name": "k", "scalar": "Float64", "targets": ["Host"], "kind": {"guard": "Default", "active_set": false, "sink": "Plain"}, "body": {"Plain": {"name": "k", "sinks": [0], "declared": [], "exempt": []}}, "facts": {"Host": "{}"}}"#,
        )
        .unwrap();
        let member = Member {
            name: "k".into(),
            sinks: roots[count - if moving { 2 } else { 1 }..count].to_vec(),
            declared: vec![],
            force: None,
            exempt: vec![],
        };
        check(ctx, &request, &member);
        // SAFETY: the context is not used again.
        unsafe { rx_ctx_free(ctx) };
    }

    #[test]
    fn an_automatic_group_with_a_static_selector_gets_the_set_a_segmented_one_gets() {
        with_kernel("AutoDispatch", false, |ctx, request, member| {
            let (units, group) = branches(ctx, request, member, Find::Auto)
                .unwrap()
                .expect("a set");
            assert_eq!((group.name.as_str(), group.k), ("dispatch0", 2));
            assert_eq!(group.branches, ["k__k0", "k__k1"]);
            assert_eq!(group.classify.as_deref(), Some("k__cls"));
            assert!(group.sweep.is_none());
            let names: Vec<&str> = units.iter().map(|u| u.member.name.as_str()).collect();
            assert_eq!(names, ["k__k0", "k__k1", "k__cls"]);
        });
        with_kernel("SegmentedDispatch", false, |ctx, request, member| {
            let (_, group) = branches(ctx, request, member, Find::Segmented)
                .unwrap()
                .expect("a split");
            assert_eq!(group.branches, ["k__k0", "k__k1"]);
            assert!(branches(ctx, request, member, Find::Auto)
                .unwrap()
                .is_none());
        });
    }

    #[test]
    fn an_automatic_group_whose_selector_the_kernel_writes_gets_no_set_and_it_is_no_error() {
        with_kernel("AutoDispatch", true, |ctx, request, member| {
            assert!(branches(ctx, request, member, Find::Auto)
                .unwrap()
                .is_none());
        });
        // the same kernel pinned to the segmented strategy is split, with the sweep unit of its moving selector
        with_kernel("SegmentedDispatch", true, |ctx, request, member| {
            let (_, group) = branches(ctx, request, member, Find::Segmented)
                .unwrap()
                .expect("a split");
            assert_eq!(group.sweep.as_deref(), Some("k__sweep"));
            assert!(group.classify.is_none());
        });
    }
}
