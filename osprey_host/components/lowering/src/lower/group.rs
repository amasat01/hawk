//! The group of a unit: the record the unit description states for a walk whose root is a `Group`. The record is filled from the IR's own facts of the group
//! (membership, scratch declarations, phase count); the phases and the scratch are never worked out again here.
use super::meta_fill::Parts;
use super::real_bytes;
use super::unit_meta as m;
use aether_writers::contract::types::Role;
use aether_writers::refuse::{bad, fault, not_written, Refusal};
use aether_writers::unit::group::hoisted_planes;
use raptor_ir::record::Ctx;
use raptor_ir::value::{text_val, Val};

/// The `Group` node among a body's sinks, if it has one.
pub(super) fn group_of(ctx: &Ctx, sinks: &[u32]) -> Option<u32> {
    sinks
        .iter()
        .copied()
        .find(|&sink| ctx.table.is_a(ctx.nodes[sink as usize].class, "Group"))
}

/// A fact of the group node, or the fault that the IR does not state it.
fn fact(ctx: &Ctx, group: u32, name: &str) -> Result<Val, Refusal> {
    ctx.field(group, name)
        .map_err(|why| fault(format!("a group without its fact {name}: {why:?}")))
}

/// A whole number the IR states, or the fault that it states something else.
fn whole(value: &Val, what: &str) -> Result<i64, Refusal> {
    match value {
        Val::Int(n) => i64::try_from(*n).map_err(|_| fault(format!("{what} {n} is out of range"))),
        other => Err(fault(format!(
            "{what} is not a whole number: {}",
            text_val(other)
        ))),
    }
}

/// The extent of an axis the IR states: a whole number, or `None` for one known only at run time.
fn extent(value: &Val, what: &str) -> Result<Option<i64>, Refusal> {
    match value {
        Val::None => Ok(None),
        other => whole(other, what).map(Some),
    }
}

/// The text items of a tuple the IR states.
fn labels(value: &Val, what: &str) -> Result<Vec<String>, Refusal> {
    let Val::Tuple(items) = value else {
        return Err(fault(format!("{what} is not a tuple")));
    };
    items
        .iter()
        .map(|item| match item {
            Val::Str(label) => Ok(label.clone()),
            _ => Err(fault(format!("{what} holds a value that is not a text"))),
        })
        .collect()
}

/// The (label, extent) pairs of an axis list the IR states.
fn axis_pairs(value: &Val, what: &str) -> Result<Vec<(String, Option<i64>)>, Refusal> {
    let Val::Tuple(items) = value else {
        return Err(fault(format!("{what} is not a tuple")));
    };
    items
        .iter()
        .map(|item| match item {
            Val::Tuple(pair) => match pair.as_slice() {
                [Val::Str(label), size] => Ok((label.clone(), extent(size, what)?)),
                _ => Err(fault(format!(
                    "{what} holds a pair that is not (label, extent)"
                ))),
            },
            _ => Err(fault(format!("{what} holds a value that is not a pair"))),
        })
        .collect()
}

/// The wire type of a scratch scalar and its width in bytes: a float is the unit's real, a whole number keeps its own width.
fn scalar_of(dtype: &str, parts: &Parts, name: &str) -> Result<(m::WireDtype, i64), Refusal> {
    match dtype {
        "f64" | "f32" => Ok((m::WireDtype::Real, real_bytes(parts.request.scalar))),
        "i64" => Ok((m::WireDtype::Int64, 8)),
        "i32" => Ok((m::WireDtype::Int32, 4)),
        "bool" => Ok((m::WireDtype::Bool, 1)),
        other => Err(not_written(format!(
            "scratch '{name}' of type {other}: a scratch of this type is not written"
        ))),
    }
}

/// The scratch planes the group declares, from the IR's declarations: each plane's elements per group are the extent of its one
/// axis `rank` times the scalars of one value.
fn scratch_planes(ctx: &Ctx, group: u32, parts: &Parts) -> Result<Vec<m::GroupScratch>, Refusal> {
    let Val::Tuple(declared) = fact(ctx, group, "scratch_decls")? else {
        return Err(fault("the scratch declarations of a group are not a tuple"));
    };
    declared
        .iter()
        .map(|decl| {
            let Val::Tuple(parts_of) = decl else {
                return Err(fault("a scratch declaration is not a tuple"));
            };
            let [Val::Str(name), Val::Str(dtype), Val::Tuple(shape), axes] = parts_of.as_slice()
            else {
                return Err(fault(
                    "a scratch declaration is not (name, type, shape, axes)",
                ));
            };
            let along = axis_pairs(axes, "the axes of a scratch")?;
            let [(_, Some(rank))] = along.as_slice() else {
                return Err(fault(format!("scratch '{name}' has no single stated axis")));
            };
            let shape: Vec<i64> = shape
                .iter()
                .map(|d| whole(d, "a scratch extent"))
                .collect::<Result<_, _>>()?;
            let (wire, width) = scalar_of(dtype, parts, name)?;
            let elements = rank * shape.iter().product::<i64>();
            Ok(m::GroupScratch {
                dtype: wire,
                shape,
                elements,
                bytes: elements * width,
                name: name.clone(),
            })
        })
        .collect()
}

/// The indices, in the launch's lane axes, of the axes an `axes` group spans, and the lanes of one group (`None`: an extent known
/// only at run time). The launch must state each spanned axis with the extent the group states.
fn spanned(ctx: &Ctx, group: u32, parts: &Parts) -> Result<(Vec<usize>, Option<i64>), Refusal> {
    let named = labels(&fact(ctx, group, "axes")?, "the axes of a group")?;
    let grid = axis_pairs(&fact(ctx, group, "grid")?, "the grid of a group")?;
    let launch = &parts.request.lane_space;
    let kernel = &parts.pending.member.name;
    let (mut at, mut lanes) = (vec![], Some(1i64));
    for label in &named {
        let stated = grid
            .iter()
            .find(|(l, _)| l == label)
            .map(|(_, extent)| *extent)
            .ok_or_else(|| {
                fault(format!(
                    "the group names the axis '{label}' outside its grid"
                ))
            })?;
        let index = launch
            .iter()
            .position(|axis| axis.label.as_deref() == Some(label))
            .ok_or_else(|| {
                bad(format!(
                    "{kernel:?}: the group spans the lane axis '{label}', which the launch does not state"
                ))
            })?;
        if launch[index].extent != stated {
            return Err(bad(format!(
                "{kernel:?}: the group states the lane axis '{label}' with extent {stated:?}, the launch with {:?}",
                launch[index].extent
            )));
        }
        at.push(index);
        lanes = lanes.zip(stated).map(|(product, n)| product * n);
    }
    Ok((at, lanes))
}

/// The slot of the plane a `key` group reads its per-lane key from: the key must be a plane of the lanes, not a computed value.
fn key_slot(ctx: &Ctx, key: u32, parts: &Parts) -> Result<usize, Refusal> {
    let kernel = &parts.pending.member.name;
    let node = &ctx.nodes[key as usize];
    let is_plane = ctx.table.is_a(node.class, "Leaf")
        && matches!(ctx.field(key, "role"), Ok(Val::Str(role)) if role == "plane_in");
    let name = match ctx.field(key, "name") {
        Ok(Val::Str(name)) if is_plane => name,
        _ => {
            return Err(not_written(format!(
                "{kernel:?}: a group by a key that is computed is not written; state the key as a per-lane plane"
            )))
        }
    };
    parts
        .typed
        .slots
        .iter()
        .position(|slot| slot.role == Role::PlaneIn && slot.name == name)
        .ok_or_else(|| {
            fault(format!(
                "the key plane '{name}' of a group is not a slot of the walk"
            ))
        })
}

/// The group record of the unit, or `None` for a unit whose walk has no group.
pub(super) fn record(parts: &Parts) -> Result<Option<m::Group>, Refusal> {
    // SAFETY: `ctx` is the live context the caller holds; it is only read here.
    let ctx = unsafe { &*parts.ctx };
    let Some(group) = group_of(ctx, &parts.pending.member.sinks) else {
        return Ok(None);
    };
    let Val::Str(kind) = fact(ctx, group, "membership")? else {
        return Err(fault("the membership of a group is not a text"));
    };
    let (membership, count, key, axes, bound) = match kind.as_str() {
        "count" => {
            let lanes = whole(&fact(ctx, group, "count")?, "the lanes of a count group")?;
            (
                m::GroupMembership::Count,
                Some(lanes),
                None,
                vec![],
                Some(lanes),
            )
        }
        "axes" => {
            let (at, lanes) = spanned(ctx, group, parts)?;
            (m::GroupMembership::Axes, None, None, at, lanes)
        }
        "key" => {
            let Val::Node(key) = fact(ctx, group, "key")? else {
                return Err(fault("a key group states no key"));
            };
            let bound = extent(
                &fact(ctx, group, "max_size")?,
                "the size bound of a key group",
            )?;
            let slot = key_slot(ctx, key, parts)?;
            (m::GroupMembership::Key, None, Some(slot), vec![], bound)
        }
        other => return Err(fault(format!("a group of membership {other}"))),
    };
    let scratch = scratch_planes(ctx, group, parts)?;
    let distinct_slots = distinct_slots(parts, group)?;
    Ok(Some(m::Group {
        membership,
        phases: whole(
            &fact(ctx, group, "phase_count")?,
            "the phase count of a group",
        )?,
        key_slot: key,
        count,
        max_size: bound,
        scratch_bytes: scratch.iter().map(|plane| plane.bytes).sum(),
        axes,
        scratch,
        distinct_slots,
    }))
}

/// The slots of the planes the host text reaches through restrict pointers, from the writers' own decision
/// (`aether_writers::unit::group::hoisted_planes`), in slot order. A written plane is the slot of that name that the unit writes.
fn distinct_slots(parts: &Parts, group: u32) -> Result<Vec<usize>, Refusal> {
    let hoisted = hoisted_planes(parts.ctx.cast(), &parts.pending.member.sinks, group)?;
    let mut slots = vec![];
    for plane in &hoisted {
        let found = parts.typed.slots.iter().position(|slot| {
            slot.name == plane.name
                && if plane.written {
                    matches!(slot.role, Role::Mutable | Role::PlaneOut | Role::Out)
                } else {
                    slot.role == Role::PlaneIn
                }
        });
        slots.push(found.ok_or_else(|| {
            fault(format!(
                "the plane '{}' the host text reaches through a pointer is not a slot of the walk",
                plane.name
            ))
        })?);
    }
    slots.sort_unstable();
    Ok(slots)
}

#[cfg(test)]
mod tests {
    //! The record of the three reference kernels, whose documents the IR stores, and of a key group; the text of a group on each backend.
    //!
    //! Oracle for the values: the comments of the reference constructors (`raptor_support::groups`): `tile` and `ring` are groups of 8
    //! lanes over one scratch `t` of 8 doubles, two phases each; `rows` spans the axis `col` of 4 lanes, has no scratch and two phases.
    use super::super::request::{Body, KernelRequest};
    use super::super::{canonical, lower, typed_walk, Cards, Pending};
    use super::*;
    use crate::facts::CardSet;
    use crate::lower::request::{LaneAxis, Member};
    use crate::testing::{load_cards, shipped, TestDevice, TestHost, ALL_CARDS};
    use aether_writers::contract::types::{Scalar, Target};
    use aether_writers::refuse::RefusalKind;
    use raptor_ir::door::{rx_ctx_free, rx_ctx_new, RX_OK};
    use raptor_ir::dump::rx_load;

    /// The stored documents of the reference kernels; the one root of each is its group.
    const TILE: &str = include_str!("../../../../../raptor/rs/support/tests/data/groups/tile.dump");
    const ROWS: &str = include_str!("../../../../../raptor/rs/support/tests/data/groups/rows.dump");
    const RING: &str = include_str!("../../../../../raptor/rs/support/tests/data/groups/ring.dump");
    const LANE_OUT: &str =
        include_str!("../../../../../raptor/rs/support/tests/data/groups/lane_out.dump");
    const LANE_VECTOR_OUT: &str =
        include_str!("../../../../../raptor/rs/support/tests/data/groups/lane_vector_out.dump");

    /// `tile` with its element type changed to f32 is not stored; a key group is `tile`'s document with a per-lane key plane `k` and
    /// a stated bound of 8 in place of the count: the key leaf is node 9, the group node 10.
    fn keyed(dump: &str) -> String {
        let key = "n|Leaf|i32|()|None|()|(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', 'k'), ('quantity', None), ('wire_index', 0), ('sorted', False), ('axes', ()))\n";
        let group = "n|Group|bool|()|None|(#9,)|(('membership', 'key'), ('count', None), ('grid', ()), ('axes', ()), ('max_size', 8), ('scratch', (#1,)), ('body_sinks', (#4, #8)))\nroot|#10\n";
        let kept: Vec<&str> = dump
            .lines()
            .filter(|l| !l.contains("|Group|") && !l.starts_with("root|"))
            .collect();
        format!("{}\n{key}{group}", kept.join("\n"))
    }

    /// The context holding `dump` with the wings' kinds declared, and its one root.
    fn loaded(dump: &str) -> (*mut Ctx, u32) {
        let ctx = rx_ctx_new();
        // SAFETY: `ctx` is the live context just made.
        assert_eq!(
            unsafe { raptor_wings::declare::rw_declare_kinds(ctx) },
            RX_OK
        );
        let (mut roots, mut found) = ([0u32; 8], 0);
        // SAFETY: `ctx` is live; the text and the buffers outlive the call.
        let status = unsafe {
            rx_load(
                ctx,
                dump.as_ptr().cast(),
                dump.len(),
                roots.as_mut_ptr(),
                roots.len(),
                &mut found,
            )
        };
        assert_eq!((status, found), (RX_OK, 1));
        (ctx, roots[0])
    }

    /// The facts answer of a backend under the shipped cards.
    fn facts_of(target: Target) -> String {
        match target {
            Target::Host => crate::testing::host_facts(&[]),
            Target::Device => crate::testing::device_facts(None, &[]),
        }
    }

    /// A request for the kernel `name` whose one sink is `group`, over the lane axes `lanes` (label, extent), on `target`.
    fn request_for(
        name: &str,
        group: u32,
        lanes: &[(&str, Option<i64>)],
        target: Target,
    ) -> KernelRequest {
        let mut request = KernelRequest::from_json(&format!(
            r#"{{"v": 1, "name": "{name}", "scalar": "Float64", "targets": ["Host"], "kind": {{"guard": "Default", "active_set": false, "sink": "Plain"}}, "body": {{"Plain": {{"name": "{name}", "sinks": [0], "declared": [], "exempt": []}}}}, "facts": {{"Host": "{{}}"}}}}"#
        ))
        .unwrap();
        request.targets = vec![target];
        request.image_format = (target == Target::Device).then(|| "Blob".to_string());
        request.facts = vec![(target, facts_of(target))];
        request.lane_space = lanes
            .iter()
            .map(|(label, extent)| LaneAxis {
                label: Some((*label).into()),
                extent: *extent,
            })
            .collect();
        request.body = Body::Plain(Member {
            name: name.into(),
            sinks: vec![group],
            declared: vec![],
            force: None,
            exempt: vec![],
        });
        request
    }

    /// A group record with the (role, name) of each slot of its walk.
    type Described = (Option<m::Group>, Vec<(Role, String)>);

    /// The group record the lowering states for `request`, with the slots of its walk. The unit description as a whole needs the
    /// facts of the unit's text, which a group body does not have yet, so the record is read by the function that fills it.
    fn described(ctx: *mut Ctx, request: &KernelRequest) -> Result<Described, Refusal> {
        let docs = CardSet::new(&shipped(), &request.cards, ALL_CARDS);
        let cards: Cards = load_cards(request, &docs)?;
        let Body::Plain(member) = &request.body else {
            unreachable!("a plain request")
        };
        let pending = Pending::of(member, None);
        let (walk, digest) = canonical(ctx, member)?;
        let typed = typed_walk(ctx, walk)?;
        let found = record(&Parts {
            ctx,
            request,
            card: &cards.deciding,
            image: cards.target.image(),
            artifact: cards.artifact.as_deref(),
            pending: &pending,
            walk,
            digest: &digest,
            typed: &typed,
            texts: &[],
            route: cards.contraction_route(request)?,
        })?;
        let slots = typed
            .slots
            .iter()
            .map(|s| (s.role, s.name.clone()))
            .collect();
        Ok((found, slots))
    }

    /// The group record of the stored document `dump` named `name` over `lanes`, with `scalar` as the unit's real.
    fn record_of(
        dump: &str,
        name: &str,
        lanes: &[(&str, Option<i64>)],
        scalar: Scalar,
    ) -> m::Group {
        let (ctx, group) = loaded(dump);
        let mut request = request_for(name, group, lanes, Target::Host);
        request.scalar = scalar;
        let found = described(ctx, &request).unwrap().0;
        // SAFETY: `ctx` is the live context made above and is not used after.
        unsafe { rx_ctx_free(ctx) };
        found.expect("a unit with a group states its record")
    }

    /// The plane `t` of 8 doubles, 64 bytes, that `tile` and `ring` declare.
    fn plane_t(dtype: m::WireDtype, bytes: i64) -> m::GroupScratch {
        m::GroupScratch {
            dtype,
            shape: vec![],
            elements: 8,
            bytes,
            name: "t".into(),
        }
    }

    #[test]
    fn tile_is_a_count_group_of_8_with_one_scratch_of_64_bytes_and_two_phases() {
        let want = m::Group {
            membership: m::GroupMembership::Count,
            phases: 2,
            count: Some(8),
            max_size: Some(8),
            scratch_bytes: 64,
            scratch: vec![plane_t(m::WireDtype::Real, 64)],
            // the input plane x and the output plane y of the reference kernel, each read or written at the lane's own element
            distinct_slots: vec![0, 1],
            ..m::Group::default()
        };
        assert_eq!(record_of(TILE, "tile", &[], Scalar::Float64), want);
    }

    #[test]
    fn ring_is_a_count_group_of_8_with_the_same_scratch_and_two_phases() {
        let found = record_of(RING, "ring", &[], Scalar::Float64);
        assert_eq!(
            (
                found.membership,
                found.count,
                found.phases,
                found.scratch_bytes
            ),
            (m::GroupMembership::Count, Some(8), 2, 64)
        );
        assert_eq!(found.scratch, vec![plane_t(m::WireDtype::Real, 64)]);
    }

    #[test]
    fn rows_spans_the_axis_col_of_4_lanes_with_no_scratch_and_two_phases() {
        let want = m::Group {
            membership: m::GroupMembership::Axes,
            phases: 2,
            max_size: Some(4),
            axes: vec![1],
            // all four planes of the reference kernel are read or written at the lane's own element
            distinct_slots: vec![0, 1, 2, 3],
            ..m::Group::default()
        };
        let lanes = [("row", None), ("col", Some(4))];
        assert_eq!(record_of(ROWS, "rows", &lanes, Scalar::Float64), want);
    }

    #[test]
    fn a_single_precision_unit_holds_its_real_scratch_in_4_bytes_per_value() {
        let found = record_of(TILE, "tile", &[], Scalar::Float32);
        assert_eq!(found.scratch, vec![plane_t(m::WireDtype::Real, 32)]);
        assert_eq!(found.scratch_bytes, 32);
    }

    #[test]
    fn a_key_group_names_the_slot_of_its_key_plane_and_its_stated_bound() {
        let (ctx, group) = loaded(&keyed(TILE));
        let request = request_for("keyed", group, &[], Target::Host);
        let (found, slots) = described(ctx, &request).unwrap();
        // SAFETY: `ctx` is the live context made above and is not used after.
        unsafe { rx_ctx_free(ctx) };
        let found = found.expect("a record");
        assert_eq!(
            (found.membership, found.max_size, found.count),
            (m::GroupMembership::Key, Some(8), None)
        );
        let key = found.key_slot.expect("a key slot");
        assert_eq!(slots[key], (Role::PlaneIn, "k".to_string()));
        assert_eq!(found.scratch_bytes, 64);
    }

    #[test]
    fn an_axes_group_over_an_axis_the_launch_does_not_state_or_states_otherwise_is_refused_by_name()
    {
        for (lanes, said) in [
            (
                vec![("row", None)],
                "the group spans the lane axis 'col', which the launch does not state",
            ),
            (
                vec![("row", None), ("col", Some(5))],
                "the group states the lane axis 'col' with extent Some(4), the launch with Some(5)",
            ),
        ] {
            let (ctx, group) = loaded(ROWS);
            let why =
                described(ctx, &request_for("rows", group, &lanes, Target::Host)).unwrap_err();
            // SAFETY: `ctx` is the live context made above and is not used after.
            unsafe { rx_ctx_free(ctx) };
            assert_eq!(why.kind, RefusalKind::BadRequest);
            assert_eq!(why.reason, format!("\"rows\": {said}"));
        }
    }

    #[test]
    fn asking_for_the_text_of_a_group_is_refused_by_name_where_it_is_not_written() {
        // `tile` holds a group reduce: each backend's writer refuses it by name
        for (target, stem) in [
            (
                Target::Host,
                "\"tile\": a group reduce is not written on the host yet",
            ),
            (
                Target::Device,
                "\"tile\": a group reduce is not written on the device yet",
            ),
        ] {
            let (ctx, group) = loaded(TILE);
            let request = request_for("tile", group, &[], target);
            let json = request.to_json();
            let why = match target {
                Target::Host => lower(&TestHost, ctx, &json, &shipped()),
                Target::Device => lower(&TestDevice, ctx, &json, &shipped()),
            }
            .map(|_| ())
            .expect_err("a refusal");
            // SAFETY: `ctx` is the live context made above and is not used after.
            unsafe { rx_ctx_free(ctx) };
            assert_eq!(
                why.kind,
                RefusalKind::NotWritten,
                "{target:?}: {}",
                why.reason
            );
            assert_eq!(why.reason, stem, "{target:?}");
        }
    }

    #[test]
    fn the_device_text_of_a_group_is_a_block_per_group_with_strided_phases() {
        let (ctx, group) = loaded(RING);
        let request = request_for("ring", group, &[], Target::Device);
        let answer = lower(&TestDevice, ctx, &request.to_json(), &shipped());
        // SAFETY: `ctx` is the live context made above and is not used after.
        unsafe { rx_ctx_free(ctx) };
        let answer = answer.expect("the device text of ring");
        let text = &answer.units[0].texts[0].text;
        assert!(
            text.contains("extern __shared__ __align__(8) char hawk_scratch[];"),
            "{text}"
        );
        assert!(
            text.contains("hawk_g = base + static_cast<std::int64_t>(blockIdx.x)"),
            "{text}"
        );
        assert!(text.contains("hawk_rank += blockDim.x"), "{text}");
        assert!(text.contains("__syncthreads();"), "{text}");
    }

    #[test]
    fn the_host_text_of_a_group_without_a_reduce_is_written_with_its_record() {
        let (ctx, group) = loaded(RING);
        let request = request_for("ring", group, &[], Target::Host);
        let answer = lower(&TestHost, ctx, &request.to_json(), &shipped());
        // SAFETY: `ctx` is the live context made above and is not used after.
        unsafe { rx_ctx_free(ctx) };
        let answer = answer.expect("the host text of ring");
        let text = &answer.units[0].texts[0].text;
        assert!(text.contains("for (std::int64_t hawk_g = base;"), "{text}");
        // the phase that reads the next lane's element is cut: an interior loop with plain indices and the edge lanes beside it
        assert!(
            text.contains("hawk_rank < hawk_hi1; ++hawk_rank) {"),
            "{text}"
        );
        assert!(text.contains("scr_t[hawk_rank + 1]"), "{text}");
        assert!(
            text.contains("scr_t[aether::edge::wrap(hawk_rank + 1, "),
            "{text}"
        );
        assert!(
            !text.lines().any(|l| l.contains("scr_t") && l.contains('%')),
            "no remainder in a scratch index: {text}"
        );
        assert_eq!(
            answer.units[0].meta.group.as_ref().map(|g| g.phases),
            Some(2)
        );
    }

    #[test]
    fn the_per_lane_outputs_of_a_group_lower_up_to_the_group_reduce_they_read() {
        // both documents commit an ordinary output (an `Assign`) of a value that holds a group reduce, which the host does not
        // write yet: the refusal names the reduce, not the output
        for (name, dump) in [("lane_out", LANE_OUT), ("lane_vector_out", LANE_VECTOR_OUT)] {
            let (ctx, group) = loaded(dump);
            let request = request_for(name, group, &[], Target::Host);
            let why = lower(&TestHost, ctx, &request.to_json(), &shipped())
                .map(|_| ())
                .expect_err("a refusal");
            // SAFETY: `ctx` is the live context made above and is not used after.
            unsafe { rx_ctx_free(ctx) };
            assert_eq!(
                why.reason,
                format!("\"{name}\": a group reduce is not written on the host yet")
            );
        }
    }
}
